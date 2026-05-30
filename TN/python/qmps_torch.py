"""Pure-PyTorch port of the QMPS overlap contraction.

Replaces the Julia/Zygote differentiable hot path: the same batched MPS-on-MPS
overlap, expressed in torch ops so that torch.autograd computes the backward
(no Zygote, no juliacall in the gradient path) and the whole thing can move to
GPU with a device flag.

Topology constants and the param layout are pulled from the Julia module at
import so this stays in lockstep with QMPSRL.jl (the source of truth).

Param layout (must match QMPSRL._real_flat_to_chunks):
  flat real vector, interleaved [re0, im0, re1, im1, ...]; complex values are
  concatenated chunk-by-chunk in QMPS tensor order; each chunk is reshaped to
  QMPS_SHAPES[q] in COLUMN-MAJOR (Julia) order.
"""
from __future__ import annotations

import numpy as np
import torch

import bridge as B
from bridge import jl

# --- topology pulled from Julia ---------------------------------------------
N: int = B.N
D_F: int = B.D_F
QMPS_SHAPES: list[tuple[int, ...]] = [
    tuple(int(x) for x in s) for s in jl.QMPSRL.QMPS_SHAPES
]
QMPS_NUMEL: list[int] = [int(np.prod(s)) for s in QMPS_SHAPES]
# Column-major reshape, precomputed as per-chunk constants so the reshape/permute
# is torch.compile-friendly (varargs range() inside the loop confuses Dynamo).
_REV_SHAPES: list[tuple[int, ...]] = [tuple(reversed(s)) for s in QMPS_SHAPES]
_FWD_PERMS: list[tuple[int, ...]] = [
    tuple(range(len(s) - 1, -1, -1)) for s in QMPS_SHAPES
]
# state bond caps (Julia STATE_BONDS is 1-indexed length N-1)
_STATE_BONDS: list[int] = [int(x) for x in jl.QMPSRL.STATE_BONDS]
_CPY: int = N // 2  # 0-indexed central tensor among the N+1 qmps tensors

_EPS = 1e-16
# Reference feature scale (csB `scale`=4.0); pulled from Julia so it stays in
# lockstep with QMPSRL.FEATURE_SCALE. Separate from the MPS-init norm factor.
_SCALE: float = float(jl.QMPSRL.FEATURE_SCALE)


def _state_cap(p: int) -> tuple[int, int]:
    """Padded (D_l, D_r) bond caps for 0-indexed physical site p."""
    dl = 1 if p == 0 else _STATE_BONDS[p - 1]
    dr = 1 if p == N - 1 else _STATE_BONDS[p]
    return dl, dr


def params_to_tensors(flat_real: torch.Tensor) -> list[torch.Tensor]:
    """(N_PARAMS_REAL,) real flat -> list of complex QMPS tensors (QMPS_SHAPES).

    Mirrors QMPSRL._real_flat_to_chunks + per-chunk column-major reshape.
    """
    f = flat_real.reshape(-1, 2)
    allc = torch.complex(f[:, 0], f[:, 1])  # (n_complex,)
    tensors: list[torch.Tensor] = []
    off = 0
    for i, numel in enumerate(QMPS_NUMEL):
        chunk = allc[off:off + numel]
        # Julia column-major reshape: row-major reshape to reversed shape, then
        # reverse the axes (perm precomputed in _REV_SHAPES/_FWD_PERMS).
        t = chunk.reshape(_REV_SHAPES[i]).permute(_FWD_PERMS[i]).contiguous()
        tensors.append(t)
        off += numel
    return tensors


def stack_states(arrays_list: list[list[np.ndarray]],
                 device=None, dtype=torch.complex128) -> list[torch.Tensor]:
    """Pad/conj/stack per-state site arrays into batched site tensors.

    arrays_list[b][p] : numpy (dl, 2, dr) complex for batch member b, site p.
    Returns psi[p] : torch (Dl_cap, 2, Dr_cap, B) complex, already conjugated.
    """
    Bn = len(arrays_list)
    psi: list[torch.Tensor] = []
    for p in range(N):
        dl, dr = _state_cap(p)
        T = torch.zeros((dl, 2, dr, Bn), dtype=dtype, device=device)
        for b in range(Bn):
            a = arrays_list[b][p]              # (dl_b, 2, dr_b)
            at = torch.from_numpy(np.ascontiguousarray(a)).to(dtype=dtype, device=device)
            T[:at.shape[0], :, :at.shape[2], b] = at.conj()
        psi.append(T)
    return psi


def overlap(psi: list[torch.Tensor],
            qmps: list[torch.Tensor]) -> torch.Tensor:
    """Batched complex overlap o[F, B]. psi already conjugated."""
    # ---- left sweep: physical sites 0 .. CPY-1 -> Lenv (Bc_l, Dmid, B) ------
    q0 = qmps[0]                                   # (s, B1)
    Lenv = torch.einsum('sk,dsrz->krz', q0, psi[0])      # (B1, Dr, B)
    for p in range(1, _CPY):
        Lenv = torch.einsum('edz,sef,dsrz->frz', Lenv, qmps[p], psi[p])

    # ---- right sweep: physical sites N-1 .. CPY -> Renv (Bc_r, Dmid, B) -----
    qlast = qmps[N]                                # (s, B_N)
    Renv = torch.einsum('sk,dsrz->kdz', qlast, psi[N - 1])  # (B_N, Dl, B)
    for p in range(N - 2, _CPY - 1, -1):
        Renv = torch.einsum('frz,sef,dsrz->edz', Renv, qmps[p + 1], psi[p])

    # ---- center combine: o[F, B] = Lenv . qmps_c . Renv --------------------
    qc = qmps[_CPY]                                # (Bl, F, Br)
    return torch.einsum('edz,efg,gdz->fz', Lenv, qc, Renv)   # (F, B)


def feature(psi: list[torch.Tensor], flat_real: torch.Tensor) -> torch.Tensor:
    """Returns (B, D_F) real feature = log(|o|^2 + eps)/N, autograd-connected
    to flat_real."""
    qmps = params_to_tensors(flat_real)
    o = overlap(psi, qmps)                         # (F, B) complex
    mag2 = o.real.pow(2) + o.imag.pow(2)           # (F, B) real
    feat = _SCALE * torch.log(mag2 + _EPS) / N     # (F, B)
    return feat.transpose(0, 1).contiguous()       # (B, F)


def feature_from_ids(state_ids, flat_real: torch.Tensor,
                     device=None, dtype=torch.complex128) -> torch.Tensor:
    """Convenience: fetch state arrays from Julia, stack, contract.
    Bypasses the cache; used by validation/tests."""
    arrays_list = [B.state_arrays(int(i)) for i in state_ids]
    psi = stack_states(arrays_list, device=device, dtype=dtype)
    return feature(psi, flat_real)


# ---------------------------------------------------------------------------
# Per-state tensor cache
# ---------------------------------------------------------------------------
# A state's padded site tensors are built once (fetched across juliacall +
# padded + conjugated) and reused on every replay sample. Keyed by the same
# integer handle Julia uses; eviction is driven by ReplayBuffer._decref so the
# cache tracks the Julia registry's lifetime. Device/dtype set the backend
# (flip to cuda for GPU); changing them clears the cache.

_DEVICE = None
_DTYPE = torch.complex64
_cache: dict[int, list[torch.Tensor]] = {}


def set_backend(device=None, dtype=torch.complex64) -> None:
    global _DEVICE, _DTYPE
    _DEVICE, _DTYPE = device, dtype
    _cache.clear()


def _build_psi_single(state_id: int) -> list[torch.Tensor]:
    arrs = B.state_arrays(int(state_id))
    out: list[torch.Tensor] = []
    for p in range(N):
        dl, dr = _state_cap(p)
        T = torch.zeros((dl, 2, dr), dtype=_DTYPE, device=_DEVICE)
        at = torch.from_numpy(np.ascontiguousarray(arrs[p])).to(dtype=_DTYPE, device=_DEVICE)
        T[:at.shape[0], :, :at.shape[2]] = at.conj()
        out.append(T)
    return out


def psi_for(state_id: int) -> list[torch.Tensor]:
    sid = int(state_id)
    c = _cache.get(sid)
    if c is None:
        c = _build_psi_single(sid)
        _cache[sid] = c
    return c


def psi_batch(state_ids) -> list[torch.Tensor]:
    """Stack cached per-site tensors into psi[p]: (Dl_cap, 2, Dr_cap, B)."""
    cols = [psi_for(i) for i in state_ids]
    return [torch.stack([c[p] for c in cols], dim=-1) for p in range(N)]


def forget(state_id: int) -> None:
    _cache.pop(int(state_id), None)


def feature_batch_ids(state_ids, flat_real: torch.Tensor) -> torch.Tensor:
    """(B, D_F) feature for cached state ids, autograd-connected to flat_real."""
    return feature(psi_batch(state_ids), flat_real)


def graph_feature(flat_param: torch.Tensor, sample_state_ids):
    """CUDA-graph the contraction for a fixed batch size, bound to flat_param.

    Returns a callable g(flat_param, *psi_tensors) -> (B, D_F) that replays the
    captured fwd (and bwd) graph instead of re-launching the ~60 einsum kernels.
    Eliminates the per-step launch overhead that dominates this small-tensor,
    launch-bound step. Inputs are copied into static buffers each call, so fresh
    psi/params per step are fine; in-place optimizer updates to flat_param are
    picked up on replay. Call as `g(flat_param, *psi_batch(ids))` with the same
    batch size as sample_state_ids.
    """
    psi = psi_batch(sample_state_ids)

    def _feat_flat(f, *p):
        return feature(list(p), f)

    return torch.cuda.make_graphed_callables(_feat_flat, (flat_param, *psi))
