"""Torch-vs-Julia parity test for the QMPS feature contraction.

The training gradient path runs entirely in torch (`qmps_torch`); Julia only
provides the forward reference `QMPSRL.qmps_feature`. This test pins the two
together so the torch port can't silently drift from the reference:

  1. forward parity  — qmps_torch.feature == QMPSRL.qmps_feature (both complex128)
  2. backward sanity — torch.autograd grad matches central finite differences of
     the torch feature (validates the autograd path that replaced Zygote).

Run (set PYTHON_JULIACALL_EXE if julia isn't on the default path):
    python test_parity.py
or under pytest:
    pytest test_parity.py
"""
from __future__ import annotations

import numpy as np
import torch

import bridge as B           # imports juliacall — must precede torch
from bridge import jl
import qmps_torch as QT

# complex128 / float64 so the parity tolerance is set by the contraction, not
# by single-precision round-off.
_DT = torch.complex128


def _make_state_ids() -> list[int]:
    """A few fresh ground states plus one stepped (entangled) trajectory."""
    ids = [B.JuliaEnv(seed=s).state_id for s in (1, 2, 3)]
    env = B.JuliaEnv(seed=7)
    s = env.state_id
    for a in (0, 4, 3, 2, 5, 6):
        s, _, _, _ = env.step(a)
    ids.append(s)
    return ids


def _julia_feats(ids) -> np.ndarray:
    """(B, D_F) reference features from the Julia forward contraction."""
    return np.stack(
        [np.asarray(jl.QMPSRL.qmps_feature(int(i)), dtype=np.float64) for i in ids],
        axis=0,
    )


def test_forward_parity() -> float:
    ids = _make_state_ids()
    flat = B.get_qmps_params().to(torch.float64)
    jl_feats = _julia_feats(ids)
    tf = QT.feature_from_ids(ids, flat, device=None, dtype=_DT).detach().numpy()
    err = float(np.abs(jl_feats - tf).max())
    # Julia sweeps the chain sequentially; the torch port does left/right-sweep
    # einsums + a center combine. Same math, different float64 associativity, so
    # expect ~1e-8 round-off at L=32 — not bit-identity.
    assert err < 1e-6, f"torch vs julia feature mismatch: max err {err:.3e}"
    return err


def test_autograd_vs_finite_diff() -> float:
    ids = _make_state_ids()
    # Stack the (constant) state tensors once; only the params vary across the
    # finite-difference evaluations.
    arrays_list = [B.state_arrays(int(i)) for i in ids]
    psi = QT.stack_states(arrays_list, device=None, dtype=_DT)

    base = B.get_qmps_params().to(torch.float64)
    gdir = torch.randn(len(ids), B.D_F, dtype=torch.float64,
                       generator=torch.Generator().manual_seed(0))

    def scalar(f: torch.Tensor) -> torch.Tensor:
        # L(params) = Σ gdir ⊙ feature(states, params); ∇L is what we check.
        return (QT.feature(psi, f) * gdir).sum()

    fp = base.clone().requires_grad_(True)
    scalar(fp).backward()
    g_auto = fp.grad.detach().numpy()

    eps = 1e-6
    idxs = np.random.default_rng(0).choice(base.numel(), size=12, replace=False)
    max_err = 0.0
    for k in idxs:
        fp_p = base.clone(); fp_p[k] += eps
        fp_m = base.clone(); fp_m[k] -= eps
        with torch.no_grad():
            g_num = (scalar(fp_p) - scalar(fp_m)).item() / (2 * eps)
        max_err = max(max_err, abs(g_auto[k] - g_num))
    assert max_err < 1e-4, f"autograd vs finite-diff mismatch: max err {max_err:.3e}"
    return max_err


if __name__ == "__main__":
    fe = test_forward_parity()
    ge = test_autograd_vs_finite_diff()
    print(f"forward parity   max|torch - julia| = {fe:.3e}  OK")
    print(f"autograd vs FD   max err            = {ge:.3e}  OK")
    print("PARITY OK")
