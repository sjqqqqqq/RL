"""Python ↔ Julia bridge for QMPSRL.

Boots juliacall against the TN/julia project, loads the QMPSRL module, and
exposes:
  - thin wrappers around the env API (new_env, step, fidelity, reward, ...)
  - QMPSOverlap, a torch.autograd.Function that calls Julia's Zygote VJP

See `tn_juliacall_setup.md` memory for the env vars this expects.
"""
from __future__ import annotations

import os
import pathlib

# Configure juliacall *before* import.
_TN_JULIA = pathlib.Path(__file__).resolve().parent.parent / "julia"
os.environ.setdefault("PYTHON_JULIACALL_EXE", "/usr/local/bin/julia")
os.environ.setdefault("PYTHON_JULIACALL_PROJECT", str(_TN_JULIA))

from juliacall import Main as jl  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

# Load the Julia module exactly once.
_loaded = False


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    jl.seval(f'include("{_TN_JULIA / "QMPSRL.jl"}")')
    jl.seval("using .Main.QMPSRL")
    _loaded = True


_ensure_loaded()


# ---------------------------------------------------------------------------
# Constants pulled from Julia (cached as Python ints/floats)
# ---------------------------------------------------------------------------

N: int = int(jl.QMPSRL.N)
N_ACTIONS: int = int(jl.QMPSRL.N_ACTIONS)
D_F: int = int(jl.QMPSRL.D_F)
N_PARAMS_REAL: int = int(jl.QMPSRL.QMPS_NPARAMS_REAL)


# ---------------------------------------------------------------------------
# Env API wrappers (thin)
# ---------------------------------------------------------------------------


class JuliaEnv:
    """Mirror of QMPSRL.Env from Python."""

    def __init__(self, seed: int = 0, f_threshold: float = 0.85, n_steps_max: int = 50):
        self._env = jl.QMPSRL.new_env(seed,
                                      f_threshold=f_threshold,
                                      n_steps_max=n_steps_max)

    @property
    def state_id(self) -> int:
        return int(self._env.state_id)

    @property
    def t(self) -> int:
        return int(self._env.t)

    def reset(self, seed: int) -> int:
        sid = jl.QMPSRL.reset_b(self._env, seed)  # juliacall maps `!` → `_b`
        return int(sid)

    def step(self, action: int):
        # action is 1-indexed on the Julia side
        nxt, r, done, f = jl.QMPSRL.step_b(self._env, int(action) + 1)
        return int(nxt), float(r), bool(done), float(f)

    def fidelity(self) -> float:
        return float(jl.QMPSRL.fidelity(self._env))


def forget_state(state_id: int) -> None:
    jl.QMPSRL.forget_state_b(int(state_id))


def registry_size() -> int:
    return int(jl.QMPSRL.registry_size())


def fidelity_id(state_id: int) -> float:
    return float(jl.QMPSRL.fidelity_id(int(state_id)))


# ---------------------------------------------------------------------------
# QMPS parameter sync
# ---------------------------------------------------------------------------


def get_qmps_params() -> torch.Tensor:
    """Pull the current Julia-side QMPS params out as a torch float32 tensor."""
    arr = np.asarray(jl.QMPSRL.get_qmps_params(), dtype=np.float64)
    return torch.from_numpy(arr).to(torch.float32)


def set_qmps_params(params: torch.Tensor) -> None:
    """Push a torch tensor into Julia as the current QMPS params."""
    flat = params.detach().to(torch.float64).cpu().numpy()
    # juliacall accepts numpy float64 arrays as Vector{Float64}.
    jl.QMPSRL.set_qmps_params_b(flat)


# ---------------------------------------------------------------------------
# Custom autograd.Function for the QMPS feature
# ---------------------------------------------------------------------------


class QMPSOverlap(torch.autograd.Function):
    """Compute feature vector and route gradient back through Julia's Zygote VJP.

    Inputs:
      state_id : non-differentiable int handle into the Julia state registry
      params   : (N_PARAMS_REAL,) float32 tensor, the flat QMPS params
                 (must match what's currently set on the Julia side)
    Output:
      feat     : (D_F,) float32 tensor
    """

    @staticmethod
    def forward(ctx, state_id, params):
        # Sync params into Julia
        set_qmps_params(params)
        # Compute feature + capture VJP
        feat_jl, grad_fn = jl.QMPSRL.qmps_feature_and_vjp(int(state_id))
        feat_np = np.asarray(feat_jl, dtype=np.float64)
        ctx.grad_fn_jl = grad_fn        # keep alive for backward
        ctx.n_params = params.numel()
        return torch.from_numpy(feat_np).to(params.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output is dL/dfeat, shape (D_F,)
        g_np = grad_output.detach().to(torch.float64).cpu().numpy()
        g_params = ctx.grad_fn_jl(g_np)
        g_params_np = np.asarray(g_params, dtype=np.float64)
        return None, torch.from_numpy(g_params_np).to(grad_output.dtype)


class QMPSOverlapBatch(torch.autograd.Function):
    """Batched QMPS feature contraction.

    Inputs:
      state_ids : tuple/list/np.ndarray of B int handles (non-differentiable)
      params    : (N_PARAMS_REAL,) float32 tensor
    Output:
      feats     : (B, D_F) float32 tensor (transposed from Julia's column-major)
    """

    @staticmethod
    def forward(ctx, state_ids, params):
        set_qmps_params(params)
        ids_np = np.asarray([int(i) for i in state_ids], dtype=np.int64)
        feats_jl, grad_fn = jl.QMPSRL.qmps_feature_and_vjp_batch(ids_np)
        feats_np = np.asarray(feats_jl, dtype=np.float64)  # (D_F, B) col-major
        ctx.grad_fn_jl = grad_fn
        ctx.batch = feats_np.shape[1]
        return torch.from_numpy(feats_np.T.copy()).to(params.dtype)  # (B, D_F)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: (B, D_F). Julia expects (D_F, B).
        g_np = grad_output.detach().to(torch.float64).cpu().numpy().T.copy()
        g_params = ctx.grad_fn_jl(g_np)
        g_params_np = np.asarray(g_params, dtype=np.float64)
        return None, torch.from_numpy(g_params_np).to(grad_output.dtype)


def feature(state_id: int, params: torch.Tensor) -> torch.Tensor:
    return QMPSOverlap.apply(state_id, params)


def feature_batch(state_ids, params: torch.Tensor) -> torch.Tensor:
    """Batched feature; returns (len(state_ids), D_F)."""
    return QMPSOverlapBatch.apply(state_ids, params)
