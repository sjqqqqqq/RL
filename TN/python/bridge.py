"""Python ↔ Julia bridge for QMPSRL (Paper Study B reproduction).

Boots juliacall against the TN/julia project, loads the QMPSRL module, and
exposes thin wrappers around the env API (JuliaEnv, fidelity_id, state_arrays,
registry_size, ...) plus the QMPS param read.

The differentiable QMPS contraction does NOT live here: it runs in pure torch
(`qmps_torch.py`) so gradients flow through torch.autograd. Julia owns only the
environment dynamics and the per-state MPS arrays; nothing crosses the language
boundary inside the gradient path.

Import-order note: juliacall must be imported before torch to avoid a known
init-order segfault. Importing this module (which imports juliacall first)
satisfies that constraint for downstream callers.
"""
from __future__ import annotations

import os
import pathlib

# Configure juliacall *before* import.
_TN_JULIA = pathlib.Path(__file__).resolve().parent.parent / "julia"
os.environ.setdefault("PYTHON_JULIACALL_EXE", "/usr/local/bin/julia")
os.environ.setdefault("PYTHON_JULIACALL_PROJECT", str(_TN_JULIA))

from juliacall import Main as jl  # noqa: E402
import numpy as np                # noqa: E402
import torch                      # noqa: E402

# Load the Julia module exactly once.
_loaded = False


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    jl.seval(f'include("{(_TN_JULIA / "QMPSRL.jl").as_posix()}")')
    jl.seval("using .Main.QMPSRL")
    _loaded = True


_ensure_loaded()


# ---------------------------------------------------------------------------
# Constants pulled from Julia (cached as Python ints/floats)
# ---------------------------------------------------------------------------

N: int             = int(jl.QMPSRL.N)
N_ACTIONS: int     = int(jl.QMPSRL.N_ACTIONS)
D_F: int           = int(jl.QMPSRL.D_F)
N_PARAMS_REAL: int = int(jl.QMPSRL.QMPS_NPARAMS_REAL)
F_THRESHOLD: float = float(jl.QMPSRL.F_THRESHOLD)
N_STEPS_MAX: int   = int(jl.QMPSRL.N_STEPS_MAX)


# ---------------------------------------------------------------------------
# BLAS thread pinning (N=4, χ=4 means tiny matmuls; threading is overhead)
# ---------------------------------------------------------------------------


def set_blas_threads(n: int = 1) -> None:
    """Pin Julia's BLAS thread count. Call from the Python entry-point after
    setting OMP/MKL/OPENBLAS env vars to the same value."""
    jl.seval(f"using LinearAlgebra; BLAS.set_num_threads({int(n)})")


# ---------------------------------------------------------------------------
# Env API wrappers (thin)
# ---------------------------------------------------------------------------


class JuliaEnv:
    """Mirror of QMPSRL.Env from Python."""

    def __init__(self,
                 seed: int = 0,
                 f_threshold: float = F_THRESHOLD,
                 n_steps_max: int   = N_STEPS_MAX,
                 _env=None):
        if _env is not None:
            self._env = _env
            return
        self._env = jl.QMPSRL.new_env(seed,
                                      f_threshold=f_threshold,
                                      n_steps_max=n_steps_max)

    @classmethod
    def at_gx(cls,
              gx: float,
              seed: int = 0,
              f_threshold: float = F_THRESHOLD,
              n_steps_max: int   = N_STEPS_MAX) -> "JuliaEnv":
        """Build an env whose initial state is the TFIM ground state at a
        specified gx (runs DMRG fresh; bypasses the U[GX_LO, GX_HI] cache).
        Used for OOD generalization eval."""
        jenv = jl.QMPSRL.new_env_at_gx(float(gx), int(seed),
                                       f_threshold=f_threshold,
                                       n_steps_max=n_steps_max)
        return cls(_env=jenv)

    @property
    def state_id(self) -> int:
        return int(self._env.state_id)

    @property
    def t(self) -> int:
        return int(self._env.t)

    def reset(self, seed: int) -> int:
        sid = jl.QMPSRL.reset_b(self._env, seed)   # juliacall maps `!` → `_b`
        return int(sid)

    def step(self, action: int):
        # Julia is 1-indexed but QMPSRL.decode_action does +1 internally,
        # so we pass the Python (0-indexed) action through unchanged.
        nxt, r, done, f = jl.QMPSRL.step_b(self._env, int(action))
        return int(nxt), float(r), bool(done), float(f)

    def fidelity(self) -> float:
        return float(jl.QMPSRL.fidelity(self._env))


def forget_state(state_id: int) -> None:
    jl.QMPSRL.forget_state_b(int(state_id))


def registry_size() -> int:
    return int(jl.QMPSRL.registry_size())


def clear_registry() -> None:
    jl.QMPSRL.clear_registry_b()


def fidelity_id(state_id: int) -> float:
    return float(jl.QMPSRL.fidelity_id(int(state_id)))


def half_chain_entropy_id(state_id: int) -> float:
    """Half-chain von Neumann entropy at bond N÷2 ↔ N÷2+1. Used for the
    paper csB inset diagnostic; not in the training reward path."""
    return float(jl.QMPSRL.half_chain_entropy_id(int(state_id)))


def state_arrays(state_id: int) -> list[np.ndarray]:
    """Per-site canonical (D_l, 2, D_r) complex arrays for a registered state.
    Used by the torch-side contraction to build its own padded state tensors;
    the Julia registry stays the source of truth for the MPS itself."""
    arrs = jl.QMPSRL.get_arrays(int(state_id))
    return [np.array(a, dtype=np.complex128, copy=True) for a in arrs]


# ---------------------------------------------------------------------------
# QMPS parameter sync
# ---------------------------------------------------------------------------


def get_qmps_params() -> torch.Tensor:
    """Pull the Julia-side QMPS init params out as a torch float32 tensor.

    Used once, to seed the torch `nn.Parameter`. After that PyTorch owns the
    canonical params; there is no push-back into Julia (the contraction runs in
    qmps_torch). Julia's QMPS_CHUNKS stay at their init value and are unused."""
    arr = np.asarray(jl.QMPSRL.get_qmps_params(), dtype=np.float64)
    return torch.from_numpy(arr).to(torch.float32)
