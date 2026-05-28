# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

Three independent projects live side-by-side under one git repo. They share research themes (RL for quantum control) but have no shared code.

- **Lattice/** — standalone PyTorch DQN scripts for a Bloch-lattice control task (`environment_quantum_state.py`, `environment_operator.py`). Each `*.py` is a runnable single-file experiment (`deepQ.py`, `deepQ_target.py`, `double_deepQ.py`). The envs `loadmat` `H0.mat`/`H1.mat`/`H2.mat` from `Lattice/`; **these `.mat` files are not in the repo** and must be supplied before either env will import.
- **QMPS/** — the **paper authors' published code** for Metz & Bukov 2022 (arXiv:2201.11790, also `TN/RL+TN.pdf`), JAX + TensorNetwork. Registered as a **git submodule** pointing at `https://github.com/frmetz/QMPS.git`. **Treat as read-only:** never edit, refactor, or reformat anything under `QMPS/`. We use it as a reference implementation to compare our `TN/` port against. See `QMPS/README.md`.

  Fresh clones must initialize the submodule:
  ```
  git clone --recurse-submodules <repo>      # or, after a plain clone:
  git submodule update --init
  ```
- **TN/** — our Julia+Python reimplementation of QMPS targeting **paper Study A** exactly. Julia (ITensorMPS + Zygote) owns the MPS contractions; PyTorch owns the agent head and optimizer; juliacall glues them. This is the active development surface.

## TN/ architecture (the main area)

The non-obvious thing about TN/ is the **two-language split** and how gradients cross it. One set of files, paper-matched defaults baked in:

```
TN/julia/QMPSRL.jl            ← TLFI Hamiltonian + DMRG, 12-action set, 5-tensor QMPS, Zygote VJP
TN/python/bridge.py           ← juliacall wrapper + torch.autograd.Function (QMPSOverlap)
TN/python/qmps_agent.py       ← nn.Module: QMPS-feature → MLP head → Q-values
TN/python/train.py            ← Double-DQN loop with buffer pre-fill and hard target copy
TN/python/replay.py           ← (Transition, ReplayBuffer) with ref-counted Julia-handle eviction
TN/julia/test_qmps.jl         ← Julia-side unit tests (norm preservation, DMRG, Zygote VJP)
```

Key patterns to preserve when editing:

1. **State handles, not arrays.** Julia owns the MPS objects in a `Dict{Int, RegEntry}` registry (`QMPSRL._registry`, with cached dense statevectors for cheap contractions). Python only sees integer IDs. `JuliaEnv.step(a)` returns `(next_id, r, done, fid)`. Never try to round-trip an MPS through Python.
2. **Custom autograd boundary.** `bridge.QMPSOverlap` / `QMPSOverlapBatch` is the **only** place gradients cross the Julia↔Python line. Forward calls `qmps_feature_and_vjp(_batch)`, stores the closure in `ctx.grad_fn_jl`, and backward invokes it. If you add a new differentiable Julia function, mirror this pattern — do not call Zygote from Python directly.
3. **Param sync direction.** PyTorch owns the canonical `qmps_params` (an `nn.Parameter`); each forward pushes it into Julia via `set_qmps_params` before calling the contraction. The Julia-side params are scratch state, not source of truth.
4. **Action indexing.** Python passes 0-indexed actions (0..N_ACTIONS-1) straight through. The Julia side does `ACTIONS[a + 1]` inside `decode_action`. Even index → δt+, odd → δt− (paper's parity-based step-size selector). Do not shuffle the action list without also re-checking the parity convention.
5. **Import order.** `bridge.py` imports juliacall, which must come before `torch`. Any new entry point should import `bridge` first.

### Paper Study A specifics (what these defaults encode)

- **Hamiltonian:** `H = −J·ΣZ_iZ_{i+1} − gx·ΣX_i − gz·ΣZ_i` (paper sign).
- **Initial state per episode:** ground state of `(J=1, gx∼U[1.0, 1.1], gz=0)` — near-critical TFIM, drawn fresh each `JuliaEnv(seed=...)`.
- **Target state:** ground state of `(J=0, gx=0, gz=1)` = `|↑↑↑↑⟩` (Z-basis product, +1 eigenvector of ΣZ).
- **Reward:** `log(F) / N` where `F = |⟨ψ|target⟩|²`. Termination at `F ≥ 0.995^N = 0.9801` for N=4.
- **Action set (paper order, 12 actions):** `[gx, ngx, gy, ngy, gz, ngz, Jx, nJx, Jy, nJy, Jz, nJz]` with `δt+ = π/12` and `δt− = π/17`.
- **QMPS topology:** 5 tensors for N=4 — `T1:(2,2)`, `T2:(2,2,4)`, `T3:(4,32,4)` (feature-only, no physical leg), `T4:(2,4,2)`, `T5:(2,2)`. Init: identity-near real part + N(0, 0.5²) noise.
- **DDQN:** double-DQN target, half-MSE loss, replay buffer pre-filled with 8000 random-action transitions before training, hard target-net copy every 10 gradient steps.
- **Schedule:** 4000 episodes, ε exp-decay from 1.0 → 0.01 with time constant `N_ep/8 = 500`.

### Constants come from Julia

`bridge.N`, `bridge.N_ACTIONS`, `bridge.D_F`, `bridge.N_PARAMS_REAL`, `bridge.F_THRESHOLD`, `bridge.N_STEPS_MAX` are read from the Julia module at import time. Change them in `QMPSRL.jl`, not in Python. Editing `N`, `CHI_BONDS`, or `D_F` invalidates any saved checkpoints (`N_PARAMS_REAL` changes).

## Running things

The repo is multi-shell (Windows PowerShell + Bash); commands below use bash-style paths.

**TN training (paper Study A defaults):**
```bash
cd TN/python
python train.py                                       # 4000-episode paper run
python train.py --episodes 1000 --out-prefix smoke    # quick smoke test
python train.py --greedy-eval 200 --eval-seed 9999    # add greedy eval after training
```
Outputs (`*_learning_curve.png`, `*_trajectory.png`, `*_loss.png`, `*_arrays.npz`, `*_model.pt`, optional `*_greedy_eval.{png,npz}`) land in `TN/`, not in `TN/python/`.

Julia env is the project at `TN/julia/`. `bridge.py` sets `PYTHON_JULIACALL_PROJECT` to that path; the first run will instantiate. `PYTHON_JULIACALL_EXE` defaults to `/usr/local/bin/julia` — override it via the same env var if your Julia is elsewhere (Windows: set `PYTHON_JULIACALL_EXE` to e.g. `C:\Users\you\AppData\Local\Programs\Julia-1.x.x\bin\julia.exe`).

**Julia-only test:**
```bash
julia --project=TN/julia TN/julia/test_qmps.jl
```

**Published QMPS code (reference, read-only):**
```bash
cd QMPS/dqn && python main.py             # writes results/ in CWD
```

**Lattice scripts:** each is a top-level script — `python Lattice/deepQ_target.py`. Requires the missing `.mat` files; flag this to the user before running.

## Conventions worth knowing

- No CI or linter configured. Verification bar is: `julia --project=TN/julia TN/julia/test_qmps.jl` passes, plus a short `train.py` smoke run produces a learning curve without NaNs.
- The `Lattice/` scripts are intentionally redundant single-file experiments; do not refactor them into shared modules unless asked.
- Action-index encoding lives in `QMPSRL.ACTIONS` and `QMPSRL.decode_action`; if you change the action set, update `N_ACTIONS`, the parity convention in `duration_for`, and the bridge's pass-through.
