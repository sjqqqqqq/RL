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
- **TN/** — work-in-progress reimplementation of QMPS using **Julia (ITensorMPS) for the MPS contractions** and **PyTorch for the agent head + optimizer**, glued by `juliacall`. This is the active development surface.

## TN/ architecture (the main area)

The non-obvious thing about TN/ is the **two-language split** and how gradients cross it.

```
TN/julia/QMPSRL.jl            ← ITensorMPS sim + QMPS feature contraction + Zygote VJP
TN/python/bridge.py           ← juliacall wrapper + torch.autograd.Function (QMPSOverlap)
TN/python/qmps_agent.py       ← nn.Module: QMPS-feature → MLP head → Q-values
TN/python/train.py            ← DDQN loop with Polyak target
TN/python/replay.py           ← (Transition, ReplayBuffer)
```

Key patterns to preserve when editing:

1. **State handles, not arrays.** Julia owns the MPS objects in a `Dict{Int, MPS}` registry (`QMPSRL._registry`); Python only sees integer IDs. `JuliaEnv.step(a)` returns `(next_id, r, done, fid)`. Never try to round-trip an MPS through Python.
2. **Custom autograd boundary.** `bridge.QMPSOverlap` / `QMPSOverlapBatch` is the **only** place gradients cross the Julia↔Python line. Forward calls `qmps_feature_and_vjp(_batch)`, stores the closure in `ctx.grad_fn_jl`, and backward invokes it. If you add a new differentiable Julia function, mirror this pattern — do not call Zygote from Python directly.
3. **Param sync direction.** PyTorch owns the canonical `qmps_params` (an `nn.Parameter`); each forward pushes it into Julia via `set_qmps_params` before calling the contraction. The Julia-side params are scratch state, not source of truth.
4. **Julia 1-indexed actions.** `bridge.JuliaEnv.step` does `int(action) + 1` before calling Julia. Python keeps 0-indexed actions everywhere else.
5. **Import order.** `bridge_paper.py` must be imported **before** `torch` to avoid juliacall/torch init issues (see comment in `train_paper.py:26`). Replicate this if you add new entry points.

### `_paper` vs non-`_paper` variants

There are parallel triplets: `QMPSRL.jl` / `QMPSRL_paper.jl`, `bridge.py` / `bridge_paper.py`, `qmps_agent.py` / `qmps_agent_paper.py`, `train.py` / `train_paper.py`. The `_paper` variants match Metz & Bukov 2023 Study A defaults (24-action → 12-action set with asymmetric `δt₊≠δt₋`, `D_F=32`, hard target-net copy every N grad steps, 40k episodes, single-threaded BLAS for cluster packing). When changing one, decide explicitly whether the change applies to both — they share filenames intentionally but diverge in hyperparameters and action sets.

### Constants come from Julia

`bridge.N`, `bridge.N_ACTIONS`, `bridge.D_F`, `bridge.N_PARAMS_REAL` are read from the Julia module at import time. Change them in `QMPSRL.jl`, not in Python. Editing `N`, `CHI_Q`, or `D_F` invalidates any saved checkpoints because `N_PARAMS_REAL` changes.

## Running things

The repo is multi-shell (Windows PowerShell + Bash via WSL); commands below use bash-style paths.

**TN training (active):**
```bash
cd TN/python
python train.py --episodes 5000           # exploratory defaults
python train_paper.py                     # paper Study A defaults (40k ep)
python train_paper.py --episodes 1000 --out-prefix smoke    # quick smoke test
```
Outputs (`*_learning_curve.png`, `*_trajectory.png`, `*_loss.png`, optional `*_greedy_eval.png`) land in `TN/`, not in `TN/python/`. Greedy eval after training: `--greedy-eval 100`.

Julia env must be the project at `TN/julia/`. The bridge sets `PYTHON_JULIACALL_PROJECT` to that path; first run will instantiate. `PYTHON_JULIACALL_EXE` defaults to `/usr/local/bin/julia` — override it via the same env var if your Julia is elsewhere (especially on Windows).

**Julia-only test:** `julia --project=TN/julia TN/julia/test_qmps.jl`

**Published QMPS code:**
```bash
cd QMPS/dqn && python main.py             # writes results/ in CWD
```

**Lattice scripts:** each is a top-level script — `python Lattice/deepQ_target.py`. Requires the missing `.mat` files; flag this to the user before running.

## Conventions worth knowing

- No tests, no linter, no CI configured. "Does training step without erroring and the learning curve plot looks sensible" is the verification bar.
- The `Lattice/` scripts are intentionally redundant single-file experiments; do not refactor them into shared modules unless asked.
- Action-index encoding lives in `decode_action` in the Julia module; if you change the action set, update both `N_ACTIONS` and `decode_action` together, and check that `bridge.JuliaEnv.step`'s +1 offset is still correct.
