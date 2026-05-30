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
- **TN/** — our Julia+Python reimplementation of QMPS targeting **paper Study B** exactly. Julia (ITensorMPS) owns the environment dynamics — DMRG ground states, gate application, the per-state MPS arrays. PyTorch owns *everything differentiable*: the QMPS feature contraction (ported to pure torch in `qmps_torch.py`), the agent head, and the optimizer. juliacall glues them. This is the active development surface. (We started on Study A; the migration to Study B and the move off Julia/Zygote to torch.autograd are both complete — see history below.)

## TN/ architecture (the main area)

The non-obvious thing about TN/ is the **two-language split** and where the boundary sits. Gradients no longer cross it: Julia is the environment, torch is the autograd. One set of files, paper-matched defaults baked in:

```
TN/julia/QMPSRL.jl            ← TLFI Hamiltonian + DMRG, 7-action set, QMPS forward reference, state registry
TN/python/qmps_torch.py       ← pure-torch QMPS overlap contraction (the differentiable hot path; CPU/GPU)
TN/python/bridge.py           ← juliacall wrapper: env API, state arrays, QMPS param read (no autograd here)
TN/python/qmps_agent.py       ← nn.Module: QMPS-feature → MLP head → Q-values
TN/python/train.py            ← Double-DQN loop with buffer pre-fill and hard target copy
TN/python/replay.py           ← (Transition, ReplayBuffer) with ref-counted Julia-handle + torch-cache eviction
TN/julia/test_qmps.jl         ← Julia-side unit tests (norm preservation, DMRG, forward feature)
TN/python/test_parity.py      ← torch-vs-Julia forward parity + autograd-vs-finite-diff check
```

Key patterns to preserve when editing:

1. **State handles, not arrays.** Julia owns the MPS objects in a `Dict{Int, RegEntry}` registry (`QMPSRL._registry`, with cached dense statevectors for cheap contractions). Python only sees integer IDs. `JuliaEnv.step(a)` returns `(next_id, r, done, fid)`. Never try to round-trip an MPS through Python.
2. **Gradients live entirely in torch.** The differentiable QMPS contraction is `qmps_torch.py` (a port of the Julia forward contraction into torch ops); `torch.autograd` computes the backward. **Nothing crosses the Julia↔Python line inside the gradient path** — Julia hands Python the per-state MPS arrays once (`bridge.state_arrays`, cached per handle in `qmps_torch._cache`), and torch does the rest. `QMPSRL.qmps_feature` (Julia, forward-only) is kept *solely* as the reference that `test_parity.py` checks the torch port against. If you change the contraction, change it in `qmps_torch.py` and keep the Julia reference + parity test in sync. Do **not** reintroduce a Zygote/`autograd.Function` bridge — that path (the old bottleneck) was deliberately removed.
3. **Params are torch-owned, one-way.** `qmps_agent` seeds its `nn.Parameter` once from `bridge.get_qmps_params()` (Julia's init), after which PyTorch is the sole source of truth. There is **no** push-back into Julia; `QMPSRL.QMPS_CHUNKS` stay at their init value and are unused in training.
4. **Action indexing.** Python passes 0-indexed actions (0..N_ACTIONS-1) straight through. The Julia side does `ACTIONS[a + 1]` inside `decode_action`. Even index → δt+, odd → δt− (paper's parity-based step-size selector). Do not shuffle the action list without also re-checking the parity convention.
5. **Import order.** `bridge.py` imports juliacall, which must come before `torch`. Any new entry point should import `bridge` first.

### Paper Study B specifics (what these defaults encode)

- **Hamiltonian:** `H = −J·ΣZ_iZ_{i+1} − gx·ΣX_i − gz·ΣZ_i` (paper sign).
- **Chain length:** `N = L = 32` (state MPS bond cap `D_QSTATE = 16`).
- **Initial state per episode:** ground state of `(J=+1 FM, gx∼U[1.0, 1.1], gz=0)` — near-critical TFIM, matching reference `QMPS/environment/env.py:400` (`J=1.0`). DMRG is expensive at L=32, so episodes sample from a 200-state pre-computed cache (`INIT_CACHE`, seed 12345).
- **Target state:** ground state of `(J=0, gx=0, gz=1)` = `|↑^N⟩` (Z-basis product, +1 eigenvector of ΣZ).
- **Reward:** `log(F) / N` where `F = |⟨ψ|target⟩|²`. Termination at `F ≥ 0.99^N` (i.e. `F_sp = F^{1/N} ≥ 0.99`), or at the 50-step cap.
- **QMPS feature transform:** `feat = FEATURE_SCALE · log(|o|² + ε) / N` with `FEATURE_SCALE = 4.0` (reference `scale`, `models.py:301`). This is **separate** from `INIT_NORM_FACTOR` (reference `factor`, which scales the MPS-tensor init) — both are 4.0 in csB but do different things; don't conflate.
- **Action set (paper order, 7 actions, `env.py:221` `# CS B`):** `[gy, gz, ngz, Jx, nJx, Jy, nJy]` with `δt+ = π/12` and `δt− = π/17`, selected by action-**index** parity (even → δt+, odd → δt−), not by the sign of h_max.
- **QMPS topology:** `N+1 = 33` tensors, all shapes derived from `N`, `CHI_Q = 32`, `D_F = 72` (no hard-coded structure). Central feature-only tensor at `N÷2+1`. Init: identity-near real part + N(0, 0.5²) noise on re/im, divided by `INIT_NORM_FACTOR = 4.0` (reference `factor` for csB; keeps the L=32 overlap magnitude off the log's ε floor).
- **DDQN:** double-DQN target, half-MSE loss, replay buffer pre-filled with 8000 random-action transitions before training, hard target-net copy every 10 gradient steps.
- **Head:** `D_F=72 → tanh(200) → tanh(200) → 7`, weights/biases N(0, 0.1²).
- **Schedule:** 40 000 episodes, γ=0.98, lr=5e-5, batch=32, ε exp-decay from 1.0 → 0.01 with time constant `N_ep/8 = 5000`.

### Constants come from Julia

`bridge.N`, `bridge.N_ACTIONS`, `bridge.D_F`, `bridge.N_PARAMS_REAL`, `bridge.F_THRESHOLD`, `bridge.N_STEPS_MAX` are read from the Julia module at import time. Change them in `QMPSRL.jl`, not in Python. Editing `N`, `CHI_BONDS`, or `D_F` invalidates any saved checkpoints (`N_PARAMS_REAL` changes).

## Running things

The repo is multi-shell (Windows PowerShell + Bash); commands below use bash-style paths.

**TN training (paper Study B defaults):**
```bash
cd TN/python
python train.py                                       # 40 000-episode paper run (prefix study_b)
python train.py --episodes 500 --out-prefix smoke     # quick smoke test
python train.py --device cuda --cuda-graphs            # GPU + CUDA-graphed contraction
python train.py --greedy-sweep 50 --sweep-points 100   # OOD eval over gx∈[1.0,1.5] (csB panels b/c)
```
A full run checkpoints + writes snapshot plots every `--checkpoint-every` episodes (default 10 000) while training continues, so you can inspect milestones and kill a plateaued run. Outputs (`*_learning_curve.png`, `*_entropy.png`, `*_trajectory.png`, `*_loss.png`, `*_arrays.npz`, `*_model.pt`, optional `*_greedy_sweep.npz`, plus `*_ep{10000,20000,30000}_*` milestone snapshots) land in `TN/`, not in `TN/python/`.

Julia env is the project at `TN/julia/`. `bridge.py` sets `PYTHON_JULIACALL_PROJECT` to that path; the first run will instantiate. `PYTHON_JULIACALL_EXE` defaults to `/usr/local/bin/julia` — override it via the same env var if your Julia is elsewhere (Windows: set `PYTHON_JULIACALL_EXE` to e.g. `C:\Users\you\.julia\juliaup\julia-1.x.x+0.x64.w64.mingw32\bin\julia.exe`).

**Tests:**
```bash
julia --project=TN/julia TN/julia/test_qmps.jl        # Julia: norm preservation, DMRG, forward feature
cd TN/python && python test_parity.py                 # torch-vs-Julia forward + autograd parity
```

**Published QMPS code (reference, read-only):**
```bash
cd QMPS/dqn && python main.py             # writes results/ in CWD
```

**Lattice scripts:** each is a top-level script — `python Lattice/deepQ_target.py`. Requires the missing `.mat` files; flag this to the user before running.

## Conventions worth knowing

- No CI or linter configured. Verification bar is: `julia --project=TN/julia TN/julia/test_qmps.jl` and `python test_parity.py` both pass, plus a short `train.py` smoke run produces a learning curve without NaNs.
- The `Lattice/` scripts are intentionally redundant single-file experiments; do not refactor them into shared modules unless asked.
- Action-index encoding lives in `QMPSRL.ACTIONS` and `QMPSRL.decode_action`; if you change the action set, update `N_ACTIONS`, the parity convention in `duration_for`, and the bridge's pass-through.
