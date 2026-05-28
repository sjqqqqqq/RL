"""Double-DQN training loop for the QMPS agent — Paper Study A defaults.

Defaults match Metz & Bukov 2022 (arXiv:2201.11790) Study A / QMPS-1:
  - 4000 episodes, γ=0.98, lr=1e-4, batch=64, buffer=8000
  - replay buffer pre-filled with random-action transitions before training
  - Double-DQN target (argmax from online net, evaluate with target net)
  - half-MSE loss
  - hard target-net copy every 10 gradient steps
  - ε schedule: ε_min + (ε_init − ε_min)·exp(−ep / (N_ep / 8))
  - MLP head: D_F → tanh(100) → tanh(100) → N_ACTIONS, N(0, 0.1²) init
  - Environment: per-episode TFIM ground state at (J=1, gx∈U[1.0,1.1], gz=0),
    target |↑↑↑↑⟩, 12-action set with δt+=π/12, δt-=π/17, F-threshold ≈ 0.98.
"""
from __future__ import annotations

# Single-threaded BLAS: N=4, χ=4 means every matmul is tiny, and threading is
# pure overhead — pack the cluster with many single-thread processes instead.
# Must be set before numpy/torch import their BLAS backends.
import os
for _k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
           "JULIA_NUM_THREADS", "PYTHON_JULIACALL_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import math
import pathlib
import random
import time

# bridge imports juliacall, which must come before torch.
import bridge as B
from qmps_agent import QMPSDQN, clone_for_target, hard_copy
from replay import ReplayBuffer, Transition

import numpy as np
import torch
import torch.optim as optim

torch.set_num_threads(1)
B.set_blas_threads(1)

_OUT_DIR = pathlib.Path(__file__).resolve().parent.parent  # TN/


# ---------------------------------------------------------------------------
# CLI / hyperparameters
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes",       type=int,   default=4_000)
    p.add_argument("--gamma",          type=float, default=0.98)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--batch",          type=int,   default=64)
    p.add_argument("--buffer",         type=int,   default=8_000)
    p.add_argument("--eps-init",       type=float, default=1.0)
    p.add_argument("--eps-min",        type=float, default=0.01)
    p.add_argument("--target-update",  type=int,   default=10,
                   help="Hard target-net copy every N gradient steps (paper n_target).")
    p.add_argument("--n-steps-max",    type=int,   default=B.N_STEPS_MAX)
    p.add_argument("--f-threshold",    type=float, default=B.F_THRESHOLD)
    p.add_argument("--hidden",         type=int,   default=100)
    p.add_argument("--seed",           type=int,   default=30)
    p.add_argument("--log-every",      type=int,   default=50)
    p.add_argument("--out-prefix",     type=str,   default="study_a")
    p.add_argument("--greedy-eval",    type=int,   default=0,
                   help="After training, run this many ε=0 episodes from fresh "
                        "initial states and report success rate / fidelity stats.")
    p.add_argument("--eval-seed",      type=int,   default=10_000_000,
                   help="Base seed for the greedy-eval init-state sampler "
                        "(kept disjoint from training seeds for reproducibility).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model  = QMPSDQN(hidden=args.hidden)
    target = clone_for_target(model)
    optim_ = optim.Adam(model.parameters(), lr=args.lr)

    buffer = ReplayBuffer(args.buffer)
    losses:    list[float] = []
    fid_per_ep: list[float] = []
    best_fid       = -1.0
    best_actions:  list[int]   = []
    best_fid_t:    list[float] = []

    env_seed_rng = random.Random(args.seed)
    grad_step_count = 0

    # ----- Pre-fill the replay buffer with random-action transitions ------
    # Paper's memory_init: run env episodes with ε = eps_init until the buffer
    # is full. Use seeds disjoint from the training-loop seed stream.
    t_prefill = time.time()
    prefill_rng = random.Random(args.seed + 1)
    while len(buffer) < args.buffer:
        env = B.JuliaEnv(seed=prefill_rng.randint(0, 2**31 - 1),
                         f_threshold=args.f_threshold,
                         n_steps_max=args.n_steps_max)
        s = env.state_id
        done = False
        while not done and len(buffer) < args.buffer:
            a = random.randrange(B.N_ACTIONS)
            s2, r, done, _ = env.step(a)
            buffer.push(Transition(s, a, r, s2, done))
            s = s2
    print(f"[prefill] buffer={len(buffer)} in {time.time() - t_prefill:.1f}s",
          flush=True)

    # ----- Training loop --------------------------------------------------
    eps_tau = max(1.0, args.episodes / 8.0)
    t_wall  = time.time()

    for ep in range(args.episodes):
        eps = args.eps_min + (args.eps_init - args.eps_min) * math.exp(-ep / eps_tau)

        # Fresh env each episode (paper: random initial state every episode).
        env = B.JuliaEnv(seed=env_seed_rng.randint(0, 2**31 - 1),
                         f_threshold=args.f_threshold,
                         n_steps_max=args.n_steps_max)
        s = env.state_id
        ep_actions: list[int] = []
        ep_fid_t:   list[float] = [env.fidelity()]
        done = False
        fid  = ep_fid_t[0]

        while not done:
            a = model.act(s, eps)
            s2, r, done, fid = env.step(a)
            buffer.push(Transition(s, a, r, s2, done))
            ep_actions.append(a)
            ep_fid_t.append(fid)
            s = s2

            _gradient_step(model, target, optim_, buffer, args, losses)
            grad_step_count += 1
            if grad_step_count % args.target_update == 0:
                hard_copy(target, model)

        fid_per_ep.append(fid)
        if fid > best_fid:
            best_fid     = fid
            best_actions = list(ep_actions)
            best_fid_t   = list(ep_fid_t)

        if ep % args.log_every == 0:
            window = fid_per_ep[-args.log_every:]
            print(f"ep {ep:5d}  eps={eps:.3f}  last_fid={fid:.3f}  "
                  f"best={best_fid:.3f}  "
                  f"mean_last_{args.log_every}={np.mean(window):.3f}  "
                  f"buf={len(buffer)}  reg={B.registry_size()}  "
                  f"loss_recent={np.mean(losses[-100:]) if losses else float('nan'):.4f}",
                  flush=True)

    elapsed = time.time() - t_wall
    print(f"\nDone in {elapsed:.1f}s. Best fidelity: {best_fid:.4f}")

    _save_checkpoint(model, target, args, elapsed, best_fid)
    _save_arrays(fid_per_ep, best_fid_t, best_actions, losses, args)
    _save_plots(fid_per_ep, best_fid_t, losses, best_actions, args)

    if args.greedy_eval > 0:
        _greedy_eval(model, args)


def _gradient_step(model: QMPSDQN,
                   target: QMPSDQN,
                   optim_: optim.Optimizer,
                   buffer: ReplayBuffer,
                   args: argparse.Namespace,
                   losses: list[float]) -> None:
    batch = buffer.sample(args.batch)
    s_ids   = [tr.s  for tr in batch]
    s2_ids  = [tr.s2 for tr in batch]
    actions = torch.tensor([tr.a for tr in batch], dtype=torch.long)
    rewards = torch.tensor([tr.r for tr in batch], dtype=torch.float32)
    dones   = torch.tensor([float(tr.done) for tr in batch], dtype=torch.float32)

    # Online net for the chosen-action Q-value (with grad)
    q_all = model.forward_batch(s_ids)                              # (B, N_ACTIONS)
    q     = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)        # (B,)

    # Double-DQN target: online net picks the action, target net evaluates it
    with torch.no_grad():
        next_argmax = model.forward_batch(s2_ids).argmax(dim=1)     # (B,)
        tq_all      = target.forward_batch(s2_ids)                  # (B, N_ACTIONS)
        tq_next     = tq_all.gather(1, next_argmax.unsqueeze(1)).squeeze(1)
        y           = rewards + args.gamma * tq_next * (1.0 - dones)

    loss = 0.5 * (q - y).pow(2).mean()                              # paper half-MSE
    optim_.zero_grad()
    loss.backward()
    optim_.step()
    losses.append(loss.item())


# ---------------------------------------------------------------------------
# Checkpoint / plot / eval helpers
# ---------------------------------------------------------------------------


def _save_checkpoint(model: QMPSDQN,
                     target: QMPSDQN,
                     args: argparse.Namespace,
                     elapsed: float,
                     best_fid: float) -> None:
    path = f"{_OUT_DIR}/{args.out_prefix}_model.pt"
    torch.save({
        "model_state_dict":  model.state_dict(),
        "target_state_dict": target.state_dict(),
        "hidden":            args.hidden,
        "args":              vars(args),
        "elapsed_s":         elapsed,
        "best_fid":          best_fid,
        "N":                 B.N,
        "D_F":               B.D_F,
        "N_ACTIONS":         B.N_ACTIONS,
        "N_PARAMS_REAL":     B.N_PARAMS_REAL,
    }, path)
    print(f"Saved model checkpoint to {path}")


def _save_arrays(fid_per_ep: list[float],
                 best_fid_t: list[float],
                 best_actions: list[int],
                 losses: list[float],
                 args: argparse.Namespace) -> None:
    path = f"{_OUT_DIR}/{args.out_prefix}_arrays.npz"
    np.savez_compressed(
        path,
        fid_per_ep   = np.asarray(fid_per_ep, dtype=np.float32),
        losses       = np.asarray(losses,     dtype=np.float32),
        best_fid_t   = np.asarray(best_fid_t, dtype=np.float32),
        best_actions = np.asarray(best_actions, dtype=np.int32),
        f_threshold  = np.float32(args.f_threshold),
    )
    print(f"Saved raw arrays to {path}")


def _greedy_eval(model: QMPSDQN, args: argparse.Namespace) -> None:
    rng = random.Random(args.eval_seed)
    terminal_fids: list[float] = []
    steps_used:    list[int]   = []
    t0 = time.time()
    for _ in range(args.greedy_eval):
        env = B.JuliaEnv(seed=rng.randint(0, 2**31 - 1),
                         f_threshold=args.f_threshold,
                         n_steps_max=args.n_steps_max)
        s = env.state_id
        done = False
        fid = env.fidelity()
        n_steps = 0
        while not done:
            a = model.act(s, eps=0.0)
            s, _, done, fid = env.step(a)
            n_steps += 1
        terminal_fids.append(fid)
        steps_used.append(n_steps)

    arr       = np.asarray(terminal_fids)
    steps_arr = np.asarray(steps_used, dtype=np.int32)
    n_succ    = int((arr >= args.f_threshold).sum())
    print(f"\n[greedy eval] {args.greedy_eval} episodes, "
          f"{time.time() - t0:.1f}s")
    print(f"  success rate (F >= {args.f_threshold:.4f}): "
          f"{n_succ}/{args.greedy_eval} = {n_succ / args.greedy_eval:.3f}")
    print(f"  fidelity: mean={arr.mean():.4f}  median={np.median(arr):.4f}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}")
    print(f"  steps used: mean={np.mean(steps_used):.1f}  "
          f"median={np.median(steps_used):.1f}  max={max(steps_used)}")

    npz = f"{_OUT_DIR}/{args.out_prefix}_greedy_eval.npz"
    np.savez_compressed(
        npz,
        terminal_fids = arr.astype(np.float32),
        steps_used    = steps_arr,
        f_threshold   = np.float32(args.f_threshold),
        n_episodes    = np.int32(args.greedy_eval),
    )
    print(f"  saved raw arrays to {npz}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(arr, bins=40, range=(0.0, 1.0))
    plt.axvline(args.f_threshold, ls="--", color="grey",
                label=f"F* = {args.f_threshold:.4f}")
    plt.xlabel("terminal fidelity (greedy, fresh init states)")
    plt.ylabel("count")
    plt.title(f"Greedy eval: success {n_succ}/{args.greedy_eval} "
              f"({n_succ / args.greedy_eval:.1%})")
    plt.legend()
    out = f"{_OUT_DIR}/{args.out_prefix}_greedy_eval.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  saved histogram to {out}")


def _save_plots(fid_per_ep: list[float],
                best_fid_t: list[float],
                losses: list[float],
                best_actions: list[int],
                args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prefix = args.out_prefix

    # Learning curve
    plt.figure()
    plt.plot(fid_per_ep, alpha=0.3, label="per-episode terminal fidelity")
    window = max(10, len(fid_per_ep) // 50)
    if len(fid_per_ep) >= window:
        smooth = np.convolve(fid_per_ep, np.ones(window) / window, mode="valid")
        plt.plot(np.arange(window - 1, len(fid_per_ep)), smooth,
                 label=f"{window}-episode rolling mean")
    plt.axhline(args.f_threshold, ls="--", color="grey", label="F* threshold")
    plt.xlabel("episode")
    plt.ylabel("terminal fidelity")
    plt.legend()
    plt.title(f"QMPS-DDQN Study A, N={B.N}, best F={max(fid_per_ep):.3f}")
    plt.savefig(f"{_OUT_DIR}/{prefix}_learning_curve.png", dpi=120, bbox_inches="tight")

    # Trajectory of the best episode
    plt.figure()
    plt.plot(best_fid_t, "o-", label="fidelity")
    plt.xlabel("protocol step")
    plt.ylabel("fidelity")
    plt.axhline(args.f_threshold, ls="--", color="grey")
    plt.title(f"Best episode (F={best_fid_t[-1]:.3f})")
    plt.legend()
    plt.savefig(f"{_OUT_DIR}/{prefix}_trajectory.png", dpi=120, bbox_inches="tight")

    # Loss
    plt.figure()
    plt.plot(losses)
    plt.xlabel("update step")
    plt.ylabel("Bellman loss")
    plt.yscale("log")
    plt.savefig(f"{_OUT_DIR}/{prefix}_loss.png", dpi=120, bbox_inches="tight")
    print(f"Saved {prefix}_learning_curve.png, {prefix}_trajectory.png, "
          f"{prefix}_loss.png in TN/")


if __name__ == "__main__":
    train(parse_args())
