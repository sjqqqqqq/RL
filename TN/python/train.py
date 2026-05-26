"""DDQN training loop for the QMPS agent.

Adapted from Lattice/deepQ_target.py. Differences:
  - state is an int handle into Julia's MPS registry, not an np.ndarray.
  - reward is the paper's log-fidelity, per-step (not the sparse Mayer reward).
  - Q-value batches loop one element at a time (the Julia overlap is not
    batched yet).
"""
from __future__ import annotations

import argparse
import pathlib
import random
import time

import numpy as np
import torch
import torch.optim as optim

import bridge as B
from qmps_agent import QMPSDQN, clone_for_target, polyak_update
from replay import ReplayBuffer, Transition

_OUT_DIR = pathlib.Path(__file__).resolve().parent.parent  # TN/


# ---------------------------------------------------------------------------
# CLI / hyperparameters
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes",       type=int,   default=5000)
    p.add_argument("--gamma",          type=float, default=0.99)
    p.add_argument("--tau",            type=float, default=0.99)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--batch",          type=int,   default=32)
    p.add_argument("--buffer",         type=int,   default=10_000)
    p.add_argument("--eps",            type=float, default=0.5)
    p.add_argument("--eps-decay",      type=float, default=0.9995)
    p.add_argument("--eps-min",        type=float, default=0.01)
    p.add_argument("--n-steps-max",    type=int,   default=50)
    p.add_argument("--f-threshold",    type=float, default=0.85)
    p.add_argument("--hidden",         type=int,   default=32)
    p.add_argument("--seed",           type=int,   default=0)
    p.add_argument("--log-every",      type=int,   default=10)
    p.add_argument("--out-prefix",     type=str,   default="study_a")
    p.add_argument("--greedy-eval",    type=int,   default=0,
                   help="After training, run this many ε=0 episodes from fresh "
                        "initial states and report success rate / fidelity stats.")
    p.add_argument("--eval-seed",      type=int,   default=10_000_000,
                   help="Base seed for the greedy-eval init-state sampler "
                        "(kept disjoint from training seeds for reproducibility).")
    return p.parse_args()


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = QMPSDQN(hidden=args.hidden)
    target = clone_for_target(model)
    optim_ = optim.Adam(model.parameters(), lr=args.lr)

    buffer = ReplayBuffer(args.buffer)
    losses: list[float] = []
    fid_per_ep: list[float] = []
    best_fid = -1.0
    best_actions: list[int] = []
    best_fid_t: list[float] = []

    eps = args.eps
    env_seed_rng = random.Random(args.seed)

    t_wall = time.time()

    for ep in range(args.episodes):
        # Fresh env each episode (paper: random initial state every episode).
        env = B.JuliaEnv(seed=env_seed_rng.randint(0, 2**31 - 1),
                         f_threshold=args.f_threshold,
                         n_steps_max=args.n_steps_max)
        s = env.state_id
        ep_actions: list[int] = []
        ep_fid_t: list[float] = [env.fidelity()]
        done = False
        fid = ep_fid_t[0]

        while not done:
            a = model.act(s, eps)
            s2, r, done, fid = env.step(a)
            buffer.push(Transition(s, a, r, s2, done))
            ep_actions.append(a)
            ep_fid_t.append(fid)
            s = s2

            if len(buffer) >= args.batch:
                _gradient_step(model, target, optim_, buffer, args, losses)
                polyak_update(target, model, args.tau)

        fid_per_ep.append(fid)
        if fid > best_fid:
            best_fid = fid
            best_actions = list(ep_actions)
            best_fid_t = list(ep_fid_t)

        eps = max(eps * args.eps_decay, args.eps_min)

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
    s_ids  = [tr.s  for tr in batch]
    s2_ids = [tr.s2 for tr in batch]
    actions = torch.tensor([tr.a for tr in batch], dtype=torch.long)
    rewards = torch.tensor([tr.r for tr in batch], dtype=torch.float32)
    dones   = torch.tensor([float(tr.done) for tr in batch], dtype=torch.float32)

    # One batched Julia call for model (with grad), one for target (no grad).
    q_all = model.forward_batch(s_ids)               # (B, N_ACTIONS)
    q     = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        tq_all = target.forward_batch(s2_ids)
        y = rewards + args.gamma * tq_all.max(dim=1).values * (1.0 - dones)

    loss = (q - y).pow(2).mean()
    optim_.zero_grad()
    loss.backward()
    optim_.step()
    losses.append(loss.item())


def _greedy_eval(model: QMPSDQN, args: argparse.Namespace) -> None:
    rng = random.Random(args.eval_seed)
    terminal_fids: list[float] = []
    steps_used: list[int] = []
    t0 = time.time()
    for k in range(args.greedy_eval):
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

    arr = np.asarray(terminal_fids)
    n_succ = int((arr >= args.f_threshold).sum())
    print(f"\n[greedy eval] {args.greedy_eval} episodes, "
          f"{time.time() - t0:.1f}s")
    print(f"  success rate (F >= {args.f_threshold}): "
          f"{n_succ}/{args.greedy_eval} = {n_succ / args.greedy_eval:.3f}")
    print(f"  fidelity: mean={arr.mean():.4f}  median={np.median(arr):.4f}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}")
    print(f"  steps used: mean={np.mean(steps_used):.1f}  "
          f"median={np.median(steps_used):.1f}  max={max(steps_used)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(arr, bins=40, range=(0.0, 1.0))
    plt.axvline(args.f_threshold, ls="--", color="grey",
                label=f"F* = {args.f_threshold}")
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
    plt.title(f"QMPS-DQN, N={B.N}, best F={max(fid_per_ep):.3f}")
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
