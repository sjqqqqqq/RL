"""Double-DQN training loop for the QMPS agent — Paper Study B defaults.

Defaults match Metz & Bukov 2022 (arXiv:2201.11790) Study B / `QMPS/figures/csB`:
  - L=32 (constants in QMPSRL.jl), FM init (J=+1, gx∼U[1.0,1.1]), z-polarized target
  - 7-action set [gy+, gz±, Jx±, Jy±]
  - 40 000 episodes, γ=0.98, lr=5e-5, batch=32, buffer=8000
  - MLP head: D_F=72 → tanh(200) → tanh(200) → 7
  - QMPS bond χ=32, feature D_F=72
  - replay buffer pre-filled with random-action transitions
  - Double-DQN target, half-MSE loss, hard target-net copy every 10 grad steps
  - ε schedule: ε_min + (ε_init−ε_min)·exp(−ep / (N_ep/8))

Per-episode entanglement-entropy logging reproduces the csB panel-c inset.
Greedy-sweep eval (--greedy-sweep M) tests OOD generalization across
gx ∈ [1.0, 1.5] with M episodes per gx point — reproduces csB panels (b)+(c).
"""
from __future__ import annotations

# Thread count must be pinned BEFORE numpy/torch import their BLAS backends,
# so we sniff --threads from sys.argv up-front before full argparse runs.
import os
import sys


def _early_threads_arg() -> int:
    default = 8
    for i, tok in enumerate(sys.argv):
        if tok == "--threads" and i + 1 < len(sys.argv):
            try:
                return int(sys.argv[i + 1])
            except ValueError:
                pass
        if tok.startswith("--threads="):
            try:
                return int(tok.split("=", 1)[1])
            except ValueError:
                pass
    return default


_THREADS = _early_threads_arg()
for _k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
           "JULIA_NUM_THREADS", "PYTHON_JULIACALL_THREADS"):
    os.environ.setdefault(_k, str(_THREADS))

import argparse
import math
import pathlib
import random
import time

# bridge imports juliacall, which must come before torch.
import bridge as B
import qmps_torch as QT
from qmps_agent import QMPSDQN, clone_for_target, hard_copy
from replay import ReplayBuffer, Transition

import numpy as np
import torch
import torch.optim as optim

torch.set_num_threads(_THREADS)
B.set_blas_threads(_THREADS)

_OUT_DIR = pathlib.Path(__file__).resolve().parent.parent  # TN/


# ---------------------------------------------------------------------------
# CLI / hyperparameters
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes",       type=int,   default=40_000)
    p.add_argument("--gamma",          type=float, default=0.98)
    p.add_argument("--lr",             type=float, default=5e-5)
    p.add_argument("--batch",          type=int,   default=32)
    p.add_argument("--buffer",         type=int,   default=8_000)
    p.add_argument("--eps-init",       type=float, default=1.0)
    p.add_argument("--eps-min",        type=float, default=0.01)
    p.add_argument("--target-update",  type=int,   default=10,
                   help="Hard target-net copy every N gradient steps (paper n_target).")
    p.add_argument("--n-steps-max",    type=int,   default=B.N_STEPS_MAX)
    p.add_argument("--f-threshold",    type=float, default=B.F_THRESHOLD)
    p.add_argument("--hidden",         type=int,   default=200)
    p.add_argument("--device",         type=str,   default="cpu",
                   help="Torch device for the QMPS contraction + head: "
                        "'cpu', 'cuda', or 'auto' (cuda if available).")
    p.add_argument("--cuda-graphs",    action="store_true",
                   help="CUDA-graph the contraction at batch size (cuda only); "
                        "~7x on the launch-bound grad step. act() stays eager.")
    p.add_argument("--seed",           type=int,   default=33)
    p.add_argument("--log-every",      type=int,   default=100)
    p.add_argument("--checkpoint-every", type=int,  default=10_000,
                   help="Save an episode-tagged checkpoint + snapshot plots every "
                        "N episodes while training continues (0 disables). Lets you "
                        "inspect at milestones and kill the run if it has plateaued.")
    p.add_argument("--out-prefix",     type=str,   default="study_b")
    p.add_argument("--threads",        type=int,   default=_THREADS,
                   help="BLAS / OMP / Julia threads (sniffed pre-import).")
    p.add_argument("--greedy-sweep",   type=int,   default=0,
                   help="After training, run this many ε=0 episodes per gx value "
                        "across gx ∈ [1.0, 1.5] (paper csB panels b/c).")
    p.add_argument("--sweep-points",   type=int,   default=100,
                   help="Number of gx grid points in [1.0, 1.5] for the sweep.")
    p.add_argument("--sweep-seed",     type=int,   default=10_000_000)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    # The QMPS contraction + per-state psi cache must live on the same device as
    # the model params. complex64 is the validated GPU path (fp32 head dtype).
    QT.set_backend(device=device, dtype=torch.complex64)

    print(f"[config] L={B.N} D_F={B.D_F} N_ACTIONS={B.N_ACTIONS} "
          f"N_PARAMS_REAL={B.N_PARAMS_REAL} F_THRESHOLD={args.f_threshold:.4f} "
          f"threads={args.threads} device={device}", flush=True)

    model  = QMPSDQN(hidden=args.hidden).to(device)
    target = clone_for_target(model)
    optim_ = optim.Adam(model.parameters(), lr=args.lr)

    buffer = ReplayBuffer(args.buffer)
    losses:        list[float] = []
    fid_per_ep:    list[float] = []
    entropy_per_ep: list[float] = []
    best_fid       = -1.0
    best_actions:  list[int]   = []
    best_fid_t:    list[float] = []

    env_seed_rng = random.Random(args.seed)
    grad_step_count = 0

    # ----- Pre-fill the replay buffer with random-action transitions ------
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

    if args.cuda_graphs:
        if device.type != "cuda":
            raise SystemExit("--cuda-graphs requires --device cuda.")
        # Take ids directly from the buffer (no random.sample) so graph capture
        # does not perturb the global RNG stream vs an eager run.
        sample_ids = [tr.s for tr in list(buffer.buf)[:args.batch]]
        model.enable_cuda_graph(sample_ids)
        target.enable_cuda_graph(sample_ids)
        optim_.zero_grad(set_to_none=True)   # drop grads left by capture warmup
        print(f"[cuda-graph] captured contraction at batch={args.batch}", flush=True)

    # ----- Training loop --------------------------------------------------
    eps_tau = max(1.0, args.episodes / 8.0)
    t_wall  = time.time()

    for ep in range(args.episodes):
        eps = args.eps_min + (args.eps_init - args.eps_min) * math.exp(-ep / eps_tau)

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
        # Log half-chain entropy of the final state (csB inset metric).
        entropy_per_ep.append(B.half_chain_entropy_id(s))

        if fid > best_fid:
            best_fid     = fid
            best_actions = list(ep_actions)
            best_fid_t   = list(ep_fid_t)

        if ep % args.log_every == 0:
            window = fid_per_ep[-args.log_every:]
            print(f"ep {ep:6d}  eps={eps:.3f}  last_fid={fid:.3f}  "
                  f"F_sp={fid**(1.0/B.N):.4f}  "
                  f"best={best_fid:.3f}  "
                  f"mean_last_{args.log_every}={np.mean(window):.3f}  "
                  f"S_ent={entropy_per_ep[-1]:.3f}  "
                  f"buf={len(buffer)}  reg={B.registry_size()}  "
                  f"loss_recent={np.mean(losses[-100:]) if losses else float('nan'):.4f}",
                  flush=True)

        # Milestone snapshot: checkpoint + plots every N episodes (training keeps
        # going). Inspect these to decide whether to let the run continue.
        ep_done = ep + 1
        if (args.checkpoint_every > 0 and ep_done % args.checkpoint_every == 0
                and ep_done < args.episodes):
            tag = f"{args.out_prefix}_ep{ep_done}"
            recent_fsp = np.mean([f ** (1.0 / B.N) for f in fid_per_ep[-500:]])
            print(f"[milestone ep {ep_done}] best_F_sp={best_fid**(1.0/B.N):.4f}  "
                  f"mean_F_sp_last500={recent_fsp:.4f}  -> saving '{tag}_*'",
                  flush=True)
            _save_checkpoint(model, target, args, time.time() - t_wall, best_fid, prefix=tag)
            _save_arrays(fid_per_ep, entropy_per_ep, best_fid_t, best_actions, losses, args, prefix=tag)
            _save_plots(fid_per_ep, entropy_per_ep, best_fid_t, losses, best_actions, args, prefix=tag)

    elapsed = time.time() - t_wall
    print(f"\nDone in {elapsed:.1f}s. Best fidelity: {best_fid:.4f} "
          f"(F_sp={best_fid**(1.0/B.N):.4f})")

    _save_checkpoint(model, target, args, elapsed, best_fid)
    _save_arrays(fid_per_ep, entropy_per_ep, best_fid_t, best_actions, losses, args)
    _save_plots(fid_per_ep, entropy_per_ep, best_fid_t, losses, best_actions, args)

    if args.greedy_sweep > 0:
        _greedy_sweep(model, args)


def _gradient_step(model: QMPSDQN,
                   target: QMPSDQN,
                   optim_: optim.Optimizer,
                   buffer: ReplayBuffer,
                   args: argparse.Namespace,
                   losses: list[float]) -> None:
    dev = args.device
    batch = buffer.sample(args.batch)
    s_ids   = [tr.s  for tr in batch]
    s2_ids  = [tr.s2 for tr in batch]
    actions = torch.tensor([tr.a for tr in batch], dtype=torch.long, device=dev)
    rewards = torch.tensor([tr.r for tr in batch], dtype=torch.float32, device=dev)
    dones   = torch.tensor([float(tr.done) for tr in batch], dtype=torch.float32, device=dev)

    # Target (no_grad) forwards FIRST, grad-bearing online forward LAST. With a
    # CUDA-graphed contraction one graph instance shares static buffers, so the
    # grad forward must be the final graph call before backward(), else a later
    # forward overwrites the activations its backward needs. Order-independent
    # for eager (these are separate computations), required for --cuda-graphs.
    with torch.no_grad():
        next_argmax = model.forward_batch(s2_ids).argmax(dim=1)
        tq_all      = target.forward_batch(s2_ids)
        tq_next     = tq_all.gather(1, next_argmax.unsqueeze(1)).squeeze(1)
        y           = rewards + args.gamma * tq_next * (1.0 - dones)

    q_all = model.forward_batch(s_ids)
    q     = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

    loss = 0.5 * (q - y).pow(2).mean()
    optim_.zero_grad()
    loss.backward()
    optim_.step()
    losses.append(loss.item())


# ---------------------------------------------------------------------------
# Checkpoint / save helpers
# ---------------------------------------------------------------------------


def _save_checkpoint(model: QMPSDQN,
                     target: QMPSDQN,
                     args: argparse.Namespace,
                     elapsed: float,
                     best_fid: float,
                     prefix: str | None = None) -> None:
    prefix = prefix or args.out_prefix
    path = f"{_OUT_DIR}/{prefix}_model.pt"
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


def _save_arrays(fid_per_ep:    list[float],
                 entropy_per_ep: list[float],
                 best_fid_t:    list[float],
                 best_actions:  list[int],
                 losses:        list[float],
                 args: argparse.Namespace,
                 prefix: str | None = None) -> None:
    prefix = prefix or args.out_prefix
    path = f"{_OUT_DIR}/{prefix}_arrays.npz"
    np.savez_compressed(
        path,
        fid_per_ep     = np.asarray(fid_per_ep,     dtype=np.float32),
        entropy_per_ep = np.asarray(entropy_per_ep, dtype=np.float32),
        losses         = np.asarray(losses,         dtype=np.float32),
        best_fid_t     = np.asarray(best_fid_t,     dtype=np.float32),
        best_actions   = np.asarray(best_actions,   dtype=np.int32),
        f_threshold    = np.float32(args.f_threshold),
    )
    print(f"Saved raw arrays to {path}")


def _greedy_sweep(model: QMPSDQN, args: argparse.Namespace) -> None:
    """Reproduce csB panels (b)+(c): for each gx in a sweep, run M greedy
    episodes from fresh DMRG ground states and record F_sp, steps, 2-site count."""
    gxs = np.linspace(1.0, 1.5, args.sweep_points)
    # action_kinds: True if action index is a 2-site (XX, YY for Study B → 3,4,5,6).
    twosite_set = {3, 4, 5, 6}

    F_sp_mean = np.zeros(args.sweep_points)
    steps_mean = np.zeros(args.sweep_points)
    twosite_mean = np.zeros(args.sweep_points)

    rng = random.Random(args.sweep_seed)
    t0 = time.time()
    M = args.greedy_sweep
    print(f"\n[greedy sweep] {args.sweep_points} gx points × {M} episodes "
          f"= {args.sweep_points * M} rollouts", flush=True)
    for k, gx in enumerate(gxs):
        F_sps, n_steps, n_twosites = [], [], []
        for _ in range(M):
            env = B.JuliaEnv.at_gx(float(gx),
                                   seed=rng.randint(0, 2**31 - 1),
                                   f_threshold=args.f_threshold,
                                   n_steps_max=args.n_steps_max)
            s = env.state_id
            done = False
            fid = env.fidelity()
            steps = 0
            twosite = 0
            while not done:
                a = model.act(s, eps=0.0)
                if a in twosite_set:
                    twosite += 1
                s, _, done, fid = env.step(a)
                steps += 1
            F_sps.append(fid ** (1.0 / B.N))
            n_steps.append(steps)
            n_twosites.append(twosite)
        F_sp_mean[k]    = float(np.mean(F_sps))
        steps_mean[k]   = float(np.mean(n_steps))
        twosite_mean[k] = float(np.mean(n_twosites))
        if k % 10 == 0 or k == args.sweep_points - 1:
            print(f"  gx={gx:.3f}  F_sp={F_sp_mean[k]:.4f}  "
                  f"steps={steps_mean[k]:.1f}  2site={twosite_mean[k]:.1f}",
                  flush=True)
    print(f"[greedy sweep] {time.time() - t0:.1f}s")

    npz = f"{_OUT_DIR}/{args.out_prefix}_greedy_sweep.npz"
    np.savez_compressed(
        npz,
        gxs           = gxs.astype(np.float32),
        F_sp_mean     = F_sp_mean.astype(np.float32),
        steps_mean    = steps_mean.astype(np.float32),
        twosite_mean  = twosite_mean.astype(np.float32),
        episodes_per_gx = np.int32(M),
    )
    print(f"  saved sweep arrays to {npz}")


def _save_plots(fid_per_ep: list[float],
                entropy_per_ep: list[float],
                best_fid_t: list[float],
                losses: list[float],
                best_actions: list[int],
                args: argparse.Namespace,
                prefix: str | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prefix = prefix or args.out_prefix

    # Learning curve — F_sp scale
    fid_arr = np.asarray(fid_per_ep)
    F_sp = fid_arr ** (1.0 / B.N)
    plt.figure(figsize=(10, 4.5))
    plt.plot(F_sp, alpha=0.25, label="per-episode terminal $F_{sp}$")
    window = max(50, len(F_sp) // 100)
    if len(F_sp) >= window:
        smooth = np.convolve(F_sp, np.ones(window) / window, mode="valid")
        plt.plot(np.arange(window - 1, len(F_sp)), smooth,
                 label=f"{window}-episode rolling mean")
    plt.axhline(0.99, ls="--", color="grey", label="$F_{sp}^*$ = 0.99")
    plt.xlabel("episode")
    plt.ylabel("$F_{sp} = F^{1/N}$")
    plt.legend()
    plt.title(f"QMPS-DDQN Study B, N={B.N}, best $F_{{sp}}$={F_sp.max():.4f}")
    plt.savefig(f"{_OUT_DIR}/{prefix}_learning_curve.png",
                dpi=120, bbox_inches="tight")
    plt.close()

    # Entanglement entropy curve
    plt.figure(figsize=(10, 3.5))
    ent = np.asarray(entropy_per_ep)
    plt.plot(ent, alpha=0.3, color="limegreen", label="per-episode terminal $S_{ent}^{N/2}$")
    if len(ent) >= window:
        smooth = np.convolve(ent, np.ones(window) / window, mode="valid")
        plt.plot(np.arange(window - 1, len(ent)), smooth, color="darkgreen",
                 label=f"{window}-episode rolling mean")
    plt.xlabel("episode")
    plt.ylabel("$S_{ent}^{N/2}$")
    plt.legend()
    plt.savefig(f"{_OUT_DIR}/{prefix}_entropy.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Best-episode trajectory (fidelity per step)
    plt.figure()
    plt.plot(best_fid_t, "o-", label="fidelity")
    plt.xlabel("protocol step")
    plt.ylabel("fidelity")
    plt.axhline(args.f_threshold, ls="--", color="grey")
    plt.title(f"Best episode (F={best_fid_t[-1]:.3f}, "
              f"$F_{{sp}}$={best_fid_t[-1]**(1.0/B.N):.4f})")
    plt.legend()
    plt.savefig(f"{_OUT_DIR}/{prefix}_trajectory.png",
                dpi=120, bbox_inches="tight")
    plt.close()

    # Loss curve
    plt.figure()
    plt.plot(losses)
    plt.xlabel("update step")
    plt.ylabel("Bellman loss")
    plt.yscale("log")
    plt.savefig(f"{_OUT_DIR}/{prefix}_loss.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {prefix}_learning_curve.png, _entropy.png, _trajectory.png, "
          f"_loss.png in TN/")


if __name__ == "__main__":
    train(parse_args())
