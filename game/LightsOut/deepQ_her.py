"""HER + DQN for Lights Out 5x5.

Adds Hindsight Experience Replay (Andrychowicz et al. 2017, `future` strategy)
on top of the curriculum + action-masking baseline (variants 3+4 of the
first-try log). Everything else is held fixed:
  - same 128-128 MLP, same target net (Polyak tau=0.99)
  - same epsilon schedule, gamma=0.95, batch 64, replay 50k
  - same curriculum: start scramble=1, promote at 0.8 solve rate
  - same action mask (each cell pressable at most once)

Only differences vs deepQ_lightsout.py:
  1. State is goal-conditioned: [board || goal] (env file does this).
  2. Reward is sparse 0/1 on board==goal. No shaping.
  3. After each episode, push k=4 HER-relabeled copies of every transition,
     using `future` goals sampled from the same trajectory.
"""
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environment_lightsout_her import Env, N, N_CELLS

env = Env(scramble=1, n_steps=25, seed=0)

learning_rate = 1e-3
gamma = 0.95
epsilon = 1.0
eps_min = 0.05
eps_decay = 0.9990
tau = 0.99
batch_size = 64
buffer_cap = 50_000
max_episodes = 8000

her_k = 1   # HER relabels per real transition (split below: future / original)
her_original_frac = 0.5   # of relabels, fraction conditioned on original goal
shaping_alpha = 1.0       # potential-based shaping on fidelity vs the goal

curriculum_max = 15
curriculum_window = 100
curriculum_threshold = 0.80
curriculum_min_eps_at_level = 200


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(env.n_states, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, env.n_actions)
        for m in (self.l1, self.l2, self.l3):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        h = torch.relu(self.l1(x))
        h = torch.relu(self.l2(h))
        return self.l3(h)

    def act(self, state, eps, mask):
        valid_idx = np.flatnonzero(mask > 0.5)
        if len(valid_idx) == 0:
            return 0
        if random.random() > eps:
            s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q = self.forward(s).squeeze(0).numpy()
            q_masked = np.where(mask > 0.5, q, -np.inf)
            return int(np.argmax(q_masked))
        return int(np.random.choice(valid_idx))


model = DQN()
target = DQN()
target.load_state_dict(model.state_dict())
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def calc_loss():
    s, a, r, s2, d, m2 = zip(*random.sample(replay_buffer, batch_size))
    s = torch.as_tensor(np.array(s), dtype=torch.float32)
    s2 = torch.as_tensor(np.array(s2), dtype=torch.float32)
    a = torch.as_tensor(a, dtype=torch.long)
    r = torch.as_tensor(r, dtype=torch.float32)
    d = torch.as_tensor(d, dtype=torch.float32)
    m2 = torch.as_tensor(np.array(m2), dtype=torch.float32)

    q = model(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q_all = target(s2).masked_fill(m2 < 0.5, float("-inf"))
        any_valid = (m2.sum(dim=1) > 0).float()
        next_q = next_q_all.max(1)[0]
        next_q = torch.where(any_valid > 0.5, next_q, torch.zeros_like(next_q))
        y = r + gamma * next_q * (1 - d)
    loss = (q - y).pow(2).mean()
    loss_conv.append(loss.item())
    return loss


def make_transition(board_t, board_tp1, goal, action, mask_tp1, is_episode_end):
    """Build a (s, a, r, s', done, next_mask) tuple for a given goal.

    Reward shape matches variant 3 of first_try.md, but goal-conditioned:
      shaping     = fid(board_tp1, g) - fid(board_t, g)  (telescopes)
      terminal    = done * (fid > 0.2) * (fid - 0.2) / (1 - fid)
    The terminal blow-up at fid -> 1 is what gave variant 3 its K=1 climb;
    it gets HER-relabeled trajectories a strong "this was a success"
    signal that the plain {0,1} reward did not.
    """
    s = np.concatenate([board_t, goal]).astype(np.float32)
    s2 = np.concatenate([board_tp1, goal]).astype(np.float32)
    achieved = np.array_equal(board_tp1, goal)
    fid_t = float(np.mean(board_t == goal))
    fid_tp1 = float(np.mean(board_tp1 == goal))
    done = bool(achieved or is_episode_end)
    f_clip = min(fid_tp1, 0.9999)
    terminal = float(done) * float(fid_tp1 > 0.2) * (f_clip - 0.2) / (1.0 - f_clip)
    r = shaping_alpha * (fid_tp1 - fid_t) + terminal
    m2 = np.zeros_like(mask_tp1) if achieved else mask_tp1
    return (s, int(action), float(r), s2, float(done), m2.astype(np.float32))


replay_buffer = deque(maxlen=buffer_cap)
loss_conv = []
fid_per_ep = []
solved_per_ep = []
scramble_per_ep = []
ep_at_current_level = 0
best_fid_overall = -1.0
best_actions = None
best_boards = None
best_fid_t = None

for episode in range(max_episodes):
    state = env.reset()
    goal = env.goal.copy()
    boards_t = [env.board.copy()]
    masks_t = [env.valid_action_mask()]
    actions_taken = []
    fid_t = [env._fidelity()]
    done = False

    while not done:
        mask = masks_t[-1]
        a = model.act(state, epsilon, mask)
        next_state, reward, done, fid = env.step(a)
        next_mask = env.valid_action_mask()

        # Push the real (un-relabeled) transition immediately so the
        # interleaved-update cadence matches the baseline.
        replay_buffer.append(
            make_transition(boards_t[-1], env.board, goal, a, next_mask, done)
        )
        actions_taken.append(a)
        boards_t.append(env.board.copy())
        masks_t.append(next_mask)
        fid_t.append(fid)
        state = next_state

        if len(replay_buffer) >= batch_size:
            loss = calc_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for tp, p in zip(target.parameters(), model.parameters()):
                    tp.data.mul_(tau).add_(p.data, alpha=1.0 - tau)

    # ---- HER relabeling ----
    # Per real transition push her_k extra copies. Each copy is, with prob
    # her_original_frac, conditioned on the original goal (all-zeros) -- this
    # duplicates the real signal and rebalances the buffer toward the
    # actual test-time goal. Otherwise it uses a `future` goal sampled from
    # the same trajectory.
    T = len(actions_taken)
    for t in range(T):
        for _ in range(her_k):
            if random.random() < her_original_frac:
                g = goal
                is_end = (t == T - 1) or np.array_equal(boards_t[t + 1], g)
            else:
                future_idx = list(range(t + 1, T + 1))
                if not future_idx:
                    continue
                tf = random.choice(future_idx)
                g = boards_t[tf]
                is_end = (t == T - 1) or (t + 1 == tf)
            replay_buffer.append(
                make_transition(boards_t[t], boards_t[t + 1], g,
                                actions_taken[t], masks_t[t + 1], is_end)
            )

    # Logging signal: did we solve the original (all-zeros) goal?
    final_fid = fid_t[-1]
    solved = int(np.array_equal(boards_t[-1], goal))
    fid_per_ep.append(final_fid)
    solved_per_ep.append(solved)
    scramble_per_ep.append(env.scramble)
    ep_at_current_level += 1

    nontrivial = solved and (len(actions_taken) >= max(1, env.scramble))
    if nontrivial and final_fid > best_fid_overall:
        best_fid_overall = final_fid
        best_actions = list(actions_taken)
        best_boards = [b.copy() for b in boards_t]
        best_fid_t = list(fid_t)

    epsilon = max(epsilon * eps_decay, eps_min)

    if (env.scramble < curriculum_max and
            ep_at_current_level >= curriculum_min_eps_at_level and
            len(solved_per_ep) >= curriculum_window):
        recent_solve_rate = np.mean(solved_per_ep[-curriculum_window:])
        if recent_solve_rate >= curriculum_threshold:
            env.scramble += 1
            ep_at_current_level = 0
            epsilon = max(epsilon, 0.3)
            print(f">>> curriculum: scramble -> {env.scramble} at ep {episode} "
                  f"(solve_rate={recent_solve_rate:.2f})")

    if episode % 50 == 0:
        recent = fid_per_ep[-50:]
        s50 = sum(solved_per_ep[-50:])
        print(f"ep {episode:4d}  eps={epsilon:.3f}  K={env.scramble}  "
              f"last_fid={final_fid:.3f}  best={best_fid_overall:.3f}  "
              f"mean50={np.mean(recent):.3f}  solved50={s50}/{len(recent)}  "
              f"buf={len(replay_buffer)}")

print(f"\nFinished. Best non-trivial fidelity: {best_fid_overall:.3f}")
if best_actions is None:
    print("(no non-trivial solve seen; plotting last episode instead)")
    best_actions = list(actions_taken)
    best_boards = [b.copy() for b in boards_t]
    best_fid_t = list(fid_t)
    best_fid_overall = fid_t[-1]
print(f"Best action sequence ({len(best_actions)} presses): {best_actions}")

# ---- plots (same layout as deepQ_lightsout.py) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))

fig, ax1 = plt.subplots()
ax1.plot(fid_per_ep, alpha=0.3, label="per-episode terminal fidelity")
window = 50
if len(fid_per_ep) >= window:
    smooth = np.convolve(fid_per_ep, np.ones(window) / window, mode="valid")
    ax1.plot(np.arange(window - 1, len(fid_per_ep)), smooth,
             label=f"{window}-ep rolling mean")
    solve_smooth = np.convolve(solved_per_ep, np.ones(window) / window, mode="valid")
    ax1.plot(np.arange(window - 1, len(solved_per_ep)), solve_smooth,
             label=f"{window}-ep solve rate", color="tab:green")
ax1.set_xlabel("episode")
ax1.set_ylabel("fidelity / solve rate")
ax1.set_ylim(0, 1.05)
ax2 = ax1.twinx()
ax2.plot(scramble_per_ep, color="tab:red", alpha=0.5, label="scramble K")
ax2.set_ylabel("scramble K")
fig.legend(loc="lower right")
plt.savefig(os.path.join(here, "learning_curve_her.png"), dpi=120, bbox_inches="tight")

plt.figure()
plt.plot(loss_conv)
plt.xlabel("update step")
plt.ylabel("loss")
plt.yscale("log")
plt.savefig(os.path.join(here, "loss_her.png"), dpi=120, bbox_inches="tight")

plt.figure()
fig, ax1 = plt.subplots()
ax1.plot(range(len(best_fid_t)), best_fid_t, "o-", label="fidelity")
ax1.set_xlabel("step")
ax1.set_ylabel("fidelity")
ax1.set_ylim(0, 1.05)
ax2 = ax1.twinx()
ax2.step(range(1, len(best_actions) + 1), best_actions, where="post",
         color="tab:orange", alpha=0.7, label="press idx")
ax2.set_ylabel("action (cell index)")
ax2.set_ylim(-0.5, env.n_actions - 0.5)
fig.legend(loc="lower right")
plt.title(f"Best episode  F={best_fid_overall:.3f}  ({len(best_actions)} presses)")
plt.savefig(os.path.join(here, "trajectory_her.png"), dpi=120, bbox_inches="tight")

n_snap = len(best_boards)
cols = min(n_snap, 8)
rows = (n_snap + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4))
axes = np.array(axes).reshape(rows, cols)
for k in range(rows * cols):
    ax = axes[k // cols, k % cols]
    ax.set_xticks([]); ax.set_yticks([])
    if k < n_snap:
        ax.imshow(best_boards[k].reshape(N, N), vmin=0, vmax=1, cmap="gray_r")
        title = "start" if k == 0 else f"a={best_actions[k-1]}"
        ax.set_title(title, fontsize=8)
    else:
        ax.axis("off")
plt.suptitle(f"Best episode boards (F={best_fid_overall:.3f})")
plt.savefig(os.path.join(here, "solved_her.png"), dpi=120, bbox_inches="tight")

print("Saved learning_curve_her.png, loss_her.png, trajectory_her.png, solved_her.png")
