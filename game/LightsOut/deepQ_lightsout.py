"""Lights Out 5x5 with target-network DQN.

Same training scheme as Lattice/deepQ_target.py:
  - 2-layer MLP Q-network
  - separate target network, Polyak soft updates
  - sparse terminal reward F/(1-F) shape
  - epsilon-greedy with exponential decay
  - uniform replay buffer

Logs:
  - learning_curve.png : per-episode terminal fidelity + rolling mean
  - loss.png           : Bellman loss vs update step
  - trajectory.png     : best episode -- fidelity over time + the press sequence
  - solved.png         : board snapshots from the best episode
"""
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environment_lightsout import Env, N

env = Env(scramble=1, n_steps=20, threshold=0.999, seed=0)

# Hyperparameters
learning_rate = 1e-3
gamma = 0.95
epsilon = 1.0
eps_min = 0.05
eps_decay = 0.9990
tau = 0.99
batch_size = 64
buffer_cap = 50_000
max_episodes = 8000

# Curriculum: start at scramble=1, promote when recent solve-rate is high
curriculum_max = 10
curriculum_window = 100
curriculum_threshold = 0.80
curriculum_min_eps_at_level = 200   # min episodes per level before promoting


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
        # mask: float32 array, 1 for valid actions, 0 for forbidden.
        valid_idx = np.flatnonzero(mask > 0.5)
        if len(valid_idx) == 0:
            return 0  # no legal action; episode will terminate anyway
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
    m2 = torch.as_tensor(np.array(m2), dtype=torch.float32)   # next-state valid mask

    q = model(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q_all = target(s2)
        # mask invalid actions to -inf so they are never the bootstrap target
        next_q_all = next_q_all.masked_fill(m2 < 0.5, float("-inf"))
        # if all actions are masked (terminal), max would be -inf; clamp to 0
        any_valid = (m2.sum(dim=1) > 0).float()
        next_q = next_q_all.max(1)[0]
        next_q = torch.where(any_valid > 0.5, next_q, torch.zeros_like(next_q))
        y = r + gamma * next_q * (1 - d)
    loss = (q - y).pow(2).mean()
    loss_conv.append(loss.item())
    return loss


replay_buffer = deque(maxlen=buffer_cap)
loss_conv = []
fid_per_ep = []
solved_per_ep = []
scramble_per_ep = []
best_fid_overall = -1.0
best_actions = None
best_boards = None
best_fid_t = None
ep_at_current_level = 0

for episode in range(max_episodes):
    state = env.reset()
    boards = [env.board.copy()]
    actions_taken = []
    fid = env._fidelity()
    fid_t = [fid]
    done = False

    while not done:
        mask = env.valid_action_mask()
        a = model.act(state, epsilon, mask)
        next_state, reward, done, fid = env.step(a, fid)
        next_mask = env.valid_action_mask()
        replay_buffer.append((state, a, reward, next_state, float(done), next_mask))
        actions_taken.append(a)
        boards.append(env.board.copy())
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

    fid_per_ep.append(fid)
    solved_per_ep.append(int(fid >= env.threshold))
    scramble_per_ep.append(env.scramble)
    ep_at_current_level += 1

    # Track best non-trivial solve (require at least scramble-many presses
    # so we don't celebrate identity solutions on already-solved boards)
    nontrivial = (fid >= env.threshold) and (len(actions_taken) >= max(1, env.scramble))
    if nontrivial and fid > best_fid_overall:
        best_fid_overall = fid
        best_actions = list(actions_taken)
        best_boards = [b.copy() for b in boards]
        best_fid_t = list(fid_t)

    epsilon = max(epsilon * eps_decay, eps_min)

    # Curriculum promotion
    if (env.scramble < curriculum_max and
            ep_at_current_level >= curriculum_min_eps_at_level and
            len(solved_per_ep) >= curriculum_window):
        recent_solve_rate = np.mean(solved_per_ep[-curriculum_window:])
        if recent_solve_rate >= curriculum_threshold:
            env.scramble += 1
            ep_at_current_level = 0
            epsilon = max(epsilon, 0.3)   # bump exploration on harder level
            print(f">>> curriculum: scramble -> {env.scramble} at ep {episode} "
                  f"(solve_rate={recent_solve_rate:.2f})")

    if episode % 50 == 0:
        recent = fid_per_ep[-50:]
        solved = sum(solved_per_ep[-50:])
        print(f"ep {episode:4d}  eps={epsilon:.3f}  K={env.scramble}  "
              f"last_fid={fid:.3f}  best={best_fid_overall:.3f}  "
              f"mean50={np.mean(recent):.3f}  solved50={solved}/{len(recent)}")

print(f"\nFinished. Best non-trivial fidelity: {best_fid_overall:.3f}")
if best_actions is None:
    # never produced a non-trivial solve; fall back to the last episode for plots
    print("(no non-trivial solve seen; plotting last episode instead)")
    best_actions = list(actions_taken)
    best_boards = [b.copy() for b in boards]
    best_fid_t = list(fid_t)
    best_fid_overall = fid
print(f"Best action sequence ({len(best_actions)} presses): {best_actions}")

# ---- plots ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))

# Learning curve, with scramble-level on a twin axis
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
plt.savefig(os.path.join(here, "learning_curve.png"), dpi=120, bbox_inches="tight")

# Loss
plt.figure()
plt.plot(loss_conv)
plt.xlabel("update step")
plt.ylabel("loss")
plt.yscale("log")
plt.savefig(os.path.join(here, "loss.png"), dpi=120, bbox_inches="tight")

# Best-episode trajectory: fidelity vs step + the press indices on a twin axis
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
plt.savefig(os.path.join(here, "trajectory.png"), dpi=120, bbox_inches="tight")

# Board snapshots from best episode
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
plt.savefig(os.path.join(here, "solved.png"), dpi=120, bbox_inches="tight")

print("Saved learning_curve.png, loss.png, trajectory.png, solved.png")
