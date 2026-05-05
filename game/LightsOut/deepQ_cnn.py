"""CNN Q-network for Lights Out 5x5.

Same training scheme as deepQ_lightsout.py (variant 3 of first_try.md):
  - target net + Polyak soft updates (tau=0.99)
  - curriculum starting at scramble=1, promote at >=0.8 solve rate
  - variant-3 reward (Δfid shaping + (F-0.2)/(1-F) terminal bonus)
  - epsilon-greedy with exponential decay
  - uniform replay
  - NO action masking (variant 3 didn't use it)

The only thing that changes is the Q-network: instead of an MLP on the
flat 50-dim [board || pressed] vector, we use a CNN on the 5x5 board
grid. Lights Out is translation-equivariant on the grid (away from
boundaries), so parameter sharing across cells is the natural
inductive bias.

This variant pushes on first_try.md's "what to try next": deeper net
(6 conv layers vs variant 7's 3) and 2x training budget (16k eps).
The 3-layer CNN already had a global receptive field; adding depth
buys *multi-hop* reasoning over the XOR transition graph, which is
what GF(2) elimination actually requires.

We feed only the board (1 channel, 5x5). The `pressed` half of the env
state is unused here. Press history is recoverable from the board for
this task, and parity-state hurt in variant 5.
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

learning_rate = 1e-3
gamma = 0.95
epsilon = 1.0
eps_min = 0.05
eps_decay = 0.9990
tau = 0.99
batch_size = 64
buffer_cap = 50_000
max_episodes = 16000

curriculum_max = 15
curriculum_window = 100
curriculum_threshold = 0.80
curriculum_min_eps_at_level = 200


class CNNQ(nn.Module):
    """5x5 board -> 25 Q-values via stacked 3x3 convs.

    Six same-padded 3x3 convs give a deep, globally-connected stack: each
    extra layer adds another hop of "what does pressing here imply about
    that cell" reasoning, the kind of multi-hop chaining variant 7's
    3-layer net plateaued on. Final 1x1 conv projects to one Q-value per
    cell, flattened to a length-25 action head.
    """
    def __init__(self, channels=32, n_layers=6):
        super().__init__()
        c = channels
        layers = [nn.Conv2d(1, c, kernel_size=3, padding=1), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Conv2d(c, c, kernel_size=3, padding=1), nn.ReLU()]
        layers += [nn.Conv2d(c, 1, kernel_size=1)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x_flat):
        # x_flat: (B, 50) [board || pressed]; we use only the board half.
        b = x_flat[..., :N * N].view(-1, 1, N, N)
        q = self.net(b)                          # (B, 1, 5, 5)
        return q.view(-1, N * N)                 # (B, 25)

    def act(self, state, eps):
        if random.random() > eps:
            s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q = self.forward(s).squeeze(0).numpy()
            return int(np.argmax(q))
        return int(np.random.randint(env.n_actions))


model = CNNQ()
target = CNNQ()
target.load_state_dict(model.state_dict())
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def calc_loss():
    s, a, r, s2, d = zip(*random.sample(replay_buffer, batch_size))
    s = torch.as_tensor(np.array(s), dtype=torch.float32)
    s2 = torch.as_tensor(np.array(s2), dtype=torch.float32)
    a = torch.as_tensor(a, dtype=torch.long)
    r = torch.as_tensor(r, dtype=torch.float32)
    d = torch.as_tensor(d, dtype=torch.float32)

    q = model(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target(s2).max(1)[0]
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
        a = model.act(state, epsilon)
        next_state, reward, done, fid = env.step(a, fid)
        replay_buffer.append((state, a, reward, next_state, float(done)))
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

    nontrivial = (fid >= env.threshold) and (len(actions_taken) >= max(1, env.scramble))
    if nontrivial and fid > best_fid_overall:
        best_fid_overall = fid
        best_actions = list(actions_taken)
        best_boards = [b.copy() for b in boards]
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
        solved = sum(solved_per_ep[-50:])
        print(f"ep {episode:4d}  eps={epsilon:.3f}  K={env.scramble}  "
              f"last_fid={fid:.3f}  best={best_fid_overall:.3f}  "
              f"mean50={np.mean(recent):.3f}  solved50={solved}/{len(recent)}")

print(f"\nFinished. Best non-trivial fidelity: {best_fid_overall:.3f}")
if best_actions is None:
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
plt.savefig(os.path.join(here, "learning_curve_cnn.png"), dpi=120, bbox_inches="tight")

plt.figure()
plt.plot(loss_conv)
plt.xlabel("update step")
plt.ylabel("loss")
plt.yscale("log")
plt.savefig(os.path.join(here, "loss_cnn.png"), dpi=120, bbox_inches="tight")

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
plt.savefig(os.path.join(here, "trajectory_cnn.png"), dpi=120, bbox_inches="tight")

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
plt.savefig(os.path.join(here, "solved_cnn.png"), dpi=120, bbox_inches="tight")

print("Saved learning_curve_cnn.png, loss_cnn.png, trajectory_cnn.png, solved_cnn.png")
