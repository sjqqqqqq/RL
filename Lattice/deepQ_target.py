"""Splitter task (environment_quantum_state.Env) with target-network DQN.

Same as deepQ.py but with the two stability tricks from double_deepQ.py:
  - separate target network updated by Polyak averaging
  - target network used for the bootstrap term in the Bellman loss

Also tracks best fidelity per episode so we can see a learning curve.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from environment_quantum_state import Env

env = Env()
# Hyperparameters
learning_rate = 0.0001
gamma = 0.999
epsilon = 0.5
tau = 0.995          # target network soft-update rate
batch_size = 400
max_episodes = 3000


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(env.n_states, 32, bias=True)
        self.l2 = nn.Linear(32, env.n_actions, bias=True)
        nn.init.xavier_uniform_(self.l1.weight)
        nn.init.xavier_uniform_(self.l2.weight)

    def forward(self, x):
        h = torch.relu(self.l1(x))
        return torch.relu(self.l2(h))

    def act(self, state, eps):
        if random.random() > eps:
            s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                return int(self.forward(s).argmax(1).item())
        return random.randrange(env.n_actions)


model = DQN()
target = DQN()
target.load_state_dict(model.state_dict())
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def calc_loss(batch_size):
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


replay_buffer = []
loss_conv = []
best_fid_per_ep = []
best_fid_overall = -1.0
best_phi_t = None
best_fid_t = None

for episode in range(max_episodes):
    env.reset()
    state = env.state
    phi_t, fid_t = [], [0.0]
    fid = 0.0
    done = False
    while not done:
        action = model.act(state, epsilon)
        next_state, reward, done, fid = env.step(action, fid)
        replay_buffer.append((state, action, reward, next_state, done))
        fid_t.append(fid)
        phi_t.append(env.actions[action])
        state = next_state
        if len(replay_buffer) >= batch_size:
            loss = calc_loss(batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Polyak soft update of target net
            with torch.no_grad():
                for tp, p in zip(target.parameters(), model.parameters()):
                    tp.data.mul_(tau).add_(p.data, alpha=1.0 - tau)

    best_fid_per_ep.append(fid)
    if fid > best_fid_overall:
        best_fid_overall = fid
        best_phi_t = list(phi_t)
        best_fid_t = list(fid_t)

    epsilon = max(epsilon * 0.995, 0.01)

    if episode % 50 == 0:
        recent = best_fid_per_ep[-50:]
        print(f"ep {episode:4d}  eps={epsilon:.3f}  "
              f"last_fid={fid:.3f}  best={best_fid_overall:.3f}  "
              f"mean_last_50={np.mean(recent):.3f}")

    if best_fid_overall > env.threshold and episode > 100:
        print(f"Solved at episode {episode}, fidelity {best_fid_overall:.3f}")
        break

print(f"\nFinished. Best fidelity: {best_fid_overall:.3f}")

# ---- plots ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Best episode trajectory
N = len(best_fid_t) - 1
dt = env.max_time / env.n_steps
t = np.arange(0, (dt * N + dt) / 2.0, dt / 2.0)
tt = [0]
for i in range(1, N):
    tt += [i * dt / 2.0, i * dt / 2.0]
tt.append(N * dt / 2.0)
phi_tt = []
for i in range(N):
    phi_tt += [best_phi_t[i], best_phi_t[i]]

plt.figure()
plt.plot(t, best_fid_t, label="Fidelity")
plt.plot(tt, phi_tt, label=r"$\phi$")
plt.xlabel(r"$t\ (\omega_r^{-1})$")
plt.title(f"Best episode (F={best_fid_overall:.3f})")
plt.legend(loc="upper left")
plt.savefig("trajectory_target.png", dpi=120, bbox_inches="tight")

# Learning curve
plt.figure()
plt.plot(best_fid_per_ep, alpha=0.3, label="per-episode terminal fidelity")
window = 50
if len(best_fid_per_ep) >= window:
    smooth = np.convolve(best_fid_per_ep, np.ones(window) / window, mode="valid")
    plt.plot(np.arange(window - 1, len(best_fid_per_ep)), smooth,
             label=f"{window}-episode rolling mean")
plt.xlabel("episode")
plt.ylabel("terminal fidelity")
plt.legend()
plt.savefig("learning_curve_target.png", dpi=120, bbox_inches="tight")

# Loss
plt.figure()
plt.plot(loss_conv)
plt.xlabel("update step")
plt.ylabel("loss")
plt.yscale("log")
plt.savefig("loss_target.png", dpi=120, bbox_inches="tight")

print("Saved trajectory_target.png, learning_curve_target.png, loss_target.png")
