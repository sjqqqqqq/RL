import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from environment_operator import Env
import random

env = Env()
#Hyperparameters
learning_rate = 0.0001
gamma = 0.999
epsilon = 0.5
tau = 0.995
batch_size = 500
max_episodes = 3000

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.state_space = env.n_states
        self.action_space = env.n_actions
        self.hidden_size = 48
  
        self.l1 = nn.Linear(self.state_space, self.hidden_size, bias=True)
        self.l2 = nn.Linear(self.hidden_size, self.action_space, bias=True)
        nn.init.xavier_uniform_(self.l1.weight)
        nn.init.xavier_uniform_(self.l2.weight)
      
    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            nn.ReLU(),
            self.l2,
        )
        return model(x)
  
    def act(self, state):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(state).unsqueeze(0))
            q_value = self.forward(state)
            action  = q_value.max(1)[1]
        else:
            action = random.randrange(env.n_actions)
        return action
  

model = DQN()
#load parameters from a pre-trained model
#model = torch.load("params")
#model.eval()
"""
#for expanding the input dimension
small_model = torch.load("params")
#initialize weights
nn.init.zeros_(model.l1.weight)
model.l1.weight.data[:,1:4] = small_model.l1.weight.data

model.l1.bias.data = small_model.l1.bias.data
model.l2.weight.data = small_model.l2.weight.data
model.l2.bias.data = small_model.l2.bias.data
"""

target = DQN()
for target_param, param in zip(target.parameters(), model.parameters()):
    target_param.data.copy_(param.data)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#scheduler = StepLR(optimizer, step_size=1, gamma=0.99)


def calc_loss_over_episode(batch_size):
    state, action, reward, next_state, done = zip(*random.sample(replay_buffer, batch_size))
    state = Variable(torch.FloatTensor(state))
    next_state = Variable(torch.FloatTensor(next_state))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))
    q_values = model(state)
    next_q_values = target(next_state).detach()
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    loss = (q_value - Variable(expected_q_value)).pow(2).mean()
    loss_conv.append(loss.item())
    return loss

replay_buffer = []
loss_conv = []
best_fid_overall = -1.0
best_phi_t = []
best_fid_t = []
best_fid_per_ep = []

for episode in range(max_episodes):
    env.reset()
    state = env.state
    phi_t = []
    done = False
    fid = 1./env.n_states
    fid_t = [fid]
    while not done:
        action = model.act(state)
        next_state, reward, done, fid = env.step(action)
        replay_buffer.append((state, action, reward, next_state, done))
        
        fid_t.append(fid)
        phi_t.append(env.actions[action])
        state = next_state
        if len(replay_buffer) >= batch_size:
            loss = calc_loss_over_episode(batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #if (len(replay_buffer) > batch_size*10):
            #    del replay_buffer[0]
        for target_param, param in zip(target.parameters(), model.parameters()):
            target_param.data.copy_(target_param.data * tau + param.data * (1.0 - tau))
    best_fid_per_ep.append(fid)
    if fid > best_fid_overall:
        best_fid_overall = fid
        best_phi_t = list(phi_t)
        best_fid_t = list(fid_t)
    if episode % 10 == 0:
        recent = best_fid_per_ep[-10:]
        print(f"ep {episode:4d}  eps={epsilon:.3f}  last_fid={fid:.3f}  "
              f"best={best_fid_overall:.3f}  mean_last_10={np.mean(recent):.3f}",
              flush=True)
    if fid > env.threshold:
        print(f"Solved at episode {episode}, fidelity {fid:.3f}", flush=True)
        break

    epsilon = max(epsilon * 0.995, 0.01)
    #if len(replay_buffer) >= batch_size:
    #    scheduler.step()

#torch.save(model, "params")
print(f"\nFinished. Best fidelity: {best_fid_overall:.3f}", flush=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fid_t = best_fid_t
phi_t = best_phi_t
N = len(fid_t)-1
dt = env.max_time/env.n_steps
t = np.arange(0, (dt*N+dt)/2., dt/2.) #in units of recoil frequency inverse
tt = [0]
for i in range(1,N):
    tt.append(i*dt/2.)
    tt.append(i*dt/2.)
tt.append(N*dt/2.)
phi_tt = []
for i in range(N):
    phi_tt.append(phi_t[i])
    phi_tt.append(phi_t[i])
plt.plot(t, fid_t, label='Fidelity')
plt.plot(tt, phi_tt, label='$\phi$')
plt.xlabel('$t\ (\omega_r^{-1})$')
plt.legend(loc="upper left")
plt.title(f"Best episode (F={best_fid_overall:.3f})")
plt.savefig("trajectory_mirror.png", dpi=120, bbox_inches="tight")
plt.figure()
plt.plot(best_fid_per_ep, alpha=0.3, label="per-episode terminal fidelity")
window = 10
if len(best_fid_per_ep) >= window:
    smooth = np.convolve(best_fid_per_ep, np.ones(window)/window, mode="valid")
    plt.plot(np.arange(window-1, len(best_fid_per_ep)), smooth,
             label=f"{window}-episode rolling mean")
plt.xlabel("episode")
plt.ylabel("terminal fidelity")
plt.legend()
plt.savefig("learning_curve_mirror.png", dpi=120, bbox_inches="tight")
plt.figure()
plt.plot(loss_conv)
plt.xlabel("update step")
plt.ylabel("loss")
plt.yscale("log")
plt.savefig("loss_mirror.png", dpi=120, bbox_inches="tight")
print("Saved trajectory_mirror.png, learning_curve_mirror.png, loss_mirror.png", flush=True)

