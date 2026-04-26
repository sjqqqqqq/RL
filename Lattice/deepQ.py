import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from environment_quantum_state import Env
import random

env = Env()
#Hyperparameters
learning_rate = 0.0001
gamma = 0.995
epsilon = 0.3
batch_size = 400
max_episodes = 200

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.state_space = env.n_states
        self.action_space = env.n_actions
        self.hidden_size = 32
  
        self.l1 = nn.Linear(self.state_space, self.hidden_size, bias=True)
        self.l2 = nn.Linear(self.hidden_size, self.action_space, bias=True)
        nn.init.xavier_uniform_(self.l1.weight)
        nn.init.xavier_uniform_(self.l2.weight)
      
    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            nn.ReLU(),
            self.l2,
            nn.ReLU()
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

"""
#for expanding the input dimension
small_model = torch.load("params")
#initialize weights
nn.init.zeros_(model.l1.weight)
model.l1.weight.data[:,1:6] = small_model.l1.weight.data

model.l1.bias.data = small_model.l1.bias.data
model.l2.weight.data = small_model.l2.weight.data
model.l2.bias.data = small_model.l2.bias.data
"""

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def calc_loss_over_episode(batch_size):
    state, action, reward, next_state, done = zip(*random.sample(replay_buffer, batch_size))
    state = Variable(torch.FloatTensor(state))
    next_state = Variable(torch.FloatTensor(next_state))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))
    q_values = model(state)
    next_q_values = model(next_state)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    loss = (q_value - Variable(expected_q_value)).pow(2).mean()
    loss_conv.append(loss)
      
    return loss

replay_buffer = []
loss_conv = []

for episode in range(max_episodes):
    env.reset()
    state = env.state
    phi_t = []
    done = False
    fid = 0.
    fid_t = [fid]
    while not done:
        action = model.act(state)
        next_state, reward, done, fid = env.step(action, fid)
        replay_buffer.append((state, action, reward, next_state, done))
        
        fid_t.append(fid)
        phi_t.append(env.actions[action])
        state = next_state
        
        if len(replay_buffer) >= batch_size:
            loss = calc_loss_over_episode(batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            """
            #use adaptive learning rate when optimizer is turned off
            learning_rate *= 0.995
            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad
            for param in model.parameters():
                param.grad.zero_()
            """
            #if (len(replay_buffer) > batch_size):
            #    del replay_buffer[0]
    if fid > env.threshold:
        break
    
    epsilon *= 0.99
    
#torch.save(model, "params")

import matplotlib.pyplot as plt
N = len(fid_t)-1
dt = env.max_time/env.n_steps
t = np.arange(0, (dt*N+dt)/2., dt/2.) #plot in units of recoil frequency inverse
#plot step functions
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
plt.show()
plt.plot(loss_conv)
plt.show()
