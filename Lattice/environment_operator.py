
import numpy as np
from scipy.linalg import expm, eig

n_states = 5
V0 = 5.
p = np.linspace(-4, 4, n_states)
H0 = np.diag(p**2/2.)
H1 = np.zeros([n_states,n_states], dtype=complex) #sin
H1[0:n_states-1, 1:n_states] += V0/4*1j*np.identity(n_states-1);
H1[1:n_states, 0:n_states-1] -= V0/4*1j*np.identity(n_states-1);
H2 = np.zeros([n_states,n_states], dtype=complex) #cos
H2[0:n_states-1, 1:n_states] += V0/4*np.identity(n_states-1);
H2[1:n_states, 0:n_states-1] += V0/4*np.identity(n_states-1);

class Env(object):
    def __init__(self):
        super(Env, self).__init__()
        self.n_actions = 5
        self.actions = [-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2]
        self.n_states = n_states
        self.Ut = np.identity(n_states, dtype=complex) #expm(-1j*(H0-H2)*1.55*2)
        self.state = np.fliplr(self.Ut).diagonal().real
        self.max_time = 10.
        self.n_steps = 50
        self.t = 0
        self.threshold = 0.94
        
    def reset(self):
        self.Ut = np.identity(n_states, dtype=complex) #expm(-1j*(H0-H2)*1.55*2)
        self.state = np.fliplr(self.Ut).diagonal().real
        self.t = 0

        return self.state

    def step(self, action):
        phi = self.actions[action];
        H = H0 + np.sin(phi)*H1 + (1 - np.cos(phi))*H2
        dt = self.max_time/self.n_steps
        dU = expm(-1j * H * dt)  # Evolution operator

        self.Ut = dU.dot(self.Ut)
        #fid = np.fliplr(self.Ut).diagonal().sum().real/self.n_states
        #fid = np.square(np.abs(np.fliplr(self.Ut).diagonal())).sum()/self.n_states
        fid = (self.n_states + np.square(np.abs(np.fliplr(self.Ut).diagonal().sum())))/(self.n_states*(self.n_states+1)) #channel fidelity
################################################################
        #reward
        done =( fid > self.threshold or self.t > self.n_steps )
        rwd = (done)*(fid > 0.2)*(fid - 0.2)/(1-fid)
        if done:
            print(fid)
        if fid > self.threshold:
            print(self.Ut)
        self.t +=1

        self.state = np.fliplr(self.Ut).diagonal().real

        return self.state, rwd, done, fid




