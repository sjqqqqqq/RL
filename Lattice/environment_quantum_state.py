
import numpy as np
from scipy.linalg import expm, eig

n_states = 7
V0 = 5.
p = np.linspace(-6, 6, n_states)
H0 = np.diag(p**2/2.)
H1 = np.zeros([n_states,n_states], dtype=complex) #sin
H1[0:n_states-1, 1:n_states] += V0/4*1j*np.identity(n_states-1);
H1[1:n_states, 0:n_states-1] -= V0/4*1j*np.identity(n_states-1);
H2 = np.zeros([n_states,n_states], dtype=complex) #cos
H2[0:n_states-1, 1:n_states] += V0/4*np.identity(n_states-1);
H2[1:n_states, 0:n_states-1] += V0/4*np.identity(n_states-1);
w, v = eig(H0 - H2)

class Env(object):
    def __init__(self):
        super(Env, self).__init__()
        self.n_actions = 5
        self.actions = [-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2]
        self.n_states = n_states
        self.psi = v[:,0]
        self.state = np.square(np.abs(self.psi))#(np.conjugate(v[:,0]) * v[:,0]).real
        self.max_time = 4.
        self.n_steps = 40
        self.t = 0
        self.phi = 0
        self.target = v[:,1]
        self.threshold = 0.95
    def reset(self):
        self.psi = v[:,0]
        self.state = np.abs(self.psi)**2
        self.t = 0
        self.phi = 0

        return self.state

    def step(self, action, fid0):
        
        self.phi = self.actions[action]
        H =  H0 + np.sin(self.phi)*H1 - np.cos(self.phi)*H2;
        dt = self.max_time/self.n_steps
        U = expm(-1j * H * dt)  # Evolution operator

        self.psi = U.dot(self.psi)
        fid = np.abs(np.conjugate(self.psi).dot(self.target)) ** 2  # infidelity (to make it as small as possible)
################################################################
        #reward
        #rwd = (((fid - fid0)>0)*00 + fid + (fid>0.4)*10*fid + (fid>0.7)*100*fid + (fid>0.9)*1000*fid + (fid>0.95)*000*fid)*0.02
        done = (fid > self.threshold or self.t > self.n_steps )
        rwd = done*fid/(1-fid)
        if done:
            print(fid)
        self.t +=1  # step counter add one

        self.state = (np.conjugate(self.psi) * self.psi).real#np.concatenate((self.psi.real, self.psi.imag))

        return self.state, rwd, done, fid
