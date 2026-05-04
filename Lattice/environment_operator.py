
import os
import numpy as np
from scipy.linalg import expm, eigh
from scipy.io import loadmat

# Use the same Hamiltonian pieces as the beam-splitter env so H(phi) is
# consistent across tasks: H(phi) = H0 + sin(phi)*H1 + (1 - cos(phi))*H2.
_here = os.path.dirname(os.path.abspath(__file__))
H0 = loadmat(os.path.join(_here, "H0.mat"))["H0"].astype(complex)
H1 = loadmat(os.path.join(_here, "H1.mat"))["H1"].astype(complex)
H2 = loadmat(os.path.join(_here, "H2.mat"))["H2"].astype(complex)
n_states = H0.shape[0]

# Bloch eigenstates of H0 (ascending energy). |0> = ground, |3> = 4th eigenstate.
_w, _v = eigh(H0)

class Env(object):
    def __init__(self):
        super(Env, self).__init__()
        self.actions = [0.4, 0.6, 0.8, 1.0, 1.2]
        self.n_actions = len(self.actions)
        self.n_states = n_states
        # Two channels for the mirror (phase-sensitive):
        #   ch1: |3> = 4th eigenstate of H0 -> +|3>
        #   ch2: |4> = 5th eigenstate of H0 -> -|4>
        self.psi1_in = _v[:, 3].astype(complex)
        self.psi1_out = _v[:, 3].astype(complex)
        self.psi2_in = _v[:, 4].astype(complex)
        self.psi2_out = -_v[:, 4].astype(complex)
        self.psi1 = self.psi1_in.copy()
        self.psi2 = self.psi2_in.copy()
        self.Ut = np.identity(n_states, dtype=complex)
        # state = concatenation of |psi1|^2 and |psi2|^2 (length 2*n_states)
        self.state = np.concatenate([np.abs(self.psi1)**2, np.abs(self.psi2)**2])
        self.n_obs = 2 * n_states
        self.max_time = 11*np.pi/12
        self.n_steps = 11
        self.n_substeps = 50  # sub-steps per decision interval for time-dependent phi
        self.t = 0
        self.threshold = 0.99

    def reset(self):
        self.Ut = np.identity(n_states, dtype=complex)
        self.psi1 = self.psi1_in.copy()
        self.psi2 = self.psi2_in.copy()
        self.state = np.concatenate([np.abs(self.psi1)**2, np.abs(self.psi2)**2])
        self.t = 0

        return self.state

    def step(self, action):
        amp = self.actions[action]
        dt_interval = self.max_time/self.n_steps
        n_sub = self.n_substeps
        dt_sub = dt_interval/n_sub
        t0 = self.t * dt_interval
        for i in range(n_sub):
            t_mid = t0 + (i + 0.5) * dt_sub
            phi = amp * np.sin(12.0 * t_mid)
            H = H0 + np.sin(phi)*H1 + (1 - np.cos(phi))*H2
            dU = expm(-1j * H * dt_sub)
            self.Ut = dU.dot(self.Ut)
            self.psi1 = dU.dot(self.psi1)
            self.psi2 = dU.dot(self.psi2)

        # Phase-sensitive average channel fidelity on the 2D subspace
        # F = (d + |sum_k <psi_k_out|U|psi_k_in>|^2) / (d(d+1)), d=2.
        a1 = np.vdot(self.psi1_out, self.psi1)
        a2 = np.vdot(self.psi2_out, self.psi2)
        fid = (2.0 + np.abs(a1 + a2)**2) / 6.0
################################################################
        #reward
        self.t += 1
        done = ( fid > self.threshold or self.t >= self.n_steps )
        rwd = (done)*(fid > 0.2)*(fid - 0.2)/(1-fid)
        if done:
            print(fid)
        if fid > self.threshold:
            print(self.Ut)

        self.state = np.concatenate([np.abs(self.psi1)**2, np.abs(self.psi2)**2])

        return self.state, rwd, done, fid




