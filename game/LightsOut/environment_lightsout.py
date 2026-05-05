"""Lights Out 5x5 as an RL environment.

Mirrors the API of Lattice/environment_quantum_state.py so the same
DQN-with-target-network training loop can be used unchanged:
    n_states, n_actions, actions, state, max_time, n_steps, threshold,
    reset(), step(action, fid0)

State : 25 binary cells flattened to a length-25 float vector.
Action: index in [0, 24]; pressing cell i toggles cell i and its
        up/down/left/right neighbors.
Target: all-off (zeros).
Reward: sparse, terminal-only -- same shape as the lattice tasks,
        rwd = done * (F > 0.2) * (F - 0.2) / (1 - F + eps).
        Capped at F just below 1 to keep the reward finite.

Scrambling: starting from the off board, apply `scramble` random
presses. This guarantees the start state is solvable in <= scramble
moves, giving the agent a curriculum knob.
"""
import numpy as np


N = 5
N_CELLS = N * N


def _neighbor_mask(i):
    """Bitmask-style toggle vector for pressing cell i on an N x N board."""
    m = np.zeros(N_CELLS, dtype=np.int8)
    r, c = divmod(i, N)
    m[i] = 1
    if r > 0:     m[(r - 1) * N + c] = 1
    if r < N - 1: m[(r + 1) * N + c] = 1
    if c > 0:     m[r * N + (c - 1)] = 1
    if c < N - 1: m[r * N + (c + 1)] = 1
    return m


# Precompute the 25 toggle vectors once.
_TOGGLES = np.stack([_neighbor_mask(i) for i in range(N_CELLS)], axis=0)


class Env:
    def __init__(self, scramble=6, n_steps=25, threshold=0.999, seed=None):
        self.n_actions = N_CELLS
        self.actions = list(range(N_CELLS))   # action labels (used only for logging/plots)
        # State = [board (25)] || [pressed parity (25)] -- gives the Q-net both
        # the current configuration AND its own commit history, so two boards
        # reached via different press subsets can be valued differently.
        self.n_states = 2 * N_CELLS
        self.scramble = scramble
        self.n_steps = n_steps
        self.max_time = float(n_steps)        # nominal "time" axis for plotting
        self.threshold = threshold
        self.rng = np.random.default_rng(seed)

        self.board = np.zeros(N_CELLS, dtype=np.int8)
        self.pressed = np.zeros(N_CELLS, dtype=np.int8)
        self.t = 0
        self.state = self._encode()

    def _encode(self):
        return np.concatenate([self.board, self.pressed]).astype(np.float32)

    def _fidelity(self):
        # fraction of cells matching the all-off target
        return float(np.mean(self.board == 0))

    def valid_action_mask(self):
        # 1 where action is allowed (cell not yet pressed), 0 where forbidden.
        return (1 - self.pressed).astype(np.float32)

    def reset(self):
        self.board = np.zeros(N_CELLS, dtype=np.int8)
        self.pressed = np.zeros(N_CELLS, dtype=np.int8)
        # apply `scramble` random unique presses (sample without replacement)
        # -- since presses commute and self-invert, distinct cells give a
        # uniformly drawn solvable scramble of exactly `scramble` parity-1 bits.
        k = min(self.scramble, N_CELLS)
        presses = self.rng.choice(N_CELLS, size=k, replace=False)
        for p in presses:
            self.board ^= _TOGGLES[p]
        self.t = 0
        self.state = self._encode()
        return self.state

    def step(self, action, fid0):
        a = int(action)
        self.board ^= _TOGGLES[a]
        self.pressed[a] = 1
        self.t += 1
        fid = self._fidelity()
        # episode ends if solved, all cells pressed (no more legal actions),
        # or step budget exhausted
        no_actions_left = bool(self.pressed.all())
        done = bool(fid >= self.threshold or no_actions_left or self.t >= self.n_steps)

        # Dense reward: per-step potential-based shaping (fid - fid0) plus a
        # terminal bonus of the lattice-style F/(1-F) shape on success.
        # The shaping term telescopes to (fid_final - fid_start) over an episode,
        # so it does not change the optimum but gives a per-step gradient signal.
        f_clip = min(fid, 0.9999)
        terminal_bonus = float(done) * float(fid > 0.2) * (f_clip - 0.2) / (1.0 - f_clip)
        rwd = (fid - fid0) + terminal_bonus

        self.state = self._encode()
        return self.state, rwd, done, fid
