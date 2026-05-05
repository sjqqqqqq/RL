"""Goal-conditioned Lights Out 5x5 for HER+DQN.

Difference from environment_lightsout.py:
  - Observation = [board (25) || goal (25)] instead of [board || pressed].
    The goal is exposed to the Q-net so HER can relabel it to a board the
    agent actually reached and recompute the reward without ambiguity.
  - Reward is sparse {0, 1} on `board == goal`. No shaping. HER turns the
    all-zeros bottleneck into a dense supervisory signal by relabeling.
  - `pressed` still exists internally to drive the action mask (each cell
    pressable at most once, as in variant 4), but is not in the obs.
"""
import numpy as np


N = 5
N_CELLS = N * N


def _neighbor_mask(i):
    m = np.zeros(N_CELLS, dtype=np.int8)
    r, c = divmod(i, N)
    m[i] = 1
    if r > 0:     m[(r - 1) * N + c] = 1
    if r < N - 1: m[(r + 1) * N + c] = 1
    if c > 0:     m[r * N + (c - 1)] = 1
    if c < N - 1: m[r * N + (c + 1)] = 1
    return m


_TOGGLES = np.stack([_neighbor_mask(i) for i in range(N_CELLS)], axis=0)


class Env:
    def __init__(self, scramble=1, n_steps=25, seed=None):
        self.n_actions = N_CELLS
        self.actions = list(range(N_CELLS))
        self.n_states = 2 * N_CELLS          # board || goal
        self.scramble = scramble
        self.n_steps = n_steps
        self.max_time = float(n_steps)
        self.rng = np.random.default_rng(seed)

        self.board = np.zeros(N_CELLS, dtype=np.int8)
        self.goal = np.zeros(N_CELLS, dtype=np.int8)
        self.pressed = np.zeros(N_CELLS, dtype=np.int8)
        self.t = 0
        self.state = self._encode()

    def _encode(self):
        return np.concatenate([self.board, self.goal]).astype(np.float32)

    def _fidelity(self):
        return float(np.mean(self.board == self.goal))

    def valid_action_mask(self):
        return (1 - self.pressed).astype(np.float32)

    def reset(self):
        self.goal = np.zeros(N_CELLS, dtype=np.int8)
        self.board = np.zeros(N_CELLS, dtype=np.int8)
        self.pressed = np.zeros(N_CELLS, dtype=np.int8)
        k = min(self.scramble, N_CELLS)
        presses = self.rng.choice(N_CELLS, size=k, replace=False)
        for p in presses:
            self.board ^= _TOGGLES[p]
        self.t = 0
        self.state = self._encode()
        return self.state

    def step(self, action):
        a = int(action)
        self.board ^= _TOGGLES[a]
        self.pressed[a] = 1
        self.t += 1
        achieved = bool(np.array_equal(self.board, self.goal))
        no_actions_left = bool(self.pressed.all())
        done = bool(achieved or no_actions_left or self.t >= self.n_steps)
        reward = 1.0 if achieved else 0.0
        self.state = self._encode()
        return self.state, reward, done, self._fidelity()
