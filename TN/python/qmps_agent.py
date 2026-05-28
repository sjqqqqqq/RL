"""QMPS-based DQN agent (Paper Study A defaults).

Q-function: features = QMPS overlap with current state, then a small MLP head
maps the D_F-dim feature vector to a Q-value per action.
"""
from __future__ import annotations

import random
from copy import deepcopy

import bridge as B  # imports juliacall — must precede torch

import torch
import torch.nn as nn


class QMPSDQN(nn.Module):
    """Paper Tab. I: 2 hidden layers, 100 neurons each, tanh activations.
    NN weights/biases initialized N(0, 0.1²) per QMPS/dqn/models_utils.py."""

    def __init__(self, hidden: int = 100):
        super().__init__()
        init = B.get_qmps_params()                  # init from Julia's QMPSRL
        self.qmps_params = nn.Parameter(init.clone())
        self.head = nn.Sequential(
            nn.Linear(B.D_F,  hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, B.N_ACTIONS),
        )
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                nn.init.normal_(m.bias,   mean=0.0, std=0.1)

    def forward(self, state_id: int) -> torch.Tensor:
        feat = B.feature(int(state_id), self.qmps_params)
        return self.head(feat)

    def forward_batch(self, state_ids) -> torch.Tensor:
        """state_ids: sequence of ints. Returns (B, N_ACTIONS) Q-values."""
        feats = B.feature_batch(state_ids, self.qmps_params)   # (B, D_F)
        return self.head(feats)

    @torch.no_grad()
    def act(self, state_id: int, eps: float) -> int:
        if random.random() < eps:
            return random.randrange(B.N_ACTIONS)
        q = self.forward(state_id)
        return int(q.argmax().item())


def clone_for_target(model: QMPSDQN) -> QMPSDQN:
    """Make a target-network copy with identical weights but no gradient tracking."""
    tgt = deepcopy(model)
    for p in tgt.parameters():
        p.requires_grad_(False)
    return tgt


def hard_copy(target: QMPSDQN, online: QMPSDQN) -> None:
    """target ← online (paper's `n_target` hard update)."""
    with torch.no_grad():
        for tp, p in zip(target.parameters(), online.parameters()):
            tp.data.copy_(p.data)
