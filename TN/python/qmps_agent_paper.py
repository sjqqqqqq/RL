"""QMPS-based DQN agent.

Q-function: features = QMPS overlap with current state, then a small MLP head
maps the d_f-dim feature vector to a Q-value per action.

Mirrors the structure of Lattice/deepQ_target.py's DQN class but with the QMPS
feature extractor in place of the input linear layer.
"""
from __future__ import annotations

import random
from copy import deepcopy

import bridge_paper as B  # imports juliacall — must precede torch

import torch
import torch.nn as nn


class QMPSDQN(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        # Paper Tab. I: 2 hidden layers, 100-200 neurons each, tanh.
        # QMPS params live as a flat real Parameter; sync to Julia on each call.
        init = B.get_qmps_params()                  # init from Julia's QMPSRL
        self.qmps_params = nn.Parameter(init.clone())
        self.head = nn.Sequential(
            nn.Linear(B.D_F, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, B.N_ACTIONS),
        )
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

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


def polyak_update(target: QMPSDQN, online: QMPSDQN, tau: float) -> None:
    """target ← τ·target + (1−τ)·online, both qmps_params and head."""
    with torch.no_grad():
        for tp, p in zip(target.parameters(), online.parameters()):
            tp.data.mul_(tau).add_(p.data, alpha=1.0 - tau)


def hard_copy(target: QMPSDQN, online: QMPSDQN) -> None:
    """target ← online (paper's `n_target` hard update)."""
    with torch.no_grad():
        for tp, p in zip(target.parameters(), online.parameters()):
            tp.data.copy_(p.data)
