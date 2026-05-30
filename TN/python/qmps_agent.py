"""QMPS-based DQN agent (Paper Study B defaults).

Q-function: features = QMPS overlap with current state, then a small MLP head
maps the D_F-dim feature vector to a Q-value per action.
"""
from __future__ import annotations

import random
from copy import deepcopy

import bridge as B  # imports juliacall — must precede torch

import torch
import torch.nn as nn

import qmps_torch as QT  # pure-torch QMPS contraction (autograd backward)


class QMPSDQN(nn.Module):
    """Paper Tab. I: 2 hidden layers, tanh activations (Study A: 100 wide,
    Study B: 200). NN weights/biases initialized N(0, 0.1²) per
    QMPS/dqn/models_utils.py.

    The QMPS feature contraction runs in torch (qmps_torch), so gradients flow
    via torch.autograd — no Julia/Zygote in the gradient path. Julia owns only
    the environment dynamics; per-state tensors are cached in qmps_torch."""

    def __init__(self, hidden: int = 200):
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

        # Optional CUDA-graphed contraction for a fixed batch size (see
        # enable_cuda_graph). None → eager. Single-state (act) always eager.
        self._graph_bs: int | None = None
        self._graphed_feature = None

    def enable_cuda_graph(self, sample_state_ids) -> None:
        """Capture a CUDA graph of the QMPS contraction at this batch size.

        Must run after .to('cuda') and after the replay buffer holds real state
        ids (the sample is used only to size/warm the capture). Reused for both
        grad and no-grad forwards at the matching batch size."""
        p = self.qmps_params
        had_grad = p.requires_grad
        p.requires_grad_(True)                       # capture needs the bwd graph
        self._graph_bs = len(sample_state_ids)
        self._graphed_feature = QT.graph_feature(p, sample_state_ids)
        if not had_grad:                             # target net: restore no-grad
            p.grad = None
            p.requires_grad_(False)

    def _feat(self, state_ids) -> torch.Tensor:
        if self._graphed_feature is not None and len(state_ids) == self._graph_bs:
            psi = QT.psi_batch(state_ids)
            return self._graphed_feature(self.qmps_params, *psi)
        return QT.feature_batch_ids(state_ids, self.qmps_params)

    def forward(self, state_id: int) -> torch.Tensor:
        return self.head(self._feat([int(state_id)]))[0]

    def forward_batch(self, state_ids) -> torch.Tensor:
        """state_ids: sequence of ints. Returns (B, N_ACTIONS) Q-values."""
        return self.head(self._feat(state_ids))

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
