"""Replay buffer with ref-counted Julia state handle eviction.

Each transition is (state_id, action, reward, next_state_id, done). state_ids
are integer handles into Julia's QMPSRL._registry. We maintain a Python-side
ref count; when an id's count drops to 0 it is forgotten on the Julia side so
the underlying MPS object becomes garbage.
"""
from __future__ import annotations

import random
from collections import defaultdict, deque
from typing import Deque, NamedTuple

import bridge_paper as B


class Transition(NamedTuple):
    s: int
    a: int
    r: float
    s2: int
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: Deque[Transition] = deque(maxlen=capacity)
        self.refcount: "defaultdict[int, int]" = defaultdict(int)

    def __len__(self) -> int:
        return len(self.buf)

    def push(self, tr: Transition) -> None:
        # If buffer is at capacity, the leftmost entry will be silently dropped
        # by deque.append. Handle eviction explicitly so we can update refcounts.
        if self.buf.maxlen is not None and len(self.buf) == self.buf.maxlen:
            old = self.buf.popleft()
            self._decref(old.s)
            self._decref(old.s2)
        self.buf.append(tr)
        self._incref(tr.s)
        self._incref(tr.s2)

    def sample(self, batch_size: int):
        return random.sample(self.buf, batch_size)

    def _incref(self, sid: int) -> None:
        self.refcount[sid] += 1

    def _decref(self, sid: int) -> None:
        c = self.refcount[sid] - 1
        if c <= 0:
            del self.refcount[sid]
            B.forget_state(sid)
        else:
            self.refcount[sid] = c
