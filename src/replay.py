"""Simple episodic replay buffer for world-model training.

Stores full episodes as numpy arrays; samples fixed-length subsequences.
"""
from __future__ import annotations

import numpy as np
import random
from dataclasses import dataclass, field
from typing import List


@dataclass
class Episode:
    obs: np.ndarray       # (T+1, H, W, 3) uint8
    actions: np.ndarray   # (T,) int64
    rewards: np.ndarray   # (T,) float32
    dones: np.ndarray     # (T,) bool

    @property
    def length(self) -> int:
        return len(self.actions)


class ReplayBuffer:
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.episodes: List[Episode] = []

    def add(self, ep: Episode):
        self.episodes.append(ep)
        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)

    @property
    def num_transitions(self) -> int:
        return sum(ep.length for ep in self.episodes)

    def sample(self, batch_size: int, seq_len: int, rng: random.Random | None = None):
        """Return dict of tensors, each of shape (B, seq_len, ...)."""
        rng = rng or random
        # Only episodes long enough
        eligible = [ep for ep in self.episodes if ep.length >= seq_len]
        if not eligible:
            raise ValueError(f"No episodes of length >= {seq_len}")
        obs_b, act_b, rew_b, done_b = [], [], [], []
        for _ in range(batch_size):
            ep = rng.choice(eligible)
            start = rng.randint(0, ep.length - seq_len)
            end = start + seq_len
            obs_b.append(ep.obs[start:end + 1])       # seq_len+1 frames (include next)
            act_b.append(ep.actions[start:end])
            rew_b.append(ep.rewards[start:end])
            done_b.append(ep.dones[start:end])
        return {
            "obs": np.stack(obs_b),        # (B, seq_len+1, H, W, 3)
            "actions": np.stack(act_b),    # (B, seq_len)
            "rewards": np.stack(rew_b),    # (B, seq_len)
            "dones": np.stack(done_b),     # (B, seq_len)
        }


def collect_random_episode(env, seed: int | None = None) -> Episode:
    obs0 = env.reset(seed=seed)
    obs_list = [obs0]
    acts, rews, dones = [], [], []
    done = False
    while not done:
        a = np.random.randint(env.num_actions)
        obs, r, done, info = env.step(a)
        obs_list.append(obs)
        acts.append(a)
        rews.append(r)
        dones.append(done)
    return Episode(
        obs=np.stack(obs_list).astype(np.uint8),
        actions=np.asarray(acts, dtype=np.int64),
        rewards=np.asarray(rews, dtype=np.float32),
        dones=np.asarray(dones, dtype=bool),
    )
