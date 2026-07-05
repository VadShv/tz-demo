"""Agents used for evaluation."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .rssm import RSSM
from .planner import random_shooting, make_reward_only_scorer, make_vlm_scorer


class RandomAgent:
    name = "random"

    def __init__(self, num_actions: int, rng: np.random.Generator):
        self.num_actions = num_actions
        self.rng = rng

    def reset(self, obs):
        pass

    def act(self, obs) -> int:
        return int(self.rng.integers(0, self.num_actions))


class WorldModelPlanner:
    """Common base: maintains posterior state from real observations, plans in imagination."""

    def __init__(self, model: RSSM, score_fn, num_seq: int, horizon: int, rng: np.random.Generator):
        self.model = model
        self.model.eval()
        self.score_fn = score_fn
        self.num_seq = num_seq
        self.horizon = horizon
        self.rng = rng
        self.device = next(model.parameters()).device
        self._state = None

    def reset(self, obs):
        # obs: (H, W, 3) uint8
        self._state = self.model.init_state(1, self.device)
        obs_t = torch.as_tensor(obs, device=self.device).unsqueeze(0)  # (1, H, W, 3)
        embed = self.model.encoder(obs_t)  # (1, embed)
        zero_act = torch.zeros(1, self.model.cfg.action_dim, device=self.device)
        with torch.inference_mode():
            self._state = self.model.obs_step(self._state, zero_act, embed)
        self._last_action = 0

    def act(self, obs) -> int:
        # Advance posterior with the last observation given the previous action
        obs_t = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        embed = self.model.encoder(obs_t)
        act_oh = F.one_hot(torch.tensor([self._last_action], device=self.device),
                           num_classes=self.model.cfg.action_dim).float()
        with torch.inference_mode():
            self._state = self.model.obs_step(self._state, act_oh, embed)
        best_action, _ = random_shooting(
            self.model,
            {"h": self._state["h"], "z": self._state["z"]},
            self.num_seq, self.horizon, self.score_fn, self.rng,
        )
        self._last_action = best_action
        return best_action


class WMRewardAgent(WorldModelPlanner):
    name = "wm_reward"

    def __init__(self, model, num_seq, horizon, rng):
        super().__init__(model, make_reward_only_scorer(), num_seq, horizon, rng)


class WMVLMAgent(WorldModelPlanner):
    name = "wm_vlm"

    def __init__(self, model, clip_scorer, num_seq, horizon, rng, discount: float = 0.9):
        super().__init__(model, make_vlm_scorer(clip_scorer, discount=discount),
                         num_seq, horizon, rng)
