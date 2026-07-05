"""Обёртка над средой MiniGrid, выдающая 64×64 RGB-наблюдения.

Используется MiniGrid-Empty-8x8-v0 с пиксельными наблюдениями. Пространство
действий сокращено до {0: left, 1: right, 2: forward} — только эти действия
имеют смысл для Empty.
"""
from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401  регистрирует среды
import numpy as np
from PIL import Image


IMG_SIZE = 64
ACTION_NAMES = ["left", "right", "forward"]
NUM_ACTIONS = len(ACTION_NAMES)


class MiniGridPixelEnv:
    """Тонкая обёртка над MiniGrid; возвращает uint8 RGB-кадры формы (H, W, 3)."""

    def __init__(self, env_id: str = "MiniGrid-Empty-8x8-v0", img_size: int = IMG_SIZE, seed: int | None = None):
        self.env_id = env_id
        self.img_size = img_size
        self._env = gym.make(env_id, render_mode="rgb_array")
        self._seed = seed
        self.num_actions = NUM_ACTIONS
        self._last_obs = None
        self._steps = 0
        self.max_steps = self._env.spec.max_episode_steps or 100

    def _render(self) -> np.ndarray:
        img = self._env.render()  # (H, W, 3) uint8
        if img.shape[0] != self.img_size:
            img = np.asarray(Image.fromarray(img).resize((self.img_size, self.img_size), Image.BILINEAR))
        return img

    def reset(self, seed: int | None = None) -> np.ndarray:
        s = seed if seed is not None else self._seed
        self._env.reset(seed=s)
        self._steps = 0
        obs = self._render()
        self._last_obs = obs
        return obs

    def step(self, action: int):
        assert 0 <= action < NUM_ACTIONS, f"недопустимое действие {action}"
        obs_dict, reward, terminated, truncated, info = self._env.step(int(action))
        self._steps += 1
        obs = self._render()
        self._last_obs = obs
        done = bool(terminated or truncated)
        # MiniGrid выдаёт разреженную положительную награду при достижении цели;
        # дополнительно прокидываем флаг success для метрик.
        info = dict(info)
        info["success"] = bool(reward > 0)
        return obs, float(reward), done, info

    def close(self):
        self._env.close()

    @property
    def last_obs(self) -> np.ndarray:
        return self._last_obs
