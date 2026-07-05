"""MPC-планировщик random shooting над обученной world-model RSSM.

Из текущего posterior-состояния сэмплим N последовательностей действий длины H,
воображаем траектории через RSSM, оцениваем каждый rollout переданной функцией
скора и возвращаем первое действие лучшей последовательности.

Функция скора `score_fn(imagined_frames_uint8, imagined_rewards)` полностью
подключаемая и возвращает скаляр на кандидата. Это позволяет подменять:
  * `reward_only`: сумма предсказанных наград от reward_head (baseline)
  * `vlm`: сумма/среднее CLIP-скоров по декодированным воображаемым кадрам
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .rssm import RSSM, _tensor_to_img


def _sample_actions(num_seq: int, horizon: int, num_actions: int, rng: np.random.Generator) -> torch.Tensor:
    return torch.as_tensor(rng.integers(0, num_actions, size=(num_seq, horizon)), dtype=torch.long)


def imagine_batch(model: RSSM, init_state: dict, actions: torch.Tensor):
    """actions: (N, H) long. Возвращает (N, H+1) декодированных кадров uint8 и (N, H) предсказанных наград.

    init_state содержит тензоры (h, z) формы (1, D); тайлим их до N.
    """
    N, H = actions.shape
    device = next(model.parameters()).device
    A = model.cfg.action_dim

    def tile(t):
        return t.expand(N, -1).contiguous()

    state = {"h": tile(init_state["h"]), "z": tile(init_state["z"])}
    acts_oh = F.one_hot(actions.to(device), num_classes=A).float()

    frames = []
    rewards = []
    # включаем начальную реконструкцию
    frames.append(_tensor_to_img(model.decoder(model.feat(state))))
    for t in range(H):
        state = model.img_step(state, acts_oh[:, t])
        # вспомогательные поля распределений внизу не нужны (остаются в словаре)
        rewards.append(model.reward_head(model.feat(state)).squeeze(-1))
        frames.append(_tensor_to_img(model.decoder(model.feat(state))))
    frames = torch.stack(frames, dim=1)   # (N, H+1, H, W, 3)
    rewards = torch.stack(rewards, dim=1) # (N, H)
    return frames, rewards


def random_shooting(
    model: RSSM,
    init_state: dict,
    num_seq: int,
    horizon: int,
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    rng: np.random.Generator,
) -> Tuple[int, dict]:
    """Возвращает лучшее первое действие и диагностику."""
    actions = _sample_actions(num_seq, horizon, model.cfg.action_dim, rng)
    with torch.inference_mode():
        frames, rewards = imagine_batch(model, init_state, actions)
        scores = score_fn(frames, rewards)  # (N,)
    best = int(torch.argmax(scores).item())
    return int(actions[best, 0].item()), {
        "best_score": float(scores[best].item()),
        "mean_score": float(scores.mean().item()),
        "actions_best": actions[best].tolist(),
    }


# ---------- функции скора ----------

def make_reward_only_scorer():
    def _fn(frames, rewards):
        # rewards: (N, H) — сумма по горизонту
        return rewards.sum(dim=1)
    return _fn


def make_vlm_scorer(clip_scorer, include_initial: bool = False, discount: float = 1.0):
    """Скор = дисконтированная сумма CLIP-скоров по кадрам rollout-а.

    Намеренно берём *будущие* воображаемые кадры (пропускаем шаг 0, если
    include_initial=False) — это удовлетворяет требованию задания «скорер применяется
    к будущим кадрам rollout-а».
    """
    def _fn(frames, rewards):
        # frames: (N, H+1, H, W, 3) uint8
        N, Tp1 = frames.shape[:2]
        start = 0 if include_initial else 1
        flat = frames[:, start:].reshape(-1, *frames.shape[2:])  # (N*T, H, W, 3)
        s = clip_scorer.score(flat)  # (N*T,)
        s = s.reshape(N, Tp1 - start)
        if discount != 1.0:
            gammas = torch.tensor([discount ** t for t in range(s.shape[1])], device=s.device)
            s = s * gammas
        return s.sum(dim=1)
    return _fn
