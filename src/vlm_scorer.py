"""VLM-скорер на базе CLIP для воображаемых будущих кадров.

По пачке декодированных кадров rollout-а world-model возвращает скалярный
скор «goal-progress» для каждого кадра — косинусное сходство между визуальным
embedding кадра и текстовым embedding целевого промпта.

Проектные решения:
  * Используется ViT-B/32 OpenAI через open_clip (компактная, пригодная для CPU).
  * Поддерживается пара контрастных промптов («goal» vs «not goal»);
    возвращаемый скор = 100·(sim_goal − sim_not_goal). На рендерах MiniGrid
    такой сигнал более дискриминативен, чем голое сходство.
  * Все вызовы CLIP выполняются под torch.inference_mode(); модель заморожена (eval).

Примечание. Промпты оставлены на английском: OpenAI-чекпойнт CLIP обучен почти
исключительно на англоязычных подписях — русские промпты дают почти нулевой сигнал.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import open_clip


class CLIPScorer:
    def __init__(
        self,
        goal_prompt: str,
        negative_prompt: str | None = None,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cpu",
    ):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(model_name)

        prompts = [goal_prompt]
        if negative_prompt is not None:
            prompts.append(negative_prompt)
        self.prompts = prompts
        with torch.inference_mode():
            tok = self.tokenizer(prompts).to(device)
            text_features = self.model.encode_text(tok)
            text_features = F.normalize(text_features, dim=-1)
        self.text_features = text_features  # (P, D)

        # Статистики нормализации CLIP
        self._mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    def _prepare(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs: (N, 3, H, W) в [-1, 1] float или (N, H, W, 3) uint8/float.
        Возвращает CLIP-нормализованный тензор 224x224.
        """
        if imgs.dtype == torch.uint8:
            imgs = imgs.float() / 255.0
            if imgs.shape[-1] == 3:
                imgs = imgs.movedim(-1, -3)
        else:
            if imgs.shape[-1] == 3 and imgs.dim() == 4:
                imgs = imgs.movedim(-1, -3)
            # От декодера RSSM ожидаем диапазон [-1, 1]
            if imgs.min() < 0:
                imgs = (imgs.clamp(-1, 1) + 1) / 2
        imgs = F.interpolate(imgs, size=224, mode="bilinear", align_corners=False)
        imgs = (imgs - self._mean) / self._std
        return imgs

    @torch.inference_mode()
    def score(self, imgs: torch.Tensor) -> torch.Tensor:
        """Скалярный скор для каждого изображения. Больше = ближе к цели."""
        x = self._prepare(imgs.to(self.device))
        feats = self.model.encode_image(x)
        feats = F.normalize(feats, dim=-1)
        sims = feats @ self.text_features.T  # (N, P)
        if sims.shape[-1] == 1:
            return sims.squeeze(-1)
        # Контрастный вариант: поощряем большее sim для цели относительно негативного промпта
        logits = 100.0 * sims  # стандартное масштабирование CLIP
        return logits[:, 0] - logits[:, 1]

    @torch.inference_mode()
    def score_numpy(self, imgs: np.ndarray) -> np.ndarray:
        t = torch.as_tensor(imgs)
        return self.score(t).cpu().numpy()
