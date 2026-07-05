"""CLIP-based VLM scorer for imagined future frames.

Given a batch of decoded frames from a world-model rollout, returns a
scalar "goal-progress" score per frame — the cosine similarity between
the CLIP image embedding and a text embedding of the goal prompt.

Design:
  * We use open_clip's ViT-B/32 OpenAI checkpoint (small, CPU-friendly).
  * A pair of contrasting prompts is supported ("goal" vs "not goal");
    the returned score is softmax(sim_goal - sim_not_goal). This is more
    discriminative than raw similarity for MiniGrid-style renders.
  * All CLIP calls run under torch.inference_mode() and the model is
    frozen (eval).
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

        # CLIP normalisation stats
        self._mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    def _prepare(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs: (N, 3, H, W) in [-1, 1] float or (N, H, W, 3) uint8/float.
        Returns CLIP-normalized 224x224 tensor.
        """
        if imgs.dtype == torch.uint8:
            imgs = imgs.float() / 255.0
            if imgs.shape[-1] == 3:
                imgs = imgs.movedim(-1, -3)
        else:
            if imgs.shape[-1] == 3 and imgs.dim() == 4:
                imgs = imgs.movedim(-1, -3)
            # From RSSM decoder we expect [-1, 1]
            if imgs.min() < 0:
                imgs = (imgs.clamp(-1, 1) + 1) / 2
        imgs = F.interpolate(imgs, size=224, mode="bilinear", align_corners=False)
        imgs = (imgs - self._mean) / self._std
        return imgs

    @torch.inference_mode()
    def score(self, imgs: torch.Tensor) -> torch.Tensor:
        """Return a scalar score per image. Higher = closer to goal."""
        x = self._prepare(imgs.to(self.device))
        feats = self.model.encode_image(x)
        feats = F.normalize(feats, dim=-1)
        sims = feats @ self.text_features.T  # (N, P)
        if sims.shape[-1] == 1:
            return sims.squeeze(-1)
        # Contrastive: prefer larger goal-sim relative to negative
        logits = 100.0 * sims  # CLIP standard scaling
        return logits[:, 0] - logits[:, 1]

    @torch.inference_mode()
    def score_numpy(self, imgs: np.ndarray) -> np.ndarray:
        t = torch.as_tensor(imgs)
        return self.score(t).cpu().numpy()
