"""Random-shooting MPC planner over a learned RSSM world model.

Given the current posterior state, sample N action sequences of length H,
imagine trajectories with the RSSM, score each rollout with a provided
scoring function, and return the first action of the best sequence.

Scoring is fully pluggable via `score_fn(imagined_frames_uint8, imagined_rewards)`
returning a scalar per candidate. This lets us swap in:
  * `reward_only`: sum of predicted rewards from the reward head (baseline)
  * `vlm`: sum/mean of CLIP scores over decoded imagined frames
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
    """actions: (N, H) long. Returns (N, H+1) decoded frames uint8 and (N, H) predicted rewards.

    init_state contains (h, z) tensors of shape (1, D); we tile to N.
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
    # include initial reconstruction
    frames.append(_tensor_to_img(model.decoder(model.feat(state))))
    for t in range(H):
        state = model.img_step(state, acts_oh[:, t])
        # remove aux dist entries not needed downstream (kept in dict is fine)
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
    """Return best first action and diagnostics."""
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


# ---------- scoring functions ----------

def make_reward_only_scorer():
    def _fn(frames, rewards):
        # rewards: (N, H) — sum over horizon
        return rewards.sum(dim=1)
    return _fn


def make_vlm_scorer(clip_scorer, include_initial: bool = False, discount: float = 1.0):
    """Score = discounted sum of CLIP-scores across the rollout frames.

    We deliberately include the *future* imagined frames (skip step 0 unless
    include_initial=True) — this satisfies the "scoring must apply to future
    rollout frames" requirement.
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
