"""Цикл обучения world-model (RSSM)."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .env import MiniGridPixelEnv
from .replay import ReplayBuffer, collect_random_episode
from .rssm import RSSM, RSSMConfig, compute_losses


def train_world_model(
    env_id: str = "MiniGrid-Empty-8x8-v0",
    num_episodes: int = 200,
    seq_len: int = 20,
    batch_size: int = 16,
    num_updates: int = 2000,
    lr: float = 3e-4,
    log_every: int = 50,
    save_path: str = "checkpoints/rssm.pt",
    seed: int = 0,
    device: str = "cpu",
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = MiniGridPixelEnv(env_id=env_id)
    buffer = ReplayBuffer(capacity=num_episodes + 10)

    print(f"[data] Сбор {num_episodes} случайных эпизодов...")
    for i in tqdm(range(num_episodes)):
        ep = collect_random_episode(env, seed=seed + i)
        buffer.add(ep)
    env.close()
    print(f"[data] Собрано. Переходов: {buffer.num_transitions}, "
          f"длиннейший эпизод: {max(ep.length for ep in buffer.episodes)}")

    # Уменьшаем seq_len, если он больше самого длинного эпизода
    max_ep_len = max(ep.length for ep in buffer.episodes)
    if seq_len > max_ep_len:
        seq_len = max(4, min(20, max_ep_len))
        print(f"[data] Сокращаю seq_len до {seq_len}")

    cfg = RSSMConfig(action_dim=env.num_actions if hasattr(env, "num_actions") else 3)
    model = RSSM(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    history = []
    t0 = time.time()
    for step in range(1, num_updates + 1):
        batch = buffer.sample(batch_size, seq_len, rng=rng)
        loss, info = compute_losses(model, batch, device)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        opt.step()
        if step % log_every == 0 or step == 1:
            info["step"] = step
            info["elapsed_s"] = round(time.time() - t0, 1)
            history.append(info)
            print(f"[train] step={step:5d} loss={info['loss']:.3f} "
                  f"recon={info['recon']:.3f} kl={info['kl']:.3f} rew={info['reward']:.3f} "
                  f"elapsed={info['elapsed_s']}s")

    torch.save({"model": model.state_dict(), "config": cfg.__dict__, "history": history}, save_path)
    print(f"[train] сохранено в {save_path}")
    return model, cfg, history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="checkpoints/rssm.pt")
    args = ap.parse_args()
    train_world_model(
        num_episodes=args.episodes, num_updates=args.updates,
        batch_size=args.batch, seq_len=args.seq, seed=args.seed, save_path=args.out,
    )
