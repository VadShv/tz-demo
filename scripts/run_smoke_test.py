"""End-to-end smoke-тест: обучает крошечную RSSM и запускает мини-эвалюацию.

Проверяет работоспособность пайплайна на CPU за ~2–3 минуты без CLIP и
~5–10 минут с CLIP.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.train_wm import train_world_model
from src.evaluate import evaluate


def main(with_vlm: bool = True):
    Path("checkpoints").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    print("=" * 60)
    print("SMOKE-ТЕСТ: обучение маленькой RSSM")
    print("=" * 60)
    train_world_model(
        num_episodes=30,
        num_updates=200,
        batch_size=8,
        seq_len=10,
        seed=0,
        save_path="checkpoints/rssm_smoke.pt",
    )

    print("\n" + "=" * 60)
    print("SMOKE-ТЕСТ: эвалюация агентов")
    print("=" * 60)
    evaluate(
        ckpt_path="checkpoints/rssm_smoke.pt",
        num_episodes=2,
        seeds=[0],
        horizon=5,
        num_seq=8,
        include_vlm=with_vlm,
    )
    print("\nSmoke-тест завершён. См. results/metrics.json и results/gifs/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-vlm", action="store_true")
    args = ap.parse_args()
    main(with_vlm=not args.no_vlm)
