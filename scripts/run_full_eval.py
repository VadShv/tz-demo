"""Full training + evaluation run recommended for a GPU (Colab T4)."""
from __future__ import annotations

from src.train_wm import train_world_model
from src.evaluate import evaluate


def main():
    train_world_model(
        num_episodes=300,
        num_updates=10000,
        batch_size=32,
        seq_len=20,
        seed=0,
        save_path="checkpoints/rssm_full.pt",
    )

    evaluate(
        ckpt_path="checkpoints/rssm_full.pt",
        num_episodes=20,
        seeds=[0, 1, 2],
        horizon=12,
        num_seq=128,
        include_vlm=True,
    )


if __name__ == "__main__":
    main()
