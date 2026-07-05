"""Цикл эвалюации: запускает каждого агента на N эпизодов через M сидов,
фиксирует success rate, средний return, длину эпизода и (опционально) сохраняет GIF.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as imageio

from .env import MiniGridPixelEnv
from .rssm import RSSM, RSSMConfig
from .vlm_scorer import CLIPScorer
from .agents import RandomAgent, WMRewardAgent, WMVLMAgent


def _load_model(ckpt_path: str, device: str = "cpu") -> RSSM:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = RSSMConfig(**ckpt["config"])
    model = RSSM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def run_episode(env, agent, max_steps: int, record: bool = False):
    obs = env.reset()
    agent.reset(obs)
    frames = [obs] if record else None
    total_reward = 0.0
    success = False
    steps = 0
    for t in range(max_steps):
        a = agent.act(obs)
        obs, r, done, info = env.step(a)
        total_reward += r
        steps += 1
        if record:
            frames.append(obs)
        if info.get("success"):
            success = True
        if done:
            break
    return {"return": total_reward, "success": success, "steps": steps, "frames": frames}


def evaluate(
    ckpt_path: str,
    num_episodes: int = 10,
    seeds: list[int] | None = None,
    env_id: str = "MiniGrid-Empty-8x8-v0",
    horizon: int = 12,
    num_seq: int = 128,
    goal_prompt: str = "a red triangle agent standing on the green goal square",
    negative_prompt: str = "a red triangle agent far from the green goal square",
    max_steps: int = 60,
    gif_dir: str = "results/gifs",
    results_path: str = "results/metrics.json",
    device: str = "cpu",
    include_vlm: bool = True,
):
    seeds = seeds or [0, 1, 2]
    Path(gif_dir).mkdir(parents=True, exist_ok=True)
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[eval] загрузка модели {ckpt_path}")
    model = _load_model(ckpt_path, device=device)

    scorer = None
    if include_vlm:
        print("[eval] загрузка CLIP...")
        scorer = CLIPScorer(goal_prompt=goal_prompt, negative_prompt=negative_prompt, device=device)

    env = MiniGridPixelEnv(env_id=env_id)

    all_results = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        agents = {
            "random": RandomAgent(env.num_actions, rng),
            "wm_reward": WMRewardAgent(model, num_seq=num_seq, horizon=horizon, rng=rng),
        }
        if include_vlm:
            agents["wm_vlm"] = WMVLMAgent(model, scorer, num_seq=num_seq, horizon=horizon, rng=rng)

        for name, agent in agents.items():
            key = f"{name}_seed{seed}"
            print(f"[eval] запуск {key} на {num_episodes} эпизодов")
            ep_returns, ep_successes, ep_steps = [], [], []
            t0 = time.time()
            for ep in range(num_episodes):
                env._seed = seed * 1000 + ep  # меняем seed среды для каждого эпизода
                record = (ep == 0)  # первый эпизод сохраняем как GIF
                out = run_episode(env, agent, max_steps=max_steps, record=record)
                ep_returns.append(out["return"])
                ep_successes.append(out["success"])
                ep_steps.append(out["steps"])
                if record and out["frames"] is not None:
                    gif_path = os.path.join(gif_dir, f"{key}_ep0.gif")
                    imageio.mimsave(gif_path, out["frames"], duration=0.15)
            elapsed = time.time() - t0
            all_results[key] = {
                "agent": name,
                "seed": seed,
                "num_episodes": num_episodes,
                "success_rate": float(np.mean(ep_successes)),
                "mean_return": float(np.mean(ep_returns)),
                "mean_steps": float(np.mean(ep_steps)),
                "elapsed_s": round(elapsed, 1),
                "returns": ep_returns,
                "successes": [bool(s) for s in ep_successes],
            }
            print(f"  {key}: success_rate={all_results[key]['success_rate']:.2f} "
                  f"return={all_results[key]['mean_return']:.3f} steps={all_results[key]['mean_steps']:.1f} "
                  f"elapsed={elapsed:.1f}s")

    env.close()

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[eval] сохранено в {results_path}")

    # Агрегация по сидам
    agg = {}
    for name in ["random", "wm_reward", "wm_vlm"]:
        vals = [v for k, v in all_results.items() if v["agent"] == name]
        if not vals:
            continue
        agg[name] = {
            "success_rate_mean": float(np.mean([v["success_rate"] for v in vals])),
            "success_rate_std": float(np.std([v["success_rate"] for v in vals])),
            "mean_return": float(np.mean([v["mean_return"] for v in vals])),
            "num_seeds": len(vals),
            "num_episodes_per_seed": vals[0]["num_episodes"],
        }
    print("\n[eval] агрегат:")
    print(json.dumps(agg, indent=2))
    return all_results, agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/rssm.pt")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--num-seq", type=int, default=128)
    ap.add_argument("--no-vlm", action="store_true")
    args = ap.parse_args()
    evaluate(
        ckpt_path=args.ckpt, num_episodes=args.episodes, seeds=args.seeds,
        horizon=args.horizon, num_seq=args.num_seq, include_vlm=not args.no_vlm,
    )
