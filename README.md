# World Model + VLM Scorer — Demo

A minimal demo that combines a Dreamer-style **RSSM world model** with a
**VLM-based scorer (CLIP)** for planning in [MiniGrid](https://minigrid.farama.org/).

```
                            ┌──────────────────┐
    real obs  ──encoder──▶  │   RSSM (world    │  ──imagined rollouts──▶  decoded frames
                            │   model, GRU +   │
                            │   stoch latent)  │  ──reward head──▶       predicted returns
                            └──────────────────┘                             │
                                                                             ▼
                                   ┌──────────────────────────┐    ┌─────────────────┐
                                   │  Random-shooting planner │◀──│ CLIP text-image │
                                   │  picks best first action  │    │ scorer  (VLM)  │
                                   └──────────────────────────┘    └─────────────────┘
```

## What's inside

| Component | File | Notes |
|---|---|---|
| MiniGrid pixel env (64×64 RGB) | `src/env.py` | `MiniGrid-Empty-8x8-v0`, 3 actions |
| Replay buffer for episodes | `src/replay.py` | sequences for RSSM training |
| RSSM world model (~2.3M params) | `src/rssm.py` | CNN encoder/decoder, GRU, stochastic latent |
| CLIP scorer over decoded frames | `src/vlm_scorer.py` | ViT-B/32 OpenAI, contrastive prompts |
| Random-shooting MPC planner | `src/planner.py` | pluggable scoring |
| Agents (Random, WM+reward, WM+VLM) | `src/agents.py` | |
| Training script | `src/train_wm.py` | |
| Eval script with metrics + GIFs | `src/evaluate.py` | |

## Install

```bash
pip install -r requirements.txt
```

## Usage

**1. Train the world model** (collects random episodes, then optimizes RSSM):

```bash
python -m src.train_wm --episodes 200 --updates 2000 --batch 16 --seq 20 \
                      --out checkpoints/rssm.pt
```

Recommended settings for a GPU (Colab T4): `--episodes 500 --updates 10000`.

**2. Evaluate all three agents**:

```bash
python -m src.evaluate --ckpt checkpoints/rssm.pt \
                      --episodes 20 --seeds 0 1 2 \
                      --horizon 12 --num-seq 128
```

Produces `results/metrics.json` and per-agent GIFs in `results/gifs/`.

## Method

**World model (RSSM).** Following the recurrent state-space model of
PlaNet ([Hafner et al. 2018](https://arxiv.org/abs/1811.04551)) and Dreamer
([Hafner et al. 2020](https://arxiv.org/abs/1912.01603) /
[Hafner et al. 2023](https://arxiv.org/abs/2301.04104)), the state
`s_t = (h_t, z_t)` combines a deterministic GRU hidden `h_t` and a
stochastic latent `z_t`. Training minimizes image reconstruction MSE +
KL(posterior‖prior) with free bits + reward MSE.

**Planner.** At each real step, we run *random shooting*: sample `N` action
sequences of length `H`, imagine each rollout in the world model, score
it, and take the first action of the best sequence. All CPU-friendly,
easily replaced by CEM.

**VLM scorer.** We use CLIP ViT-B/32 with contrastive prompts:

- positive: `"a red triangle agent standing on the green goal square"`
- negative: `"a red triangle agent far from the green goal square"`

The score is `100 · (cos_sim(pos) − cos_sim(neg))` averaged
(discounted) across the *future* decoded frames of the rollout. This
satisfies the requirement that scoring is applied to imagined future
observations, not just the current one.

**Baselines.**

1. `random` — uniform random policy
2. `wm_reward` — same MPC planner, but scores rollouts with the RSSM's
   own predicted-reward head instead of the VLM

## Results

See `report/report.pdf` for the full write-up, and
`results/metrics.json` for raw numbers.

## Repository layout

```
wm-vlm-demo/
├── src/                  # library
├── scripts/              # convenience scripts
├── configs/default.yaml  # hyperparameters
├── checkpoints/          # trained RSSM checkpoints
├── results/              # metrics + GIFs
├── report/report.pdf     # short PDF report
└── notebooks/demo.ipynb  # Colab-friendly notebook
```

## Notes on running on CPU vs GPU

The included smoke-test config (`scripts/run_smoke_test.py`) trains for
~200 updates on 30 random episodes — enough to verify the pipeline
end-to-end in ~1 minute on CPU, but *not* enough to get meaningful
policy performance. For real evaluation numbers, run the full config on
a GPU (Colab T4 free tier is sufficient).

## References

- Hafner et al., **PlaNet** — Learning Latent Dynamics for Planning from Pixels. [arXiv:1811.04551](https://arxiv.org/abs/1811.04551)
- Hafner et al., **Dreamer-V3** — Mastering Diverse Domains through World Models. [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
- Radford et al., **CLIP** — Learning Transferable Visual Models. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- [MiniGrid](https://minigrid.farama.org/) — Chevalier-Boisvert et al., 2023
