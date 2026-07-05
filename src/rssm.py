"""Compact RSSM (Recurrent State-Space Model) world model in the style of PlaNet/Dreamer.

Design choices (kept small for CPU training):
  * Encoder: 4-layer CNN 64x64 -> 256-d embed
  * Deterministic recurrent state h_t: GRU with hidden size 128
  * Stochastic latent z_t: diagonal Gaussian, dim=16, produced by
    prior p(z_t | h_t) and posterior q(z_t | h_t, e_t)
  * Decoder: mirrored transposed CNN -> 64x64x3 reconstruction
  * Reward head: 2-layer MLP on (h_t, z_t) -> scalar

Losses:
  * Recon: MSE on normalized [-1, 1] images
  * KL: KL(posterior || prior) with free bits
  * Reward: MSE
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class RSSMConfig:
    action_dim: int = 3
    embed_dim: int = 256
    deter_dim: int = 128
    stoch_dim: int = 16
    hidden_dim: int = 128
    img_size: int = 64
    kl_free_bits: float = 1.0
    kl_scale: float = 1.0


def _img_to_tensor(x: torch.Tensor) -> torch.Tensor:
    """uint8 (..., H, W, 3) -> float32 (..., 3, H, W) in [-1, 1]."""
    if x.dtype == torch.uint8:
        x = x.float() / 127.5 - 1.0
    if x.shape[-1] == 3:
        x = x.movedim(-1, -3)
    return x


def _tensor_to_img(x: torch.Tensor) -> torch.Tensor:
    """float (..., 3, H, W) in [-1, 1] -> uint8 (..., H, W, 3)."""
    x = (x.clamp(-1, 1) + 1) * 127.5
    x = x.movedim(-3, -1).to(torch.uint8)
    return x


class ConvEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(inplace=True),   # 31
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(inplace=True),  # 14
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(inplace=True), # 6
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU(inplace=True),# 2
        )
        self.fc = nn.Linear(256 * 2 * 2, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., H, W, 3) uint8 or float
        orig_shape = x.shape[:-3]
        x = _img_to_tensor(x)
        x = x.reshape(-1, 3, x.shape[-2], x.shape[-1])
        h = self.net(x)
        h = h.flatten(1)
        h = self.fc(h)
        return h.reshape(*orig_shape, -1)


class ConvDecoder(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        # start from a 1x1 spatial feature and upsample to 64x64
        self.fc = nn.Linear(feat_dim, 256)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride=2),  # 1 -> 5
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 5, stride=2),   # 5 -> 13
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 6, stride=2),    # 13 -> 30
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 6, stride=2),     # 30 -> 64
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        orig_shape = feat.shape[:-1]
        h = self.fc(feat.reshape(-1, feat.shape[-1]))
        h = h.reshape(-1, 256, 1, 1)
        img = self.net(h)  # (-1, 3, 64, 64)
        return img.reshape(*orig_shape, 3, img.shape[-2], img.shape[-1])


class RSSM(nn.Module):
    def __init__(self, cfg: RSSMConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ConvEncoder(cfg.embed_dim)

        feat_dim = cfg.deter_dim + cfg.stoch_dim

        # Recurrent core
        self.act_stoch_to_hidden = nn.Sequential(
            nn.Linear(cfg.stoch_dim + cfg.action_dim, cfg.hidden_dim), nn.ELU(inplace=True),
        )
        self.gru = nn.GRUCell(cfg.hidden_dim, cfg.deter_dim)

        # Prior p(z_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(cfg.deter_dim, cfg.hidden_dim), nn.ELU(inplace=True),
            nn.Linear(cfg.hidden_dim, 2 * cfg.stoch_dim),
        )
        # Posterior q(z_t | h_t, e_t)
        self.post_net = nn.Sequential(
            nn.Linear(cfg.deter_dim + cfg.embed_dim, cfg.hidden_dim), nn.ELU(inplace=True),
            nn.Linear(cfg.hidden_dim, 2 * cfg.stoch_dim),
        )

        self.decoder = ConvDecoder(feat_dim)
        self.reward_head = nn.Sequential(
            nn.Linear(feat_dim, cfg.hidden_dim), nn.ELU(inplace=True),
            nn.Linear(cfg.hidden_dim, 1),
        )

    # ---------- state utilities ----------
    def init_state(self, batch_size: int, device):
        return {
            "h": torch.zeros(batch_size, self.cfg.deter_dim, device=device),
            "z": torch.zeros(batch_size, self.cfg.stoch_dim, device=device),
        }

    def _split_dist(self, params):
        mean, std = torch.chunk(params, 2, dim=-1)
        std = F.softplus(std) + 0.1
        return mean, std

    def _sample(self, mean, std):
        if self.training:
            return mean + std * torch.randn_like(std)
        return mean

    # ---------- one-step transitions ----------
    def img_step(self, prev_state, action_onehot):
        """Prior step: given (h_{t-1}, z_{t-1}, a_{t-1}) -> h_t, prior z_t."""
        x = torch.cat([prev_state["z"], action_onehot], dim=-1)
        x = self.act_stoch_to_hidden(x)
        h = self.gru(x, prev_state["h"])
        prior_params = self.prior_net(h)
        prior_mean, prior_std = self._split_dist(prior_params)
        z = self._sample(prior_mean, prior_std)
        return {"h": h, "z": z, "prior_mean": prior_mean, "prior_std": prior_std}

    def obs_step(self, prev_state, action_onehot, embed):
        """Posterior step: also incorporates observation embedding."""
        prior = self.img_step(prev_state, action_onehot)
        post_input = torch.cat([prior["h"], embed], dim=-1)
        post_params = self.post_net(post_input)
        post_mean, post_std = self._split_dist(post_params)
        z = self._sample(post_mean, post_std)
        return {
            "h": prior["h"], "z": z,
            "prior_mean": prior["prior_mean"], "prior_std": prior["prior_std"],
            "post_mean": post_mean, "post_std": post_std,
        }

    def feat(self, state):
        return torch.cat([state["h"], state["z"]], dim=-1)

    def decode(self, state):
        return self.decoder(self.feat(state))

    def predict_reward(self, state):
        return self.reward_head(self.feat(state)).squeeze(-1)

    # ---------- rollout in imagination ----------
    def imagine(self, init_state, action_seq_onehot):
        """action_seq_onehot: (B, H, A). Returns list of states length H+1 incl init."""
        states = [init_state]
        s = init_state
        for t in range(action_seq_onehot.shape[1]):
            s = self.img_step(s, action_seq_onehot[:, t])
            states.append(s)
        return states

    # ---------- training pass ----------
    def observe(self, obs_seq, actions_onehot):
        """obs_seq: (B, T+1, H, W, 3) uint8; actions_onehot: (B, T, A).

        Returns list of posterior states length T+1 (state[0] uses only obs[0] via a
        zero-init prior; from t=1 onward transitions use actions).
        """
        B, Tp1 = obs_seq.shape[:2]
        T = Tp1 - 1
        device = actions_onehot.device
        embeds = self.encoder(obs_seq)  # (B, T+1, embed)

        # Bootstrap: use a zero action to run one posterior step at t=0
        prev = self.init_state(B, device)
        zero_act = torch.zeros(B, self.cfg.action_dim, device=device)
        s0 = self.obs_step(prev, zero_act, embeds[:, 0])
        states = [s0]
        for t in range(T):
            s = self.obs_step(states[-1], actions_onehot[:, t], embeds[:, t + 1])
            states.append(s)
        return states, embeds


def kl_divergence(mean_q, std_q, mean_p, std_p):
    var_q = std_q ** 2
    var_p = std_p ** 2
    kl = torch.log(std_p / std_q) + (var_q + (mean_q - mean_p) ** 2) / (2 * var_p) - 0.5
    return kl.sum(dim=-1)


def compute_losses(model: RSSM, batch, device):
    obs = torch.as_tensor(batch["obs"], device=device)  # uint8 (B, T+1, H, W, 3)
    acts = torch.as_tensor(batch["actions"], device=device, dtype=torch.long)  # (B, T)
    rews = torch.as_tensor(batch["rewards"], device=device, dtype=torch.float32)  # (B, T)

    A = model.cfg.action_dim
    acts_oh = F.one_hot(acts, num_classes=A).float()

    states, _ = model.observe(obs, acts_oh)  # length T+1
    # Stack fields
    def stk(key):
        return torch.stack([s[key] for s in states], dim=1)  # (B, T+1, D)
    feat = torch.cat([stk("h"), stk("z")], dim=-1)
    recon = model.decoder(feat)  # (B, T+1, 3, H, W)
    target = _img_to_tensor(obs)  # (B, T+1, 3, H, W)
    recon_loss = F.mse_loss(recon, target)

    # KL only on steps t>=1 (state[0] has no meaningful prior)
    post_mean = torch.stack([s["post_mean"] for s in states[1:]], dim=1)
    post_std = torch.stack([s["post_std"] for s in states[1:]], dim=1)
    prior_mean = torch.stack([s["prior_mean"] for s in states[1:]], dim=1)
    prior_std = torch.stack([s["prior_std"] for s in states[1:]], dim=1)
    kl = kl_divergence(post_mean, post_std, prior_mean, prior_std)  # (B, T)
    kl_loss = torch.clamp(kl, min=model.cfg.kl_free_bits).mean()

    # Reward on t>=1
    feat_rew = feat[:, 1:]
    pred_rew = model.reward_head(feat_rew).squeeze(-1)
    reward_loss = F.mse_loss(pred_rew, rews)

    total = recon_loss + model.cfg.kl_scale * kl_loss + reward_loss
    return total, {
        "loss": total.item(),
        "recon": recon_loss.item(),
        "kl": kl_loss.item(),
        "reward": reward_loss.item(),
    }
