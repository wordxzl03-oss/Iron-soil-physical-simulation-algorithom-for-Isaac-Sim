"""Base algorithms and serialization helpers."""

from __future__ import annotations

import torch
from stable_baselines3 import SAC

from .config import SACConfig


def build_sac(env, config: SACConfig) -> SAC:
    """Build the reference continuous-control SAC policy."""
    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return SAC(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        buffer_size=max(config.buffer_size, config.timesteps),
        learning_starts=min(config.learning_starts, max(100, config.timesteps // 10)),
        batch_size=config.batch_size,
        gamma=config.gamma,
        tau=config.tau,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs={"net_arch": list(config.hidden_sizes)},
        seed=config.seed,
        device=device,
        verbose=0,
    )


def load_sac(path, env=None, device: str = "auto") -> SAC:
    return SAC.load(path, env=env, device=device)
