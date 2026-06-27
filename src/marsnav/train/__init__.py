"""Training entry points (SB3 PPO in v1)."""

from marsnav.train.ppo_flat import PPOFlatConfig, train_ppo_flat

__all__ = ["PPOFlatConfig", "train_ppo_flat"]
