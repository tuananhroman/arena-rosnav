import re
from typing import List, TYPE_CHECKING

import torch

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.utils import configure_logger, constant_fn
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

import wandb

if TYPE_CHECKING:
    from rl_utils.cfg import TrainingCfg


def setup_wandb(
    run_name: str = None,
    group: str = None,
    config: "TrainingCfg" = None,
    agent_id: str = None,
    to_watch: List[torch.nn.Module] = [],
) -> None:
    """
    Set up Weights and Biases (wandb) for tracking and visualizing training.

    This function logs into wandb, initializes a new run with the given training
    configuration, and sets up monitoring for TensorBoard and Gym environments.
    It also watches the policy model for logging gradients and parameters.

    Args:
        train_cfg (TrainingCfg): The training configuration object containing
            settings and parameters for the training session.
        rl_model (RL_Model): The reinforcement learning model to be tracked
            and monitored by wandb.

    Returns:
        None
    """
    wandb.login()
    wandb.init(
        name=run_name if run_name else config.arena_cfg.monitoring.wandb.run_name,
        group=group if group else config.arena_cfg.monitoring.wandb.group,
        project=config.arena_cfg.monitoring.wandb.project_name,
        tags=config.arena_cfg.monitoring.wandb.tags,
        entity=None,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=False,
        config=config.model_dump(),
        id=agent_id,
    )
    for module in to_watch:
        wandb.watch(module, log_graph=True)
