from functools import partial
from typing import Tuple

import rosnav_rl
import rosnav_rl.model.dreamerv3 as dreamerv3
from rosnav_rl import SupportedRLFrameworks

import rl_utils.cfg as arena_cfg
from rl_utils.tools.config import load_training_config

# from rl_utils.tools.constants import SIMULATION_NAMESPACES
from rl_utils.tools.env_utils import make_envs
from rl_utils.tools.model_utils import setup_wandb
from rl_utils.trainer.arena_trainer import ArenaTrainer


class DreamerV3Trainer(ArenaTrainer):
    """DreamerV3Trainer class for training agents using DreamerV3 framework.ks.DREAMER_V3

    This class implements the ArenaTrainer interface for the DreamerV3 reinforcement learning
    framework. It handles the setup of the agent, environment configuration, and training process
    specific to DreamerV3.

    Attributes:
        __framework (SupportedRLFrameworks): The RL framework identifier, set to DREAMER_V3.
        environment (Tuple[dreamerv3.Parallel, dreamerv3.Parallel]): Tuple containing training
            and evaluation environments.
        config (arena_cfg.TrainingCfg): Configuration for the trainer.

    Note:
        This class requires a configuration of type ArenaDreamerV3Cfg.
    """

    __framework = SupportedRLFrameworks.DREAMER_V3
    _config_type = arena_cfg.ArenaBaseCfg
    environment: Tuple[dreamerv3.Parallel, dreamerv3.Parallel]

    def __init__(self, config: arena_cfg.TrainingCfg):
        super().__init__(config, config.resume)

    def _setup_monitoring(self, *args, **kwargs):
        if (
            self.config.arena_cfg.monitoring.wandb
            and not self.config.arena_cfg.general.debug_mode
        ):
            setup_wandb(
                run_name=self.config.agent_cfg.name,
                group=self.config.arena_cfg.monitoring.wandb.group,
                config=self.config,
                to_watch=[self.agent.model.model],
                agent_id=self.config.agent_cfg.name,
            )

    def _setup_agent(self, *args, **kwargs):
        """
        Set up the reinforcement learning agent for training.

        This method initializes the agent using the RoboNav RL agent class with the configured parameters.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None

        Note:
            The agent is created with the agent configuration from the config object and
            is linked to the agent_state_container to maintain state during training.
        """
        self.agent = rosnav_rl.RL_Agent(
            agent_cfg=self.config.agent_cfg,
            agent_state_container=self.agent_state_container,
        )
        self.agent.initialize_model()

    def _setup_environment(self, *args, **kwargs):
        """
        Set up the training and evaluation environments for DreamerV3.

        This method initializes environments using the configuration parameters from the agent.
        It creates training environments with specified wrappers and configurations and sets up
        evaluation environments (currently identical to training environments).

        The environment setup supports a debug mode where environments are created as dummy environments
        instead of parallel environments.

        Wrappers applied to environments:
        - WoTruncatedFlag: Wrapper that removes truncated flag from environment
        - TimeLimit: Limits the duration of episodes
        - SelectAction: Selects the 'action' key from the action dictionary
        - UUID: Adds a unique identifier to each environment
        - ResetWoInfo: Resets environment without requiring info
        - ChannelFirsttoLast: Converts observations from channel-first to channel-last format

        Args:
            *args: Variable length argument list passed to the environment creation.
            **kwargs: Arbitrary keyword arguments passed to the environment creation.

        Returns:
            None: Sets self.environment as a tuple of (train_envs, eval_envs).
        """
        train_env_fncs = make_envs(
            rl_agent=self.agent,
            n_envs=self.config.arena_cfg.general.n_envs,
            max_steps=self.config.arena_cfg.general.max_num_moves_per_eps,
            init_env_by_call=False,
            namespace_fn="/task_generator_node/jackal",
            simulation_state_container=self.simulation_state_container,
            wrappers=[
                dreamerv3.WoTruncatedFlag,
                partial(
                    dreamerv3.TimeLimit,
                    duration=self.config.arena_cfg.general.max_num_moves_per_eps,
                ),
                partial(dreamerv3.SelectAction, key="action"),
                dreamerv3.UUID,
                dreamerv3.ResetWoInfo,
                dreamerv3.ChannelFirsttoLast,
            ],
        )

        if self.config.arena_cfg.general.debug_mode:
            train_envs = [dreamerv3.Damy(init_fnc()) for init_fnc in train_env_fncs]
        else:
            train_envs = [
                dreamerv3.Parallel(lambda: init_fnc(), "daemon")
                for init_fnc in train_env_fncs
            ]
        eval_envs = train_envs
        self.environment = (train_envs, eval_envs)

    def _train_impl(self, *args, **kwargs):
        """
        Implementation of the training process for DreamerV3 agent.

        This method follows proper OOP design by calling self.agent.model.train()
        and passes the StagedCfg model directly for curriculum learning.

        Parameters:
            *args: Variable length argument list (unused, for compatibility).
            **kwargs: Arbitrary keyword arguments (unused, for compatibility).
        """
        # Extract curriculum configuration from arena_cfg
        curriculum_config = None
        if (hasattr(self.config, 'arena_cfg') and 
            hasattr(self.config.arena_cfg, 'task') and 
            hasattr(self.config.arena_cfg.task, 'staged') and
            self.config.arena_cfg.task.tm_modules == "staged"):
            curriculum_config = self.config.arena_cfg.task.staged

        verbose = getattr(self.config.arena_cfg.general, "verbose", 0)

        # Call model's train method directly - proper OOP design
        # Pass the StagedCfg model directly instead of converting to dict
        self.agent.model.train(
            train_envs=self.environment[0],
            eval_envs=self.environment[1],
            curriculum_config=curriculum_config,
            node=self.node,
            verbose=verbose,
        )


if __name__ == "__main__":
    config = load_training_config("dreamer_training_config.yaml")

    trainer = DreamerV3Trainer(config)
    trainer.train()
