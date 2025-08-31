from dataclasses import dataclass
from functools import partial

import rclpy
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

import rl_utils.cfg as arena_cfg
import rl_utils.utils.paths as Paths
from rl_utils.envs.wrappers import TimeSyncWrapper
from rl_utils.stable_baselines3.eval_callbacks.initialization import init_sb3_callbacks
from rl_utils.tools.config import load_training_config
from rl_utils.tools.env_utils import make_envs, sb3_wrap_env
from rl_utils.tools.model_utils import setup_wandb
from rl_utils.trainer.arena_trainer import (
    ArenaTrainer,
    SupportedRLFrameworks,
    TrainingHookStages,
)


@dataclass
class SB3Environment:
    train_env: VecEnv
    eval_env: VecEnv

    def close(self) -> None:
        """Close the training and evaluation environments."""
        self.train_env.close()
        self.eval_env.close()


class StableBaselines3Trainer(ArenaTrainer):
    """
    StableBaselines3Trainer class for managing RL training with the Stable Baselines 3 framework.

    This trainer handles the setup and execution of RL training processes using the
    Stable Baselines 3 framework in the arena-rosnav environment. It manages configuration,
    environment setup, agent initialization, and training workflow.

    Attributes:
        __framework (SupportedRLFrameworks): Identifier for the SB3 framework
        config_manager (SB3ConfigManager): Manages the training configuration
        environment (SB3Environment): Contains training and evaluation environments
        general_cfg: General training configuration parameters
        monitoring_cfg: Configuration for monitoring tools
        task_cfg: Task-specific configuration
        normalization_cfg: Configuration for observation normalization
        callbacks_cfg: Configuration for training callbacks
        profiling_cfg: Configuration for performance profiling
        robot_cfg: Robot configuration parameters
        agent_cfg: Agent-specific configuration

    Methods:
        __init__(config): Initialize the trainer with configuration
        __unpack_config(): Extract SB3-specific configuration fields
        _register_framework_specific_hooks(): Register SB3-specific training hooks
        _setup_agent(): Initialize the RL agent
        _setup_environment(): Set up training and evaluation environments
        _create_environments(): Create the environments for training and evaluation
        _setup_callbacks(environment): Initialize training callbacks
        _setup_simulation_state_container(): Initialize simulation state container
        _setup_agent_state_container(): Set up the agent state container
        _setup_monitoring(): Configure monitoring tools like W&B
        _train_impl(*args, **kwargs): Core training implementation
        _complete_model_initialization(train_env): Finalize model initialization
    """

    __framework = SupportedRLFrameworks.STABLE_BASELINES3
    _config_type = arena_cfg.ArenaSB3Cfg
    environment: SB3Environment

    def _register_framework_specific_hooks(self) -> None:
        """Register framework specific hooks for training.

        This method sets up the following hooks:
            - BEFORE_SETUP:
                - Set the curriculum file parameter in the task generator server
                - Set the current curriculum stage in the task generator server
            - AFTER_SETUP:
                - Transfer weights from a source model if configured

        The hooks are executed at specific stages during the training process.
        """

        def _transfer_weights(_):
            transfer_cfg = self.config.agent_cfg.framework.algorithm.transfer_weights
            if transfer_cfg:
                self.agent.model.transfer_weights(
                    source_dir=transfer_cfg.source_dir,
                    source_checkpoint=transfer_cfg.source_checkpoint,
                    include=transfer_cfg.include,
                    exclude=transfer_cfg.exclude,
                )

        self.hook_manager.register(
            TrainingHookStages.AFTER_SETUP,
            [_transfer_weights],
        )

    def _setup_agent(self) -> None:
        """Initializes the reinforcement learning agent.

        This method creates an instance of the RL_Agent class using the
        configuration parameters specified in the config object and the
        current agent state container.

        Returns:
            None
        """
        import rosnav_rl

        self.agent = rosnav_rl.RL_Agent(
            agent_cfg=self.config.agent_cfg,
            agent_state_container=self.agent_state_container,
        )

    def _setup_environment(self) -> None:
        """Sets up the training environment.

        This method performs the following steps:
            1. Creates the necessary environments.
            2. Sets up callbacks for the environment.
            3. Completes the model initialization using the training environment.

        Returns:
            None
        """
        self._create_environments()
        self._setup_callbacks(self.environment)
        self._complete_model_initialization(self.environment.train_env)

    def _create_environments(self) -> None:
        """Creates training and evaluation environments for the RL agent.

        This method initializes the training and evaluation environments using the specified configurations.
        It first creates environment function factories for both training and evaluation environments
        using the `make_envs` function. Then, it wraps these environments for compatibility with
        Stable Baselines 3 (SB3) using the `sb3_wrap_env` function. Finally, it sets up the environments
        for the agent and stores them in an SB3Environment container.
        The environments are configured with parameters from the trainer's config, including the number
        of environments, maximum steps per episode, and whether to initialize environments on call.

        Returns:
            None

        Note:
            Training environments use the TRAIN_NS namespace, while evaluation environments use EVAL_NS.
        """

        train_env_fncs = make_envs(
            node=self._supervisor_node,
            rl_agent=self.agent,
            n_envs=self.config.arena_cfg.general.n_envs,
            max_steps=self.config.arena_cfg.general.max_num_moves_per_eps,
            init_env_by_call=not self.config.arena_cfg.general.debug_mode,
            namespace_fn=lambda _: "/task_generator_node/jackal",
            simulation_state_container=self.simulation_state_container,
            wrappers=[partial(TimeSyncWrapper, control_hz=1)],
        )
        eval_env_fncs = make_envs(
            node=self._supervisor_node,
            rl_agent=self.agent,
            n_envs=1,
            namespace_fn=lambda _: "/task_generator_node/jackal",
            max_steps=self.config.arena_cfg.callbacks.periodic_evaluation.max_num_moves_per_eps,
            init_env_by_call=False,
            simulation_state_container=self.simulation_state_container,
            wrappers=[partial(TimeSyncWrapper, control_hz=1)],
        )
        train_env, eval_env = sb3_wrap_env(
            node=self._supervisor_node,
            train_env_fncs=train_env_fncs,
            eval_env_fncs=eval_env_fncs,
            general_cfg=self.config.arena_cfg.general,
            monitoring_cfg=self.config.arena_cfg.monitoring,
            profiling_cfg=self.config.arena_cfg.profiling,
        )
        train_env = self.agent.model.setup_environment(train_env)
        eval_env = self.agent.model.setup_environment(eval_env, is_training=False)
        self.environment = SB3Environment(train_env, eval_env)

    def _setup_callbacks(self, environment: SB3Environment) -> None:
        """Initialize training callbacks."""
        self.eval_cb = init_sb3_callbacks(
            node=self._supervisor_node,
            eval_env=environment.eval_env,
            n_envs=self.config.arena_cfg.general.n_envs,
            tm_modules=self.config.arena_cfg.task.tm_modules,
            callback_cfg=self.config.arena_cfg.callbacks,
            task_cfg=self.config.arena_cfg.task,
            model_save_path=self.paths[Paths.Agent].path,
            eval_log_path=self.paths[Paths.AgentEval].path,
            debug_mode=self.config.arena_cfg.general.debug_mode,
        )

    def _setup_monitoring(self) -> None:
        """Set up monitoring tools if not in debug mode."""
        if (
            not self.config.arena_cfg.general.debug_mode
            and self.config.arena_cfg.monitoring.wandb
        ):
            setup_wandb(
                run_name=self.config.agent_cfg.name,
                group=self.config.agent_cfg.framework.algorithm.architecture_name,
                config=self.config,
                to_watch=[self.agent.model.model.policy],
                agent_id=self.config.agent_cfg.name,
            )

    def _train_impl(self, *args, **kwargs) -> None:
        """Implementation of training logic."""
        self.agent.train(
            total_timesteps=self.config.agent_cfg.framework.algorithm.parameters.total_timesteps,
            callback=self.eval_cb,
            progress_bar=self.config.agent_cfg.framework.algorithm.parameters.show_progress_bar,
        )

    def _complete_model_initialization(self, train_env: VecEnv) -> None:
        """Complete the initialization of the RL model.

        This method sets up the tensorboard logging directory and checkpoint path for model resumption,
        then initializes the model with the provided training environment.

        Args:
            train_env (VecEnv): The vectorized training environment to use for model initialization.

        Returns:
            None

        Note:
            - Tensorboard logs are disabled in debug mode.
            - Checkpoint path is only used when resuming training from a previous checkpoint.
            - After initialization, the training environment is attached to the model.
        """

        tensorboard_log_path = (
            self.paths[Paths.AgentTensorboard].path
            if not self.config.arena_cfg.general.debug_mode
            else None
        )
        checkpoint_path = (
            None
            if not self.is_resume
            else self.paths[Paths.Agent].path
            / self.config.agent_cfg.framework.algorithm.checkpoint
        )

        self.agent.initialize_model(
            env=train_env,
            no_gpu=self.config.arena_cfg.general.no_gpu,
            tensorboard_log_path=tensorboard_log_path,
            checkpoint_path=checkpoint_path,
        )
        self.agent.model.environment = train_env


def main():
    rclpy.init()
    config = load_training_config("sb_training_config")

    trainer = StableBaselines3Trainer(config)

    trainer.train()
    trainer.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
