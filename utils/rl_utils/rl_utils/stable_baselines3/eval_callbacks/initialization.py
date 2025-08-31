from typing import TYPE_CHECKING, List, Optional

from rl_utils.node import SupervisorNode
from rl_utils.cfg.arena_cfg.task import TaskCfg
from rosnav_rl.utils.stable_baselines3.callbacks import (
    RosnavEvalCallback,
    StopTrainingOnRewardThreshold,
    StopTrainingOnSuccessThreshold,
)
from rosnav_rl.utils.stable_baselines3.staged_train_callback import StagedTrainCallback
from stable_baselines3.common.vec_env import VecEnv

if TYPE_CHECKING:
    from rl_utils.cfg.sb3_cfg import ArenaCallbacksCfg


def _create_stop_training_callbacks(
    threshold_type: str,
    threshold: float,
    verbose: int = 1,
) -> List:
    """Create stop training callbacks with direct parameters.

    Args:
        threshold_type: Type of threshold ("rew" or "succ")
        threshold: Threshold value
        verbose: Verbosity level

    Returns:
        List of stop training callback instances
    """
    if threshold_type == "rew":
        callback = StopTrainingOnRewardThreshold(
            reward_threshold=threshold,
            verbose=verbose,
        )
    else:  # "succ"
        callback = StopTrainingOnSuccessThreshold(
            success_threshold=threshold,
            verbose=verbose,
        )

    return [callback]


def init_sb3_callbacks(
    node: SupervisorNode,
    eval_env: VecEnv,
    n_envs: int,
    tm_modules: List[str],
    model_save_path: str,
    eval_log_path: str,
    callback_cfg: "ArenaCallbacksCfg",
    debug_mode: bool,
    task_cfg: Optional[TaskCfg] = None,
) -> RosnavEvalCallback:
    """
    Initialize Stable Baselines3 (SB3) callbacks for training and evaluation.

    Args:
        node: ROS2 supervisor node
        eval_env (VecEnv): The evaluation environment.
        n_envs (int): Number of environments.
        tm_modules (List[str]): List of training modules.
        model_save_path (str): Path to save the model.
        eval_log_path (str): Path to save evaluation logs.
        callback_cfg (CallbacksCfg): Configuration for callbacks.
        debug_mode (bool): If True, disables model saving for debugging purposes.

    Returns:
        RosnavEvalCallback: The evaluation callback configured with the specified settings.
    """
    # Extract configuration variables directly
    stop_train_cfg = callback_cfg.stop_training_on_threshold
    periodic_eval_cfg = callback_cfg.periodic_evaluation

    # Initialize callbacks
    callbacks = []

    # Add curriculum callback if staging is enabled
    if (
        "staged" in tm_modules
        and task_cfg
        and task_cfg.staged
        and task_cfg.staged.curriculum_definition
    ):
        curriculum_stages = {}
        for stage in task_cfg.staged.curriculum_definition:
            stage_dict = stage.model_dump(by_alias=True, exclude_none=True)
            for param_name, param_value in stage_dict.items():
                curriculum_stages.setdefault(param_name, []).append(param_value)

        if curriculum_stages:
            curriculum_cb = StagedTrainCallback(
                node=node,
                train_stages=curriculum_stages,
                threshold_type=task_cfg.staged.threshold_type,
                upper_threshold=task_cfg.staged.upper_threshold,
                lower_threshold=task_cfg.staged.lower_threshold,
                num_envs=n_envs,
                parameter_node_template=task_cfg.staged.parameter_node_template,
                timeout=task_cfg.staged.timeout,
                starting_stage=task_cfg.staged.starting_stage,
                verbose=1,
            )
            callbacks.append(curriculum_cb)

    # Add stop training callbacks
    if stop_train_cfg:
        stop_callbacks = _create_stop_training_callbacks(
            threshold_type=stop_train_cfg.threshold_type,
            threshold=stop_train_cfg.threshold,
            verbose=stop_train_cfg.verbose,
        )
        callbacks.extend(stop_callbacks)

    # Get curriculum callback for eval callback integration
    curriculum_cb = next(
        (cb for cb in callbacks if isinstance(cb, StagedTrainCallback)), None
    )

    # Get stop training callback for eval callback integration
    stop_cb = next(
        (
            cb
            for cb in callbacks
            if isinstance(
                cb, (StopTrainingOnRewardThreshold, StopTrainingOnSuccessThreshold)
            )
        ),
        None,
    )

    # Create main evaluation callback
    eval_cb = RosnavEvalCallback(
        eval_env=eval_env,
        n_eval_episodes=periodic_eval_cfg.n_eval_episodes,
        eval_freq=periodic_eval_cfg.eval_freq,
        log_path=eval_log_path,
        best_model_save_path=None if debug_mode else model_save_path,
        deterministic=True,
        callback_on_eval_end=curriculum_cb,
        callback_on_new_best=stop_cb,
    )
    return eval_cb
