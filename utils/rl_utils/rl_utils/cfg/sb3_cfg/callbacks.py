from rosnav_rl.model.stable_baselines3.cfg import CallbacksCfg


# TrainingCurriculumCfg moved to task.staged configuration
# This maintains the legacy model validator behavior for backwards compatibility
class TrainingCurriculumCfg:
    """Legacy class - curriculum configuration moved to task.staged section."""

    pass


class ArenaCallbacksCfg(CallbacksCfg):
    # training_curriculum configuration moved to task.staged section
    pass
