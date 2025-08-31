import rl_utils.cfg as cfg
from rl_utils.tools.constants import TRAINING_CONSTANTS

from .general import load_config


def load_training_config(path: str) -> cfg.TrainingCfg:
    return cfg.TrainingCfg.model_validate(
        load_config(path), strict=True, from_attributes=True
    )
