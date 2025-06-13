import os
from typing import TYPE_CHECKING


import yaml
from pydantic import BaseModel
from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import get_lexer_by_name

if TYPE_CHECKING:
    from rl_utils.trainer.arena_trainer import ArenaTrainer

from rl_utils.utils.paths import PathDictionary, PathFactory
from ament_index_python.packages import get_package_share_directory


def write_config_yaml(config: dict, path: str) -> None:
    with open(path, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def print_dict(hyperparams: dict) -> None:
    print("\n--------------------------------")
    print("         HYPERPARAMETERS         \n")
    for param, param_val in hyperparams.items():
        print("{:30s}{:<10s}".format(f"{param}:", str(param_val)))
    print("--------------------------------\n\n")


def print_base_model(hyperparams: BaseModel) -> None:
    print("\n--------------------------------")
    print("         HYPERPARAMETERS         \n")
    yaml_str = yaml.dump(
        hyperparams.model_dump(), default_flow_style=False, sort_keys=False
    )
    colorful_yaml = highlight(yaml_str, get_lexer_by_name("yaml"), TerminalFormatter())
    print(colorful_yaml)
    print("--------------------------------\n\n")


def create_directories(
    paths: dict,
    resume_name: str,
    log_evaluation: bool,
    use_wandb: bool,
) -> None:
    create_model_directory(paths, resume_name)
    paths["eval"] = create_evaluation_directory(paths, log_evaluation)
    paths["tb"] = create_tensorboard_directory(paths, use_wandb)


def create_model_directory(paths: dict, resume_name: str) -> None:
    """
    Create model directory if not in debug mode and resume_name is None.
    Raise FileNotFoundError if checkpoint is not found.

    :param PATHS: Dictionary containing paths
    :param resume_name: Name of the resume file
    :param checkpoint_name: Name of the checkpoint file
    :param debug_mode: Boolean indicating if debug mode is enabled
    """
    if resume_name is None:
        os.makedirs(paths["model"])


def create_evaluation_directory(paths: dict, log_evaluation: bool) -> str:
    """
    Create evaluation directory if log_evaluation is enabled and not in debug mode.

    :param PATHS: Dictionary containing paths
    :param log_evaluation: Boolean indicating if evaluation logging is enabled
    :return: Path to evaluation directory or None
    """
    if log_evaluation:
        if not os.path.exists(paths["eval"]):
            os.makedirs(paths["eval"])
        return paths["eval"]
    return None


def create_tensorboard_directory(paths: dict, use_wandb: bool) -> str:
    """
    Create tensorboard directory if use_wandb is enabled and not in debug mode.

    :param PATHS: Dictionary containing paths
    :param use_wandb: Boolean indicating if wandb is used
    :param debug_mode: Boolean indicating if debug mode is enabled
    :return: Path to tensorboard directory or None
    """
    if use_wandb:
        if not os.path.exists(paths["tb"]):
            os.makedirs(paths["tb"])
        return paths["tb"]
    return None


def load_config(file_path: str) -> dict:
    """
    Load config parameters from config file
    """
    with open(file_path, "r", encoding="utf-8") as target:
        config = yaml.load(target, Loader=yaml.FullLoader)
    return config


def get_robot_yaml_path() -> str:
    robot_model = "jackal"  # TODO: Fetch robot from parameter server

    simulation_setup_path = get_package_share_directory("arena_simulation_setup")
    return os.path.join(
        simulation_setup_path, "entities", "robots", robot_model, f"model_params.yaml"
    )


def setup_paths_dictionary(
    trainer: "ArenaTrainer", is_debug_mode: bool = False
) -> PathDictionary:
    trainer.paths = PathFactory.get_paths(trainer.config.agent_cfg.name)
    if not is_debug_mode:
        trainer.paths.create_all()


def load_yaml(file_path: str) -> dict:
    with open(file_path) as file:
        return yaml.load(file, Loader=yaml.FullLoader)
