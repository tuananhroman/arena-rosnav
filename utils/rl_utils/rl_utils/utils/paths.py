import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, Optional, Type

from ament_index_python.packages import get_package_share_directory

# import rospy

__all__ = [
    "PathComponent",
    "Agent",
    "AgentTensorboard",
    "AgentEval",
    "ConfigComponent",
    "TrainingCurriculum",
    "RewardFunction",
    "RobotSetting",
    "PathFactory",
    "RosPackages",
]


@dataclass(frozen=True)
class RosPackages:
    """Centralized ROS package paths"""

    # TODO: Consider using a more dynamic approach to fetch package paths
    SIMULATION_SETUP: Path = Path(get_package_share_directory("arena_simulation_setup"))
    ROSNAV: Path = Path(get_package_share_directory("rosnav_rl"))
    ARENA_BRINGUP: Path = Path(get_package_share_directory("arena_bringup"))


class PathComponent(ABC):
    """Base class for all path components"""

    @cached_property
    @abstractmethod
    def path(self) -> Path:
        """Returns the complete path for this component"""
        pass

    def exists(self) -> bool:
        """Check if the path exists"""
        return self.path.exists()

    def create(self) -> Path:
        """Create directories if they don't exist"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return self.path


class AgentComponent(PathComponent):
    """Base class for agent-related paths"""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self._base = RosPackages.ROSNAV / "agents" / agent_name


class Agent(AgentComponent):
    """Main agent path"""

    @cached_property
    def path(self) -> Path:
        return self._base


class AgentLogs(AgentComponent):
    """Base class for agent log paths"""

    def __init__(self, agent_name: str, log_type: str):
        super().__init__(agent_name)
        self.log_type = log_type

    @cached_property
    def path(self) -> Path:
        return self._base / f"{self.log_type}_logs"


class AgentTensorboard(AgentLogs):
    """Agent tensorboard logs"""

    def __init__(self, agent_name: str):
        super().__init__(agent_name, "training")


class AgentEval(AgentLogs):
    """Agent evaluation logs"""

    def __init__(self, agent_name: str):
        super().__init__(agent_name, "eval")


class ConfigComponent(PathComponent):
    """Base class for configuration paths"""

    def __init__(self, file_name: str = ""):
        self.file_name = file_name
        self._base = RosPackages.ARENA_BRINGUP / "configs" / "training"

    @cached_property
    def path(self) -> Path:
        return self._base


class TrainingCurriculum(ConfigComponent):
    """Training curriculum paths"""

    @cached_property
    def path(self) -> Path:
        return self._base / "training_curriculums" / self.file_name


class RewardFunction(ConfigComponent):
    """Reward function paths"""

    @cached_property
    def path(self) -> Path:
        file_name = (
            f"{self.file_name}.yaml"
            if not self.file_name.endswith(".yaml")
            else self.file_name
        )
        return self._base / "reward_functions" / file_name


class RobotSetting(PathComponent):
    """Robot setting paths"""

    def __init__(self, robot_model: Optional[str] = None):
        # TODO: Consider fetching the default robot model from ROS parameters
        self.robot_model = (
            robot_model or "jackal"
        )  # Default to 'jackal' if not specified

    @cached_property
    def path(self) -> Path:
        return (
            RosPackages.SIMULATION_SETUP
            / "robot"
            / self.robot_model
            / f"{self.robot_model}.model.yaml"
        )


class PathDictionary(dict):
    def create_all(self):
        for path in self.values():
            path.create()


class PathFactory:
    """Factory class to create path instances"""

    @staticmethod
    def get_paths(
        agent_name: str,
    ) -> Dict[Type[PathComponent], PathComponent]:
        """Generate all required paths for the agent"""
        return PathDictionary(
            {
                Agent: Agent(agent_name),
                AgentTensorboard: AgentTensorboard(agent_name),
                AgentEval: AgentEval(agent_name),
                RobotSetting: RobotSetting(),
                ConfigComponent: ConfigComponent(),
            }
        )
