import os
import random
import time
from typing import List, Optional

from rospkg import RosPack

from task_generator.constants import Constants
from task_generator.tasks.task_factory import TaskFactory
from task_generator.tasks.base_task import BaseTask
from task_generator.shared import DynamicObstacle, Obstacle, Waypoint

import xml.etree.ElementTree as ET

from task_generator.tasks.utils import ITF_Obstacle, ITF_Random, Scenario, ITF_Scenario, ScenarioMap, ScenarioObstacles
from task_generator.utils import rosparam_get


def get_attrib(element: ET.Element, attribute: str, default: Optional[str] = None) -> str:
    val = element.get(attribute)
    if val is not None:
        return str(val)

    sub_elem = element.find(attribute)
    if sub_elem is not None:
        return str(sub_elem.text)

    if default is not None:
        return default

    raise ValueError(f"attribute {attribute} not found in {element}")


@TaskFactory.register(Constants.TaskMode.PARAMETRIZED)
class ParametrizedTask(BaseTask):
    """
        The random task spawns static and dynamic
        obstacles on every reset and will create
        a new robot start and goal position for
        each task.
    """

    itf_scenario: ITF_Scenario
    itf_random: ITF_Random
    itf_obstacle: ITF_Obstacle

    def __init__(self, **kwargs):
        BaseTask.__init__(self, **kwargs)
        self.itf_scenario = ITF_Scenario(self)
        self.itf_random = ITF_Random(self)
        self.itf_obstacle = ITF_Obstacle(self)

    @BaseTask.reset_helper(parent=BaseTask)
    def reset(self, **kwargs):

        def callback():

            # TODO temp
            self.obstacle_manager.reset()
            # end

            self.obstacle_manager.respawn(
                lambda: self.itf_scenario.setup_scenario(self._generate_scenario()))
            time.sleep(1)

            return False

        return {}, callback

    def _generate_scenario(
        self,
        static_obstacles: Optional[int] = None,
        dynamic_obstacles: Optional[int] = None
    ) -> Scenario:

        robot_positions: List[Waypoint] = []  # may be needed in the future idk

        for manager in self.robot_managers:

            start_pos = self.map_manager.get_random_pos_on_map(
                manager.safe_distance)
            goal_pos = self.map_manager.get_random_pos_on_map(
                manager.safe_distance, forbidden_zones=[start_pos])

            manager.reset(start_pos=start_pos, goal_pos=goal_pos)

            robot_positions.append(start_pos)
            robot_positions.append(goal_pos)

        self.map_manager.init_forbidden_zones()

        obstacle_ranges = self.itf_random.load_obstacle_ranges()

        if dynamic_obstacles is None:
            dynamic_obstacles = random.randint(*obstacle_ranges.dynamic)

        if static_obstacles is None:
            static_obstacles = random.randint(*obstacle_ranges.static)

        xml_path = os.path.join(
            RosPack().get_path("arena_bringup"),
            "configs",
            "parametrized",
            rosparam_get(str, "~configuration/task_mode/parametrized/file")
        )

        tree = ET.parse(xml_path)
        root = tree.getroot()

        assert isinstance(
            root, ET.Element) and root.tag == "random", "not a random.xml desc"

        dynamic_obstacles_array: List[DynamicObstacle] = list()
        static_obstacles_array: List[Obstacle] = list()
        interactive_obstacles_array: List[Obstacle] = list()

        # Create static obstacles
        for config in root.findall("./static/obstacle") or []:
            for i in range(
                random.randint(
                    int(get_attrib(config, "min")),
                    int(get_attrib(config, "max"))
                )
            ):
                obstacle = self.itf_obstacle.create_obstacle(
                    name=f'{get_attrib(config, "name")}_static_{i+1}',
                    model=self.model_loader.bind(get_attrib(config, "model"))
                )
                obstacle.extra["type"] = get_attrib(config, "type", "")
                static_obstacles_array.append(obstacle)

        # Create interactive obstacles
        for config in root.findall("./interactive/obstacle") or []:
            for i in range(
                random.randint(
                    int(get_attrib(config, "min")),
                    int(get_attrib(config, "max"))
                )
            ):
                obstacle = self.itf_obstacle.create_obstacle(
                    name=f'{get_attrib(config, "name")}_interactive_{i+1}',
                    model=self.model_loader.bind(get_attrib(config, "model"))
                )
                obstacle.extra["type"] = get_attrib(config, "type", "")
                static_obstacles_array.append(obstacle)

        # Create dynamic obstacles
        for config in root.findall("./dynamic/obstacle") or []:
            for i in range(
                random.randint(
                    int(get_attrib(config, "min")),
                    int(get_attrib(config, "max"))
                )
            ):
                obstacle = self.itf_obstacle.create_dynamic_obstacle(
                    name=f'{get_attrib(config, "name")}_dynamic_{i+1}',
                    model=self.dynamic_model_loader.bind(
                        get_attrib(config, "model"))
                )
                obstacle.extra["type"] = get_attrib(config, "type", "")
                dynamic_obstacles_array.append(obstacle)

        return Scenario(
            obstacles=ScenarioObstacles(
                dynamic=dynamic_obstacles_array,
                static=static_obstacles_array,
                interactive=interactive_obstacles_array
            ),
            map=ScenarioMap(
                yaml=dict(),
                xml=ET.ElementTree(
                    ET.Element("dummy")),
                path=""
            ),
            robots=[]
        )
