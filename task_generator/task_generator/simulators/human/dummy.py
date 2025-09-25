from collections.abc import Sequence
from task_generator.simulators.human import BaseHumanSimulator
from task_generator.shared import DynamicObstacle, Obstacle


class DummyHumanSimulator(BaseHumanSimulator):

    def _spawn_obstacles_impl(
        self,
        obstacles,
    ) -> Sequence[Obstacle | None]:
        return obstacles

    def _spawn_dynamic_obstacles_impl(
        self,
        obstacles,
    ) -> Sequence[DynamicObstacle | None]:
        return obstacles

    def _remove_obstacles_impl(
        self,
    ) -> bool:
        return True

    def _spawn_walls_impl(
        self,
        walls,
    ) -> bool:
        return True

    def _spawn_doors_impl(
        self,
        doors,
    ) -> bool:
        return True

    def _spawn_robot_impl(
        self,
        robots,
    ) -> Sequence[bool]:
        return (True,) * len(robots)

    def _remove_robot_impl(
        self,
        robots,
    ) -> Sequence[bool]:
        return (True,) * len(robots)

    def _move_robot_impl(
        self,
        robots,
    ) -> Sequence[bool]:
        return (True,) * len(robots)
