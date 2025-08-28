from arena_rclpy_mixins.ROSParamServer import ROSParamT
from arena_simulation_setup.worlds.world import World
from arena_simulation_setup.worlds.scenario import Scenario
from task_generator.tasks.obstacles import Obstacles, TM_Obstacles


class TM_Scenario(TM_Obstacles):

    _config: ROSParamT[Scenario]

    def _parse_scenario(self, scenario: str) -> Scenario:
        return World(self.node._world_manager.world_name).scenario(scenario).load()

    def reset(self, **kwargs) -> Obstacles:
        return self._config.value.static, self._config.value.dynamic

    def __init__(self, **kwargs):
        TM_Obstacles.__init__(self, **kwargs)

        default_scenario = 'default'
        if default_scenario not in (scenarios := World(self.node._world_manager.world_name).scenario.list()):
            default_scenario = next(iter(scenarios), None)
        if default_scenario is None:
            raise ValueError(f"No scenarios found in world {self.node._world_manager.world_name}")

        self._config = self.node.ROSParam[Scenario](
            self.namespace('file'),
            default_scenario,
            parse=self._parse_scenario,
        )
