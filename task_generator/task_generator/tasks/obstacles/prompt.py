from task_generator.tasks.obstacles import Obstacle, DynamicObstacle, Obstacles, TM_Obstacles
import attrs
from arena_rclpy_mixins.ROSParamServer import ROSParamT
import os
from huggingface_hub import InferenceClient
import json
import itertools

@attrs.define()
class _ParsedConfig:
    static: list[Obstacle]
    dynamic: list[DynamicObstacle]

class TM_Prompt(TM_Obstacles):
    """
    Prompt task generator for obstacles.

    This class generates obstacles based on a prompt configuration.

    Attributes:
        _config (Config): Configuration object for obstacle generation.

    Methods:
        __init__(**kwargs): Initializes the TM_Prompt object.
        reset(**kwargs): Resets the obstacle generation with the specified parameters.
    """

    _config: ROSParamT[_ParsedConfig]

    def _prompt_to_config(self, prompt: str) -> dict:
        return {} # Out of credits
        response = self.inference_client.chat.completions.create(
            model="moonshotai/Kimi-K2-Instruct",
            messages=[
                {"role": "system", "content": self.context},
                {"role": "user", "content": f"Generate pedestrian waypoints for a simulation where: {prompt}. Only return valid JSON under the 'dynamic' field, using the format above,  with no explanation, thoughts, or extra text."}
            ],
            temperature=0.3,
            top_p=0.9,
            stream=False,
        )

        answer = response.choices[0].message.content

        if answer.startswith("```json"):
            answer = answer.strip("```json").strip("```").strip()
        elif answer.startswith("```"):
            answer = answer.strip("```").strip()

        # Parse it into a Python dict
        try:
            config = json.loads(answer)
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", e)
            config = None

        return config
            
    def _parse_prompt(self, prompt: str) -> _ParsedConfig:
        """
        Parses the prompt to generate obstacles config.

        Args:
            prompt (str): The prompt for generating obstacles config.

        Returns:
            _ParsedConfig: Parsed configuration containing static and dynamic obstacles.
        """
        config = self._prompt_to_config(prompt)

        static_obstacles: list[Obstacle]
        dynamic_obstacles: list[DynamicObstacle]

        static_obstacles = [
            Obstacle.parse(obs)
            for obs
            in itertools.chain(
                config.get("obstacles", {}).get("static", []),
                config.get("obstacles", {}).get("interactive", []),
            )
        ]

        dynamic_obstacles = [
            DynamicObstacle.parse(obs)
            for obs
            in config.get("obstacles", {}).get("dynamic", [])
        ]

        return _ParsedConfig(static=static_obstacles, dynamic=dynamic_obstacles)

    def reset(self, **kwargs) -> Obstacles:
        return self._config.value.static, self._config.value.dynamic

    def __init__(self, **kwargs):
        TM_Obstacles.__init__(self, **kwargs)
        self.inference_client = InferenceClient(
            provider="together",
            api_key=os.environ["HF_TOKEN"],
        )

        self.context = """
            You are a simulator agent that outputs only JSON-formatted data for pedestrian simulation in an indoor hospital environment.

            Output must strictly follow this structure:
            ```json
            "obstacles": {
            "static": [],
            "dynamic": [
                {
                "name": "1",
                "pos": [24.0, 2.0, 0],
                "type": "adult",
                "model": "gazebo_actor",
                "waypoints": [[27.1, 7.0, 0], [17.7, 7.0, 0]],
                "waypoint_mode": 1
                }
            ]
            }
            Do NOT explain anything. Output JSON only. Use realistic (x, y, 0) coordinates.
        """
        
        self._config = self.node.ROSParam[_ParsedConfig](
            self.namespace('user_prompt'),
            default='empty space',
            parse=self._parse_prompt
        )
