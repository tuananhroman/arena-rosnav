from task_generator.tasks.obstacles import Obstacle, DynamicObstacle, Obstacles, TM_Obstacles
import attrs
from arena_rclpy_mixins.ROSParamServer import ROSParamT
import os
from huggingface_hub import InferenceClient
from transformers import AutoModelForCausalLM, AutoTokenizer
from arena_simulation_setup.worlds.world import World
import json
import itertools
import time
import yaml


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

    def _prompt_to_config(self, prompt: str, local: bool=True) -> dict:
        world = World(self.node._world_manager.world_name)
        with open(world.world_path) as file:
            zones = yaml.safe_load(file).get("zones", {})
        world_info = json.dumps(zones)

        messages = [
            {
                "role": "system",
                "content": f"{self.context}. Generate data base on this world data as below: {world_info}"
            },
            {
                "role": "user", 
                "content": f"Generate pedestrian waypoints for a simulation where: {prompt}. Only return valid JSON under the 'dynamic' field, using the format above,  with no explanation, thoughts, or extra text."
            }
        ]

        if local:
            model_name = "Qwen/Qwen3-0.6B"

            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            # Format using Qwen chat template
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            self.node.get_logger().info("Start inference...")
            start = time.time()

            # Tokenize input
            inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
            
            # Generate output
            outputs = model.generate(
                **inputs,
                max_new_tokens=32768,
            )

            # Extract generated tokens (excluding prompt)
            generated_ids = outputs[0][len(inputs.input_ids[0]):]
            answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            end = time.time()
            self.node.get_logger().info(f"Inference done, took: {end-start:.1f}s")

        else:
            self.node.get_logger().info("Start inference...")
            start = time.time()
            response = self.inference_client.chat.completions.create(
                model="moonshotai/Kimi-K2-Instruct",
                messages=messages,
                temperature=0.3,
                top_p=0.9,
                stream=False,
            )

            answer = response.choices[0].message.content
            end = time.time()
            self.node.get_logger().info(f"Inference done, took: {end-start:.1f}s")

        if answer.startswith("```json"):
            answer = answer.strip("```json").strip("```").strip()
        elif answer.startswith("```"):
            answer = answer.strip("```").strip()

        # Parse it into a Python dict
        try:
            config = json.loads(answer)
        except json.JSONDecodeError as e:
            self.node.get_logger().error("Failed to parse JSON from LLM response:", e)
            self.node.get_logger().error("Returning empty config!")
            config = {}

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
            You are a simulator agent that outputs only JSON-formatted data for pedestrian simulation with provided specific information about the simulation map.

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

            The world information is provided in this JSON-formated data as described below: The map is composed of a list of zones. Each zone has the following fields:
            - `description`: a human-readable name of the zone.
            - `name`: a unique identifier.
            - `walls`: a list of wall segments, each defined by two 2D points [[x1, y1], [x2, y2]].
            - `corners`: a list of 2D points [x, y] marking the zone's corners.
            - `mat`: the material of the floor (can be empty).
            - `entities`: contains static objects in the zone. Each static object has:
            - `name`: the object's unique name.
            - `model`: the type of object (e.g., `shelf`).
            - `pose`: a list [x, y, yaw] representing the object's position and rotation.
        """

        self._config = self.node.ROSParam[_ParsedConfig](
            self.namespace('user_prompt'),
            value='empty space',
            parse=self._parse_prompt
        )
