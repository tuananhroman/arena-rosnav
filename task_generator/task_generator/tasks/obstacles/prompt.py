from task_generator.tasks.obstacles import Obstacle, DynamicObstacle, CustomDynamicObstacle, Obstacles, CustomObstacles, TM_Obstacles
import attrs
from arena_rclpy_mixins.ROSParamServer import ROSParamT
import os
from arena_simulation_setup.worlds.world import World
import json
import itertools
import time
import yaml
from openai import OpenAI
from task_generator.simulators.human.hunav.hunav import HunavDynamicObstacle
from ament_index_python.packages import get_package_share_directory
from task_generator.tasks.obstacles.prompt_utils import ARENA_CONTEXT, BEHAVIOR_TREE_CONTEXT, LOCAL_LM, REMOTE_LM, Root


@attrs.define()
class _ParsedConfig:
    static: list[Obstacle]
    dynamic: list[DynamicObstacle]


@attrs.define()
class PromptConfig:
    user_prompt: ROSParamT[str]
    top_p: ROSParamT[float]
    behavior_tree: ROSParamT[bool]


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

    _config: PromptConfig


    def preprocess_world_description(self, world_description: dict) -> str:
        """
        Preprocesses the world description, keeps corners and walls only and converts them to 2D format.

        Args:
            world_description : dict
                The world description to preprocess.

        Returns:
            parsed : str
                The preprocessed JSON formatted str world description.
        """
        parsed = {}

        parsed["zones"] = []
        for zone in world_description.get("zones", []):
            parsed_zone = {
                "name": zone.get("name", ""),
                "corners": [[corner['x'], corner['y']] for corner in zone.get("corners", [])],
                "walls": [[[wall['start']['x'], wall['start']['y']], [wall['end']['x'], wall['end']['y']]] for wall in zone.get("walls", [])],
            }
            parsed["zones"].append(parsed_zone)

        return json.dumps(parsed, indent=2)


    def _prompt_to_config(self, prompt: str, top_p: float, use_behavior_tree: bool, local: bool=False) -> dict:
        world = World(self.node._world_manager.world_name)
        with open(world.world_path) as file:
            world_description = yaml.safe_load(file)

        world_info = self.preprocess_world_description(world_description)

        if use_behavior_tree:
            messages = [
                {
                    "role": "system",
                    "content": f"{self.use_behavior_tree_context}. Generate data base on this world data as below: {world_info}"
                },
                {
                    "role": "user", 
                    "content": f"Generate pedestrian waypoints for a simulation where: {prompt}. Only return valid JSON under the 'dynamic' field, using the format above,  with no explanation, thoughts, or extra text."
                }
            ]
        else:
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
        if local: # Currently not supported
            return {}
            from huggingface_hub import InferenceClient
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(LOCAL_LM, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(LOCAL_LM)
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
            if "GEMINI_API_KEY" not in os.environ:
                self.node.get_logger().error("GEMINI_API_KEY environment variable not set!")
                self.node.get_logger().error("Returning empty config!")
                return {}
            
            self.inference_client = OpenAI(
                api_key=os.environ["GEMINI_API_KEY"],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            self.node.get_logger().warn("Start inference...")
            start = time.time()
            response = self.inference_client.chat.completions.create(
                model=REMOTE_LM,
                messages=messages,
                top_p=top_p,
                stream=False,
            )

            answer = response.choices[0].message.content
            end = time.time()
            self.node.get_logger().warn(f"Inference done, took: {end-start:.1f}s")

        if answer.startswith("```json"):
            answer = answer.strip("```json").strip("```").strip()
        elif answer.startswith("```"):
            answer = answer.strip("```").strip()

        # Parse it into a Python dict
        try:
            config = json.loads(answer)
            if use_behavior_tree:
                behavior_tree_xml = Root.parse_raw(answer).to_xml()

                return behavior_tree_xml

        except json.JSONDecodeError as e:
            self.node.get_logger().error("Failed to parse JSON from LLM response:", e)
            self.node.get_logger().error("Returning empty config!")
            config = {}

        # with open("/home/nguyen/test_llm_output.json", "w") as file:
        #     json.dump(config, file)

        return config

    def _parse_prompt(self, prompt: str, top_p: float, use_behavior_tree: bool) -> _ParsedConfig:
        """
        Parses the prompt to generate obstacles config.

        Args:
            prompt (str): The prompt for generating obstacles config.

        Returns:
            _ParsedConfig: Parsed configuration containing static and dynamic obstacles.
        """
        config = self._prompt_to_config(prompt, top_p, use_behavior_tree)

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
            CustomDynamicObstacle.parse(obs)
            for obs
            in config.get("obstacles", {}).get("dynamic", [])
        ]

        return _ParsedConfig(static=static_obstacles, dynamic=dynamic_obstacles)

    def reset(self, **kwargs) -> CustomObstacles:
        user_prompt: str = kwargs.get(
            "user_prompt",
            self._config.user_prompt.value,
        )

        top_p: float = kwargs.get(
            "top_p",
            self._config.top_p.value
        )

        use_behavior_tree: bool = kwargs.get(
            "behavior_tree",
            self._config.behavior_tree.value,
        )

        parsed_config = self._parse_prompt(user_prompt, top_p, use_behavior_tree)

        return parsed_config.static, parsed_config.dynamic

    def __init__(self, **kwargs):
        TM_Obstacles.__init__(self, **kwargs)
        # self.inference_client = InferenceClient(
        #     provider="together",
        #     api_key=os.environ["HF_TOKEN"],
        # )

        def _load_config(filename: str = "default.yaml") -> "HunavDynamicObstacle":
            """Load config from YAML file in arena_bringup configs."""

            # second priority: Install space
            config_path = os.path.join(
                get_package_share_directory("arena_bringup"),
                "configs",
                "hunav_agents",
                filename
            )

            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)

                agent_config = config['hunav_loader']['ros__parameters']['agent1']
                return agent_config
            
            except Exception as e:
                raise RuntimeError(f"Error loading config from {config_path}") from e
            
        default_hunav_config = _load_config() # Is not used yet

        self.context = ARENA_CONTEXT

        self._config = PromptConfig(
            user_prompt=self.node.ROSParam[str](
                self.namespace('user_prompt'),
                value='An empty space with no pedestrian.',
                parse=lambda prompt: prompt
            ),
            top_p=self.node.ROSParam[float](
                self.namespace('top_p'),
                value=0.3,
                parse=lambda p: p
            ),
            behavior_tree=self.node.ROSParam[bool](
                self.namespace('behavior_tree'),
                value=False,
                parse=lambda use: use
            )
        )
