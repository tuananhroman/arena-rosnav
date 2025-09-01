from task_generator.tasks.obstacles import Obstacle, DynamicObstacle, CustomDynamicObstacle, Obstacles, CustomObstacles, TM_Obstacles
import attrs
from arena_rclpy_mixins.ROSParamServer import ROSParamT
import os
from arena_simulation_setup.worlds.world import World
import json
import itertools
import time
import yaml
from google import genai
import chromadb
from task_generator.simulators.human.hunav.hunav import HunavDynamicObstacle
from ament_index_python.packages import get_package_share_directory
from task_generator.tasks.obstacles.prompt_utils import ARENA_CONTEXT, BEHAVIOR_TREE_CONTEXT, LOCAL_LM, REMOTE_LM, CHROMA_DB_PATH, BT_REF_DOC_PATH, Root, process_json_doc, create_chroma_db, get_chroma_collection, get_relevant_bt_nodes
import pprint
import tempfile
import xml.etree.ElementTree as ET
from typing import Dict


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


    def llm_bt_output_to_config(self, llm_output: Dict)-> Dict:
        try:
            config = {
                "obstacles": {
                    "static": [],
                    "dynamic": []
                }
            }

            tmp_dir = tempfile.TemporaryDirectory()
            for id, hunav in enumerate(llm_output.get("hunav_agents")):
                hunav: Dict

                hunav_config = {
                    "id": id,
                    "name": hunav.get("name"),
                    "pos": hunav.get("pos"),
                    "model": hunav.get("model"),
                    "waypoints": hunav.get("waypoints")
                }

                bt_root: Dict = hunav.get("bt_root")
                behavior_tree_xml = Root.model_validate_json(json.dumps(bt_root)).to_xml()
                
                with tempfile.NamedTemporaryFile(
                    mode='w+t', 
                    suffix='.xml',
                    dir=tmp_dir.name,
                    delete=False
                ) as tmp_xml_file:
                    hunav_config.update({
                        "behavior_tree": os.path.join(tmp_dir.name, tmp_xml_file.name)
                    })
                    tmp_xml_file.write(
                        ET.tostring(
                            behavior_tree_xml, 
                            encoding="UTF-8", 
                            method='xml', 
                            xml_declaration=True
                        ).decode("utf-8")
                    )

                    with open(f"/home/nguyen/{id}.xml", 'w+t') as file:
                        file.write(
                            ET.tostring(
                                behavior_tree_xml, 
                                encoding="UTF-8", 
                                method='xml', 
                                xml_declaration=True
                            ).decode("utf-8")
                        )
                
                config["obstacles"]["dynamic"].append(hunav_config)
                
        except Exception as e:
            self.node.get_logger().error(f"Failed to parse Behavior tree from LLM response: {e}")
            self.node.get_logger().error("Returning empty config!")
            config = {}

        return config
    

    def setup_chroma(self):
        if os.path.isdir(CHROMA_DB_PATH):
            self.chroma_collection = get_chroma_collection(CHROMA_DB_PATH, self.inference_client)
        else:
            processed_doc = process_json_doc(
                BT_REF_DOC_PATH
            )
            self.chroma_collection = create_chroma_db(
                documents=processed_doc,
                db_path=CHROMA_DB_PATH,
                client=self.inference_client
            )


    def _prompt_to_config(self, prompt: str, top_p: float, use_behavior_tree: bool, local: bool=False) -> dict:
        world = World(self.node._world_manager.world_name)
        with open(world.world_path) as file:
            world_description = yaml.safe_load(file)

        world_info = self.preprocess_world_description(world_description)

        messages = []

        if use_behavior_tree:
            self.setup_chroma()

            if "bt" not in self.cached_context.keys(): # system context is not cached (due to initialization)
                cache = self.inference_client.caches.create(
                    model=REMOTE_LM,
                    config=genai.types.CreateCachedContentConfig(
                        display_name="bt-context",
                        system_instruction="You always stick to the facts in the sources provided, and never make up new facts. Now look at these provided materials, and answer the following questions.",
                        contents=BEHAVIOR_TREE_CONTEXT
                    )
                )
                self.cached_context.update({"bt": cache.name})
                
            bt_nodes = get_relevant_bt_nodes(
                query=f"What are the nodes should be used for creating the behavior tree as described below: \"{prompt}\"",
                collection=self.chroma_collection,
            )

            messages.append(
                f"Generate hunav agents data for a simulation base on this world data as below: {world_info}, where: {prompt}. Use these behavior tree nodes only: {bt_nodes}. Only return valid JSON using the format declared in the system context, with no explanation, thoughts, or extra text."
            )

        else:
            if "arena" not in self.cached_context.keys(): # system context is not cached (due to initialization)
                cache = self.inference_client.caches.create(
                    model=REMOTE_LM,
                    config=genai.types.CreateCachedContentConfig(
                        display_name="arena-context",
                        system_instruction="You always stick to the facts in the sources provided, and never make up new facts. Now look at these provided materials, and answer the following questions.",
                        contents=ARENA_CONTEXT
                    )
                )
                self.cached_context.update({"arena": cache.name})

            messages.append(
                f"Generate pedestrian waypoints for a simulation base on this world data as below: {world_info}, where: {prompt}. Only return valid JSON under the 'dynamic' field, using the format declared in the system context, with no explanation, thoughts, or extra text."
            )
        
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
            self.node.get_logger().warn("Start inference...")
            start = time.time()
            response = self.inference_client.models.generate_content(
                model=REMOTE_LM,
                contents=messages,
                config=genai.types.GenerateContentConfig(
                    cached_content=self.cached_context["bt"] if use_behavior_tree else self.cached_context["arena"],
                    top_p=top_p,
                    thinking_config=genai.types.ThinkingConfig(
                        include_thoughts=False,
                        thinking_budget=0
                    ),
                )
            )

            answer = response.text
            end = time.time()
            self.node.get_logger().warn(f"Inference done, took: {end-start:.1f}s")

        if answer.startswith("```json"):
            answer = answer.strip("```json").strip("```").strip()
        elif answer.startswith("```"):
            answer = answer.strip("```").strip()

        # Parse it into a Python dict
        try:
            if use_behavior_tree:
                with open("/home/nguyen/test_llm_output.json", "w") as file:
                    json.dump(json.loads(answer), file)
                config = self.llm_bt_output_to_config(json.loads(answer))
            else:
                config = json.loads(answer)

        except json.JSONDecodeError as e:
            self.node.get_logger().error(f"Failed to parse JSON from LLM response: {e}")
            self.node.get_logger().error("Returning empty config!")
            config = {}

        with open("/home/nguyen/scenario.json", "w") as file:
            json.dump(config, file)

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
            
        # default_hunav_config = _load_config() # Is not used yet

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


        if "GEMINI_API_KEY" not in os.environ:
                self.node.get_logger().error("GEMINI_API_KEY environment variable not set!")
                self.node.get_logger().error("Returning empty config!")
                return {}
        
        self.inference_client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"]
        )

        self.cached_context: Dict[str, str] = {}  # Whether the prompt context need to be changed and fed into LLM model 