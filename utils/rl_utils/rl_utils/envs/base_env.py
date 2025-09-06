import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Twist

# from rosnav_rl.observations import (
#     DoneObservation,
#     ObservationCollectorUnit,
#     ObservationManager,
#     get_required_observation_units,
from rosnav_rl.observations.factory.factory import (
    create_observation_manager_from_config,
)
from rosnav_rl.reward.reward_function import RewardFunction
from rosnav_rl.spaces import BaseSpaceManager
from rosnav_rl.states import SimulationStateContainer
from rosnav_rl.utils.rostopic import Namespace
from rosnav_rl.utils.type_aliases import EncodedObservationDict, ObservationDict
from rosnav_rl_msgs.srv import GetCommand
from std_srvs.srv import Empty as EmptySrv

from rl_utils.node import SupervisorNode
from rl_utils.utils.envs import determine_termination, get_twist_from_action
from rl_utils.utils.type_alias.observation import InformationDict

from rosnav_rl.utils.logging import flush_errors_decorator


class ArenaBaseEnv(ABC, gymnasium.Env):
    """Abstract base class for Arena reinforcement learning environments.

    This class provides a foundational structure for creating Gymnasium-compliant
    environments that interact with a ROS2-based simulation. It manages the
    synchronization between the agent's actions and the simulation's command
    requests, handles observation collection, reward computation, and episode
    management.

    The core synchronization mechanism relies on a ROS2 service. The simulation
    requests a command, which blocks the service call. The `step()` method,
    running in a separate thread (e.g., the main RL training loop), waits for
    this request, provides the agent's action as a response, and then proceeds
    with its own logic.

    Attributes:
        node (SupervisorNode): The ROS2 node used for communication.
        ns (Namespace): The ROS2 namespace for the agent.
        action_space (gymnasium.spaces.Box): The action space, defined by the space manager.
        observation_space (gymnasium.spaces.Dict): The observation space, defined by the space manager.
        is_train_mode (bool): Flag indicating if the environment is in training mode.
        observation_collector (ObservationManager): Manages the collection of observations.
        metadata (dict): Standard Gymnasium metadata.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        ns: Union[str, Namespace],
        space_manager: Union[BaseSpaceManager, Dict[str, Any]],
        reward_function: Union[RewardFunction, Dict[str, Any]],
        node: Optional[SupervisorNode] = None,
        simulation_state_container: Optional[SimulationStateContainer] = None,
        max_steps_per_episode: int = 100,
        init_by_call: bool = False,
        wait_for_obs: bool = False,
        obs_unit_kwargs: Optional[Dict[str, Any]] = None,
        train_mode: bool = True,
        *args,
        **kwargs,
    ):
        """Initializes the ArenaBaseEnv.

        Args:
            node (SupervisorNode): The ROS2 node instance.
            ns (Union[str, Namespace]): The namespace for ROS2 topics and services.
            space_manager (Union[BaseSpaceManager, Dict[str, Any]]): An instance or configuration
                dict for a class that defines the action and observation spaces.
            reward_function (Union[RewardFunction, Dict[str, Any]]): An instance or configuration
                dict for a class that calculates the reward.
            simulation_state_container (Optional[SimulationStateContainer]): A container for sharing
                state data across different components. Defaults to None.
            max_steps_per_episode (int): The maximum number of steps before an episode is
                truncated. Defaults to 100.
            init_by_call (bool): If True, ROS-dependent components are not initialized in the
                constructor but must be initialized by a manual call to `_initialize_environment()`.
                Defaults to False.
            wait_for_obs (bool): If True, the ObservationManager will wait for all observation
                sources to publish at least once before proceeding. Defaults to False.
            obs_unit_kwargs (Optional[Dict[str, Any]]): A dictionary of keyword arguments to be
                passed to the constructors of individual observation units. Defaults to None.
        """
        super().__init__()
        self.node = node
        self.ns = Namespace(ns) if isinstance(ns, str) else ns

        self._is_train_mode = train_mode
        if self.is_train_mode and reward_function is None:
            raise ValueError("A reward function is required for training mode.")

        self._initialize_agent_components(space_manager, reward_function)
        self.__simulation_state_container = simulation_state_container

        self._obs_unit_kwargs = obs_unit_kwargs or {}
        self.__wait_for_obs = wait_for_obs

        self._steps_curr_episode = 0
        self._episode = 0
        self._max_steps_per_episode = max_steps_per_episode
        self.__is_first_step = True

        # Synchronization mechanism for step() and ROS service callback
        self._action_condition = threading.Condition()
        self._pending_action: Optional[np.ndarray] = None
        self._action_is_available = False  # True when step() provides an action
        self._action_is_consumed = True  # True when service consumes the action

        if not init_by_call:
            self._initialize_environment()

    def _initialize_environment(self):
        """Initializes ROS-dependent components and the observation manager."""
        if self.node is None:
            # rclpy.init()  # Initialize ROS in worker process
            env_node_name = f"{self.ns.to_string()}_env".replace("/", "_")
            self.node = SupervisorNode(node_name=env_node_name)

        if self.is_train_mode:
            self._setup_ros_services()

        self._setup_observation_manager()

    def _setup_ros_services(self):
        """Creates ROS2 services and clients required for training."""
        task_srv_name = self.ns.simulation_ns("reset_task").to_string()
        self._reset_task_srv = self.node.create_client(
            EmptySrv,
            task_srv_name,
            callback_group=rclpy.callback_groups.MutuallyExclusiveCallbackGroup(),
        )

        if not self._reset_task_srv.wait_for_service(timeout_sec=3.0):
            self.node.get_logger().warn(
                f"Service '{task_srv_name}' not available after 3 seconds."
            )

        self._setup_action_service()

    def _setup_action_service(self):
        service_name = self.ns("get_command").to_string()
        self.node.get_logger().info(f"Creating get_command service at: {service_name}")
        self._get_command_srv = self.node.create_service(
            GetCommand,
            service_name,
            self._on_get_command_request,
            callback_group=rclpy.callback_groups.MutuallyExclusiveCallbackGroup(),
        )
        self.node.get_logger().info(f"Service {service_name} created successfully!")

    def _setup_observation_manager(self):
        """Configures and initializes the ObservationManager."""
        # TODO: Implement observation manager setup
        with open(
            "/home/le/arena4_ws_exp/src/planners/rosnav_rl/rosnav_rl/rosnav_rl/observations/observations.yaml",
            "r",
        ) as file:
            config = yaml.safe_load(file)

        # Create the observation manager from the configuration
        self.observation_collector = create_observation_manager_from_config(
            config=config,
            node=self.node,
            ns=self.ns.to_string(),
            simulation_state_container=self.simulation_state_container,
            wait_for_obs=self.__wait_for_obs,  # Wait for topics to be available
        )

    @property
    def action_space(self) -> gymnasium.spaces.Box:
        return self._model_space_manager.action_space

    @property
    def observation_space(self) -> gymnasium.spaces.Dict:
        return self._model_space_manager.observation_space

    @property
    def simulation_state_container(self) -> SimulationStateContainer:
        return self.__simulation_state_container

    @property
    def is_train_mode(self) -> bool:
        return self._is_train_mode

    def _initialize_agent_components(
        self,
        space_manager: Union[BaseSpaceManager, Dict[str, Any]],
        reward_function: Union[RewardFunction, Dict[str, Any]],
    ):
        """Initializes space manager and reward function from instances or dicts."""
        self._model_space_manager = (
            BaseSpaceManager(**space_manager)
            if isinstance(space_manager, dict)
            else space_manager
        )
        self._reward_function = (
            RewardFunction(**reward_function)
            if isinstance(reward_function, dict)
            else reward_function
        )

        assert isinstance(self._model_space_manager, BaseSpaceManager)
        assert isinstance(self._reward_function, RewardFunction)

    def _decode_action(self, action: np.ndarray) -> np.ndarray:
        """Decodes the given action using the model space encoder."""
        return self._model_space_manager.decode_action(action)

    def _encode_observation(
        self, observation: ObservationDict
    ) -> EncodedObservationDict:
        """Encodes the given observation using the model space encoder."""
        return self._model_space_manager.encode_observation(observation)

    def _wait_for_action_consumption(self, timeout: float = 20.0) -> None:
        """
        Waits for the `get_command` service to consume the action set by `step()`.

        This method is called from `step()` to synchronize with the ROS service.
        It waits until `_on_get_command_request` signals that it has taken the action.

        Args:
            timeout (float): Maximum time to wait in seconds.

        Raises:
            RuntimeError: If the timeout is exceeded, indicating a likely deadlock
                          or issue with the simulation controller.
        """
        with self._action_condition:
            # Wait until the service signals that it has consumed the action.
            # The `wait_for` method returns False on timeout.
            if not self._action_condition.wait_for(
                lambda: self._action_is_consumed, timeout=timeout
            ):
                self.node.get_logger().error(
                    f"Timeout waiting for action to be consumed after {timeout}s. "
                    "The simulation controller may not be requesting commands. "
                    f"Action available: {self._action_is_available}, "
                    f"Action consumed: {self._action_is_consumed}"
                )
                # Force reset the state to prevent permanent deadlock
                self._action_is_consumed = True
                self._action_is_available = False
                self._pending_action = None
                raise RuntimeError("Action consumption timeout")

    @flush_errors_decorator
    def step(
        self, action: np.ndarray
    ) -> Tuple[EncodedObservationDict, float, bool, bool, InformationDict]:
        """
        Processes a single step in the environment.

        This method performs the following key actions:
        1. Decodes the action provided by the RL agent.
        2. Makes the action available for the `get_command` ROS service.
        3. Notifies the service that an action is ready.
        4. Waits for the service to consume the action, ensuring synchronization.
        5. After the action is consumed, it retrieves the new observation from the simulation.
        6. Calculates the reward and determines if the episode has terminated.
        7. Encodes the observation and returns the standard Gymnasium step tuple.
        """
        # if self.__is_first_step:
        #     self._setup_action_service()

        decoded_action = self._decode_action(action)

        # Make the action available to the service and notify it.
        with self._action_condition:
            if not self._action_is_consumed:
                self.node.get_logger().warn(
                    "New action is being set, but previous one was not consumed. "
                    "This may indicate a synchronization issue."
                )
            self._pending_action = decoded_action
            self._action_is_available = True
            self._action_is_consumed = False
            # Notify the waiting service thread that an action is ready.
            self._action_condition.notify()

        # Wait for the simulation controller to request and consume the action.
        self._wait_for_action_consumption()

        # Once the action is consumed, proceed with the environment step.
        obs_dict = self.observation_collector.get_observations(
            simulation_state_container=self.__simulation_state_container,
            is_first=self.__is_first_step,
        )

        reward, reward_info = self._reward_function.get_reward(
            obs_dict=obs_dict,
            simulation_state_container=self.__simulation_state_container,
        )

        self._steps_curr_episode += 1
        info, done = determine_termination(
            reward_info=reward_info,
            curr_steps=self._steps_curr_episode,
            max_steps=self._max_steps_per_episode,
        )
        obs_dict["is_terminal"] = done
        self.__is_first_step = False

        return (
            self._encode_observation(obs_dict),
            reward,
            done,
            False,
            info,
        )

    def _on_get_command_request(
        self, request: GetCommand.Request, response: GetCommand.Response
    ) -> GetCommand.Response:
        """
        ROS2 service callback for receiving command requests from the simulation.

        This callback waits for an action to be made available by the `step()` method,
        responds to the service request with that action, and then signals to the
        `step()` method that the action has been consumed.

        Args:
            request: The service request (unused).
            response: The service response to be filled with the action.

        Returns:
            The service response containing the Twist command.
        """
        with self._action_condition:
            # Wait until an action is available from the step() method.
            # A timeout is included to prevent the service from hanging indefinitely.
            if not self._action_condition.wait_for(
                lambda: self._action_is_available, timeout=5.0
            ):
                self.node.get_logger().warn(
                    "[Service] Timeout waiting for a new action from the agent. "
                    "Responding with a zero-velocity command."
                )
                response.twist = Twist()
                return response

            # An action is available, so consume it.
            assert self._pending_action is not None
            response.twist = get_twist_from_action(self._pending_action)
            self.node.get_logger().debug(
                f"[Service] Responding with action: {self._pending_action}"
            )

            # Reset flags and notify the waiting step() method.
            self._pending_action = None
            self._action_is_available = False
            self._action_is_consumed = True
            self._action_condition.notify()

        return response

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[EncodedObservationDict, InformationDict]:
        """
        Resets the environment to its initial state and returns an initial observation.
        """
        super().reset(seed=seed)

        # Reset synchronization state
        with self._action_condition:
            self._pending_action = None
            self._action_is_available = False
            self._action_is_consumed = True
            self._action_condition.notify_all()  # Wake up any waiting threads

        # Reset episode-specific variables
        self._episode += 1
        self._steps_curr_episode = 0
        self.__is_first_step = True

        self.node.get_logger().info(
            f"[{self.ns.to_string()}] Resetting environment for episode {self._episode}..."
        )

        self._before_task_reset()
        self.reset_task()
        self._after_task_reset()

        self._reward_function.reset()

        # Get the initial observation after reset.
        obs_dict = self.observation_collector.get_observations(
            is_terminal=False, is_first=True
        )
        obs_dict["is_first"] = True  # Indicate it's the first observation

        info = {}
        return self._encode_observation(obs_dict), info

    def close(self):
        """Cleans up resources, like ROS2 services and subscribers."""
        self.node.get_logger().info(
            "Closing environment and shutting down ROS components."
        )

        self.observation_collector.shutdown()
        if getattr(self, "_get_command_srv", None):
            self._get_command_srv.destroy()
        if getattr(self, "_reset_task_srv", None):
            self._reset_task_srv.destroy()

    def reset_task(self):
        if not self._reset_task_srv or not self._reset_task_srv.service_is_ready():
            self.node.get_logger().warn("Reset task service client is not available.")
            return False

        completion_event = threading.Event()
        result_container = {"success": False, "exception": None}

        def done_callback(future):
            try:
                result = future.result()
                result_container["success"] = result is not None
            except Exception as e:
                result_container["exception"] = e
            finally:
                completion_event.set()

        future = self._reset_task_srv.call_async(EmptySrv.Request())
        future.add_done_callback(done_callback)

        # Wait with timeout
        if completion_event.wait(timeout=2.5):
            if result_container["success"]:
                self.node.get_logger().debug("Service call successful.")
            else:
                self.node.get_logger().error(
                    f"Service call failed: {result_container['exception']}"
                )
        else:
            self.node.get_logger().error("Service call timeout.")

    def _before_task_reset(self):
        """Hook for executing actions before the task is reset."""
        pass

    def _after_task_reset(self):
        """Hook for executing actions after the task is reset."""
        pass
