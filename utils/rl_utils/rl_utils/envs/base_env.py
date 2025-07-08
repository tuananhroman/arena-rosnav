import subprocess
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rosnav_rl.observations import (
    DoneObservation,
    ObservationCollectorUnit,
    ObservationManager,
    get_required_observation_units,
)
from rosnav_rl.reward.reward_function import RewardFunction
from rosnav_rl.spaces import BaseSpaceManager
from rosnav_rl.states import SimulationStateContainer
from rosnav_rl.utils.rostopic import Namespace
from rosnav_rl.utils.type_aliases import EncodedObservationDict, ObservationDict
from rosnav_rl_msgs.srv import GetCommand
from std_srvs.srv import Empty as EmptySrv

from rl_utils.node import SupervisorNode
from rl_utils.utils.envs import determine_termination
from rl_utils.utils.type_alias.observation import InformationDict
from rclpy.executors import SingleThreadedExecutor

import time
import os


def get_twist_from_action(action: np.ndarray) -> Twist:
    """
    Converts an action array to a Twist message.

    Args:
        action (np.ndarray): The action array containing linear and angular velocities.

    Returns:
        Twist: A Twist message with the linear and angular velocities set.
    """
    twist = Twist()
    twist.linear.x = float(action[0])
    twist.linear.y = float(action[1])
    twist.linear.z = float(action[2])
    return twist


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
        node: SupervisorNode,
        ns: Union[str, Namespace],
        space_manager: Union[BaseSpaceManager, Dict[str, Any]],
        reward_function: Union[RewardFunction, Dict[str, Any]],
        simulation_state_container: Optional[SimulationStateContainer] = None,
        max_steps_per_episode: int = 100,
        init_by_call: bool = False,
        wait_for_obs: bool = False,
        obs_unit_kwargs: Optional[Dict[str, Any]] = None,
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

        self._is_train_mode = node.get_parameter_or("/train_mode", True)
        if self._is_train_mode and reward_function is None:
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
        self._service_request_event = threading.Event()
        self._action_lock = threading.Lock()
        self._pending_action: Optional[np.ndarray] = None
        self._action_consumed = True  # Track if action was consumed by service callback
        self._step_counter = 0  # For debugging
        self._service_active = (
            False  # Track if we're currently processing a service request
        )

        self._shutdown_event = threading.Event()
        self._spin_thread = threading.Thread(target=self._spin_loop)
        self._spin_thread.start()

        self._first_env_step = True

        if not init_by_call:
            self._initialize_environment()

    def _initialize_environment(self):
        """Initializes ROS-dependent components and the observation manager."""
        if self.is_train_mode:
            self._setup_ros_services()

        self._setup_observation_manager()

    def _spin_loop(self):
        """Continuously spins the ROS2 node in a background thread."""
        while not self._shutdown_event.is_set():
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def _setup_ros_services(self):
        """Creates ROS2 services and clients required for training."""
        # self._setup_action_service()
        task_srv_name = str(self.ns.simulation_ns("reset_task"))
        self._reset_task_srv = self.node.create_client(
            EmptySrv,
            task_srv_name,
            callback_group=rclpy.callback_groups.MutuallyExclusiveCallbackGroup(),
        )

        if not self._reset_task_srv.wait_for_service(timeout_sec=3.0):
            self.node.get_logger().warn(
                f"Service '{task_srv_name}' not available after 3 seconds."
            )

    def _setup_action_service(self):
        service_name = str(self.ns("get_command"))
        self.node.get_logger().info(f"Creating get_command service at: {service_name}")
        self._get_command_srv = self.node.create_service(
            GetCommand,
            service_name,
            self._on_get_command_request,
            callback_group=rclpy.callback_groups.MutuallyExclusiveCallbackGroup(),
        )

    def _setup_observation_manager(self):
        """Configures and initializes the ObservationManager."""
        obs_list = self._model_space_manager.observation_space_list
        if self.is_train_mode:
            obs_list += self._reward_function.reward_units

        required_obs_units: List[Type[ObservationCollectorUnit]] = (
            get_required_observation_units(obs_list)
        )

        self.observation_collector = ObservationManager(
            node=self.node,
            ns=self.ns,
            obs_structure=required_obs_units,
            simulation_state_container=self.__simulation_state_container,
            obs_unit_kwargs=self._obs_unit_kwargs,
            wait_for_obs=self.__wait_for_obs,
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
        self, observation: ObservationDict, *args, **kwargs
    ) -> EncodedObservationDict:
        """Encodes the given observation using the model space encoder."""
        return self._model_space_manager.encode_observation(observation, **kwargs)

    def _populate_action(self, action: np.ndarray):
        """Stores the action and waits for the service request to proceed."""
        self._step_counter += 1
        step_id = self._step_counter

        # Check if previous action was consumed
        with self._action_lock:
            if not self._action_consumed:
                self.node.get_logger().warn(
                    f"[Step {step_id}] Previous action was not consumed by controller service - "
                    "this may indicate synchronization issues"
                )

            # Store the new action for the service callback to use
            self._pending_action = action
            self._action_consumed = False  # Mark as not yet consumed
            self._service_active = True  # Mark that we're expecting a service request
            self.node.get_logger().info(
                f"[Step {step_id}] Action stored, waiting for service request..."
            )

        # Wait for the service callback to signal that a request is pending
        timeout_sec = 30.0
        start_time = self.node.get_clock().now()

        while True:
            if self._service_request_event.wait(timeout=2.0):
                self.node.get_logger().info(
                    f"[Step {step_id}] Service request received, action sent to controller"
                )
                break

            current_time = self.node.get_clock().now()
            elapsed = (current_time - start_time).nanoseconds / 1e9

            if elapsed > timeout_sec:
                with self._action_lock:
                    self._service_active = False

                self.node.get_logger().error(
                    f"[Step {step_id}] Timeout waiting for service request after {elapsed:.1f}s"
                )
                raise RuntimeError(
                    "Service request timeout - controller may not be running or has timed out"
                )

            self.node.get_logger().warn(
                f"[Step {step_id}] Still waiting for service request... ({elapsed:.1f}s elapsed)"
            )

    def step(
        self, action: np.ndarray
    ) -> Tuple[EncodedObservationDict, float, bool, bool, InformationDict]:
        """
        Processes a single step in the environment.

        This method stores the action, waits for a service request, responds with the action,
        then continues with environment step logic while PPO calculates the next action.
        """
        if self.__is_first_step:
            self._setup_action_service()

        # Clear the event ONLY if no service is currently being processed
        # This prevents clearing an event that was just set by an incoming service request
        with self._action_lock:
            if not self._service_active:
                self._service_request_event.clear()

        self._populate_action(self._decode_action(action))

        # Now proceed with the environment step logic
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
            self._encode_observation(obs_dict, is_done=done),
            reward,
            done,
            False,
            info,
        )

    def _on_get_command_request(
        self, request: GetCommand.Request, response: GetCommand.Response
    ) -> GetCommand.Response:
        """
        ROS2 service callback for receiving command requests.

        This callback immediately responds with the pending action and signals
        the step method to continue with environment logic.
        """
        self.node.get_logger().info(
            f"[Service] Request received (step {self._step_counter})"
        )

        # Check if we're actually expecting a service request
        with self._action_lock:
            if not self._service_active:
                self.node.get_logger().warn(
                    "[Service] Received unexpected service request - no step is waiting"
                )
                # Still respond with zero action to prevent controller hanging
                response.twist = Twist()
                return response

            # Fill response with the pending action (thread-safe access)
            if self._pending_action is not None:
                twist = get_twist_from_action(self._pending_action)
                response.twist = twist
                self.node.get_logger().info(
                    f"[Service] Responding with action: {self._pending_action}"
                )
                self._action_consumed = True  # Mark action as consumed
                self._service_active = False  # Mark as no longer waiting for service
            else:
                self.node.get_logger().warn(
                    "[Service] No pending action available, sending zero action"
                )
                response.twist = Twist()

        # Signal the step method that the service request has been handled
        self._service_request_event.set()
        self.node.get_logger().info("[Service] Event set, returning response")

        return response

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[EncodedObservationDict, InformationDict]:
        """
        Resets the environment to its initial state and returns an initial observation.
        """
        # Superclass call (recommended by gymnasium)
        super().reset(seed=seed)

        if getattr(self, "_get_command_srv", None):
            self._get_command_srv.destroy()

        # Clear the service request event to prevent post-reset deadlocks
        self._service_request_event.clear()

        # Reset episode-specific variables
        self._episode += 1
        self._steps_curr_episode = 0

        self.node.get_logger().info("Resetting environment...")

        self._before_task_reset()

        self.reset_task()
        self._reward_function.reset()
        self._steps_curr_episode = 0

        self._after_task_reset()

        obs_dict = self.observation_collector.get_observations(
            is_terminal=False, is_first=True
        )
        obs_dict[DoneObservation.name] = True
        self.__is_first_step = True

        # Reset action consumption tracking for new episode
        with self._action_lock:
            self._action_consumed = True
            self._step_counter = 0  # Reset step counter
            self._service_active = False  # Ensure service is marked as inactive
        return {}
        # return self._encode_observation(obs_dict), {}

    def close(self):
        """Cleans up resources, like ROS2 services and subscribers."""
        self._shutdown_event.set()
        if self._spin_thread and self._spin_thread.is_alive():
            self._spin_thread.join()

        self.observation_collector.shutdown()
        if self._get_command_srv:
            self._get_command_srv.destroy()
        if self._reset_task_srv:
            self._reset_task_srv.destroy()

    def reset_task(self):
        """
        Calls the task reset service in a separate thread to avoid deadlocking
        the main ROS2 executor.
        """
        if not self._reset_task_srv or not self._reset_task_srv.service_is_ready():
            self.node.get_logger().warn("Reset task service client is not available.")
            return False

        future = self._reset_task_srv.call_async(EmptySrv.Request())
        try:
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=5.0)
            if future.result() is not None:
                self.node.get_logger().debug(
                    f"Service call to '{self._reset_task_srv.srv_name}' was successful."
                )
            else:
                self.node.get_logger().error(
                    f"Service call to '{self._reset_task_srv.srv_name}' failed: {future.exception()}"
                )
        except Exception as e:
            self.node.get_logger().error(
                f"Exception while calling '{self._reset_task_srv.srv_name}': {e}"
            )

    def _before_task_reset(self):
        """Hook for executing actions before the task is reset."""
        pass

    def _after_task_reset(self):
        """Hook for executing actions after the task is reset."""
        pass
