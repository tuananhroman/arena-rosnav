from typing import Any, Dict, Optional, Tuple, Type, Union

import gymnasium
import numpy as np
import rclpy
import rclpy.callback_groups as callback_groups
from geometry_msgs.msg import Twist
from rosnav_rl.observations import (
    DoneObservation,
    ObservationManager,
    get_required_observation_units,
)
from rosnav_rl.reward.reward_function import RewardFunction
from rosnav_rl.spaces import BaseSpaceManager
from rosnav_rl.states import SimulationStateContainer
from rosnav_rl.utils.rostopic import Namespace
from rosnav_rl.utils.type_aliases import EncodedObservationDict, ObservationDict
from std_srvs.srv import Empty as EmptySrv

from rl_utils.node import SupervisorNode
from rl_utils.utils.envs import determine_termination
from rl_utils.utils.type_alias.observation import InformationDict


class GazeboEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        node: SupervisorNode,
        ns: Union[str, Namespace],
        space_manager: Union[BaseSpaceManager, Dict[str, Any]],
        reward_function: Union[RewardFunction, Dict[str, Any]],
        simulation_state_container: Optional[SimulationStateContainer] = None,
        max_steps_per_episode=100,
        init_by_call: bool = False,
        wait_for_obs: bool = False,
        obs_unit_kwargs=None,
        task_generator_kwargs=None,
        *args,
        **kwargs,
    ):
        """
        Initialize the GazeboEnv environment.

        Args:
            node (SupervisorNode): The ROS2 supervisor node for communication and parameter management.
            ns (Union[str, Namespace]): The namespace for the environment, either as string or Namespace object.
            space_manager (Union[BaseSpaceManager, Dict[str, Any]]): Manager for observation and action spaces.
            reward_function (Union[RewardFunction, Dict[str, Any]]): Function or configuration for reward calculation.
            simulation_state_container (Optional[SimulationStateContainer], optional): Container for simulation state.
                Defaults to None.
            max_steps_per_episode (int, optional): Maximum number of steps allowed per episode. Defaults to 100.
            init_by_call (bool, optional): Whether to defer initialization until explicit call. Defaults to False.
            wait_for_obs (bool, optional): Whether to wait for observations during initialization. Defaults to False.
            obs_unit_kwargs (dict, optional): Additional keyword arguments for observation units. Defaults to None.
            task_generator_kwargs (dict, optional): Additional keyword arguments for task generator. Defaults to None.
            *args: Variable length argument list passed to parent class.
            **kwargs: Arbitrary keyword arguments passed to parent class.

        Raises:
            ValueError: If reward_function is None when in training mode.

        Note:
            The environment will automatically initialize unless init_by_call is set to True.
            Training mode is determined by the '/train_mode' ROS parameter.
        """
        super(GazeboEnv, self).__init__()
        self.ns = Namespace(ns) if type(ns) is str else ns
        self.node = node

        self._debug_mode = node.get_parameter_or("/debug_mode", False)
        self._is_train_mode = node.get_parameter_or("/train_mode", True)

        if self._is_train_mode and reward_function is None:
            raise ValueError("Reward function is required for the training.")

        self._initialize_agent_components(
            space_manager=space_manager,
            reward_function=reward_function,
        )
        self.__simulation_state_container = simulation_state_container

        self._obs_unit_kwargs = obs_unit_kwargs if obs_unit_kwargs else {}
        self._task_generator_kwargs = (
            task_generator_kwargs if task_generator_kwargs else {}
        )

        self.__wait_for_obs = wait_for_obs

        self._steps_curr_episode = 0
        self._episode = 0
        self._max_steps_per_episode = max_steps_per_episode
        self.__is_first = True

        if not init_by_call:
            self.init()

    def init(self):
        """
        Initializes the environment for training or evaluation.

        If the environment is in training mode, it sets up the environment accordingly.
        It then determines the required observation units based on the reward function
        and the observation space list. If a full range laser is attached to the robot,
        it adds the FullRangeLaserCollector to the observation units.

        Finally, it initializes the ObservationManager with the required observation units
        and other necessary parameters.

        Attributes:
            is_train_mode (bool): Indicates if the environment is in training mode.
            _setup_env_for_training (function): Sets up the environment for training.
            _reward_function (object): The reward function used in the environment.
            _model_space_manager (object): Manages the observation space list.
            __simulation_state_container (object): Contains the state of the simulation.
            _obs_unit_kwargs (dict): Additional keyword arguments for observation units.
            __wait_for_obs (bool): Indicates if the environment should wait for observations.
            ns (str): Namespace for the observation manager.
        """
        if self.is_train_mode:
            self._setup_env_for_training()

        # Determine the required observation units based on the reward function and observation space
        # If in training mode, include reward units; otherwise, only observation space list
        required_obs_units = get_required_observation_units(
            self._reward_function.reward_units
            + self._model_space_manager.observation_space_list
            if self.is_train_mode
            else self._model_space_manager.observation_space_list
        )

        # TODO: FullRangeLaser was used for collision detection in the past.
        # if self.__simulation_state_container.robot.laser_state.attach_full_range_laser:
        #     required_obs_units.append(FullRangeLaserCollector)

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
        """
        Returns the action space of the environment.

        Returns:
            action_space (object): The action space of the environment.
        """
        return self._model_space_manager.action_space

    @property
    def observation_space(self) -> gymnasium.spaces.Dict:
        """
        Returns the observation space of the environment.

        Returns:
            gym.Space: The observation space of the environment.
        """
        return self._model_space_manager.observation_space

    @property
    def simulation_state_container(self) -> SimulationStateContainer:
        return self.__simulation_state_container

    def _initialize_agent_components(
        self,
        space_manager: BaseSpaceManager,
        reward_function: RewardFunction,
    ):
        if isinstance(space_manager, BaseSpaceManager):
            self._model_space_manager = space_manager

        if isinstance(reward_function, RewardFunction):
            self._reward_function = reward_function

        if isinstance(space_manager, dict):
            self._model_space_manager = BaseSpaceManager(**space_manager)

        if isinstance(reward_function, dict):
            self._reward_function = RewardFunction(**reward_function)

        assert isinstance(self._model_space_manager, BaseSpaceManager)
        assert isinstance(self._reward_function, RewardFunction)

    def _setup_env_for_training(self):
        task_srv_name = str(self.ns.simulation_ns("reset_task"))
        self._reset_task_srv = self.node.create_client(EmptySrv, task_srv_name)

        while not self._reset_task_srv.wait_for_service(timeout_sec=3.0):
            self.node.get_logger().info(
                f"Waiting for service {task_srv_name} to be available..."
            )

        # agent action publisher
        self.agent_action_pub = self.node.create_publisher(
            Twist,
            str(self.ns("cmd_vel_raw")),
            1,
            callback_group=callback_groups.MutuallyExclusiveCallbackGroup(),
        )

    def _pub_action(self, action: np.ndarray):
        """
        Publishes the given action to the agent's action topic.

        Args:
            action (np.ndarray): The action to be published. It should be a 1D numpy array of length 3,
                                 representing the linear x, linear y, and angular z components of the action.

        Raises:
            AssertionError: If the length of the action array is not 3.
        """
        assert len(action) == 3

        action_msg = Twist()
        action_msg.linear.x = action[0]
        action_msg.linear.y = action[1]
        action_msg.angular.z = action[2]

        self.agent_action_pub.publish(action_msg)

    def _decode_action(self, action: np.ndarray) -> np.ndarray:
        """
        Decodes the given action using the model space encoder.

        Args:
            action (np.ndarray): The action to be decoded.

        Returns:
            np.ndarray: The decoded action.
        """
        return self._model_space_manager.decode_action(action)

    def _encode_observation(
        self, observation: ObservationDict, *args, **kwargs
    ) -> EncodedObservationDict:
        """
        Encodes the given observation using the model space encoder.

        Args:
            observation (ObservationDict): The observation to be encoded.

        Returns:
            The encoded observation.
        """
        return self._model_space_manager.encode_observation(observation, **kwargs)

    def step(
        self, action: np.ndarray
    ) -> Tuple[EncodedObservationDict, float, bool, bool, InformationDict]:
        """
        Execute one step in the environment using the given action.

        Args:
            action (np.ndarray): The action to be taken in the environment.

        Returns:
            Tuple[EncodedObservationDict, float, bool, bool, dict]: A tuple containing:
                - EncodedObservationDict: The encoded observation dictionary after applying the model space manager.
                - float: The reward obtained after taking the action.
                - bool: A flag indicating if the episode has ended.
                - bool: A flag indicating if the episode was truncated (always False in this implementation).
                - InformationDict: Additional information about the step.
        """
        self._pub_action(self._decode_action(action))

        obs_dict: ObservationDict = self.observation_collector.get_observations(
            simulation_state_container=self.__simulation_state_container,
            is_first=self.__is_first,
        )

        # calculate reward
        reward, reward_info = self._reward_function.get_reward(
            obs_dict=obs_dict,
            simulation_state_container=self.__simulation_state_container,
        )

        self._steps_curr_episode += 1

        # info
        info, done = determine_termination(
            reward_info=reward_info,
            curr_steps=self._steps_curr_episode,
            max_steps=self._max_steps_per_episode,
        )

        obs_dict.update({"is_terminal": done})
        self.__is_first = False

        return (
            self._encode_observation(obs_dict, is_done=done),
            reward,
            done,
            False,
            info,
        )

    def reset(
        self, seed=None, options=None
    ) -> Tuple[EncodedObservationDict, InformationDict]:
        """
        Resets the environment to an initial state and returns an initial observation.

        Args:
            seed (int, optional): The seed for random number generation. Defaults to None.
            options (dict, optional): Additional options for the reset. Defaults to None.

        Returns:
            Tuple[EncodedObservationDict, InformationDict]:
                A tuple containing the encoded observation dictionary and an information dictionary.
        """
        super().reset(seed=seed)
        self._episode += 1

        self._before_task_reset()

        self.reset_task()
        self._reward_function.reset()
        self._steps_curr_episode = 0

        self._after_task_reset()

        obs_dict: Dict[str, Any] = self.observation_collector.get_observations(
            is_terminal=False, is_first=True
        )

        obs_dict.update({DoneObservation.name: True})

        self.__is_first = True
        return (
            self._encode_observation(obs_dict),
            dict(),
        )

    def close(self):
        """
        Close the environment.

        """
        self.observation_collector.shutdown()
        self._reset_task_srv.destroy()

    def reset_task(self):
        """
        Resets the task in the environment by calling the reset service.

        This method is typically used to reset the environment to a new task or scenario.
        It ensures that the task is reset properly and prepares the environment for a new episode.

        """
        if self._reset_task_srv and self._reset_task_srv.service_is_ready():
            future = self._reset_task_srv.call_async(EmptySrv.Request())
            rclpy.spin_until_future_complete(
                self.node, future, timeout_sec=5.0
            )  # Added timeout
            if future.done():
                response = future.result()
                if response is None:
                    self.node.get_logger().error(
                        f"Service call to '{self._reset_task_srv.srv_name}' failed: {future.exception()}"
                    )
            else:
                self.node.get_logger().error(
                    f"Service call to '{self._reset_task_srv.srv_name}' timed out."
                )
        elif self._reset_task_srv:
            self.node.get_logger().warn(
                f"Service '{self._reset_task_srv.srv_name}' not ready."
            )

    def _before_task_reset(self):
        """
        Perform any necessary steps before resetting the task.

        """
        pass

    def _after_task_reset(self):
        """
        Perform any necessary steps after resetting the task.

        """
        pass

    @property
    def is_train_mode(self) -> bool:
        return self._is_train_mode
