import gymnasium as gym
import time
from rclpy.node import Node
from rclpy.time import Time  # Import Time for type hinting


class TimeSyncWrapper(gym.Wrapper):
    def __init__(self, env, control_hz: float = 10.0, warning_slop: float = 0.1):
        """
        A Gym Wrapper to synchronize step calls to a specific control frequency using ROS 2 time.

        Args:
            env: The Gym environment to wrap.
            control_hz: The desired control frequency in Hz.
            warning_slop: A factor to allow for a small deviation in control frequency before issuing a warning.
                          E.g., a value of 0.1 means that a warning will be issued if the actual frequency
                          is off by more than 10% of the desired interval.
        """
        super().__init__(env)
        if not isinstance(env.node, Node):
            raise ValueError("A valid rclpy.node.Node must be provided.")
        self.node = env.node
        self.clock = self.node.get_clock()

        if control_hz <= 0:
            raise ValueError("control_hz must be positive.")
        # Store interval in nanoseconds for precise comparison with rclpy.time.Time objects
        self.control_interval_nanosec = int(1e9 / control_hz)
        self.warning_slop_nanosec = int(self.control_interval_nanosec * warning_slop)

        # Time when the last env.step() was allowed to initiate.
        # Initialized to current time, so the first step call will also adhere to the interval logic.
        self.last_step_initiation_time: Time = self.clock.now()

    def _initialize_environment(self):
        """
        Initializes the environment if it has an _initialize_environment method.
        This is useful for environments that require some setup before use.
        """
        if hasattr(self.env, "_initialize_environment"):
            return self.env._initialize_environment()
        else:
            self.node.get_logger().warn(
                "The wrapped environment does not have an _initialize_environment method."
            )
            return None

    def _now(self) -> Time:
        """Returns the current ROS time as an rclpy.time.Time object."""
        return self.clock.now()

    def step(self, action):
        """
        Executes a step in the environment, ensuring the control frequency is respected.
        If called too frequently, this method will block (while spinning the node)
        until the control interval has passed since the last step initiation.
        """
        current_time = self._now()
        elapsed_nanosec = (current_time - self.last_step_initiation_time).nanoseconds

        # Warn if the actual interval is longer than the desired one, indicating a missed control frequency.
        # We check this before the waiting loop.
        if elapsed_nanosec > self.control_interval_nanosec + self.warning_slop_nanosec:
            desired_hz = 1e9 / self.control_interval_nanosec
            actual_hz = 1e9 / elapsed_nanosec
            self.node.get_logger().warn(
                f"Control frequency missed! "
                f"Desired: {desired_hz:.2f}Hz, Actual: {actual_hz:.2f}Hz"
            )

        # Wait if the time elapsed since the last step initiation is less than the control interval.
        # The loop continues as long as the duration since the last step is less than our target interval.
        # Time differences result in rclpy.duration.Duration, which has a nanoseconds attribute.
        while (
            current_time - self.last_step_initiation_time
        ).nanoseconds < self.control_interval_nanosec:
            # Sleep for a very short duration.
            # The SupervisorNode handles spinning in a background thread.
            time.sleep(0.0001)  # Sleep for 0.1ms
            current_time = self._now()

        # Update the initiation time for the current step (which is now allowed to proceed)
        # It's important to take a fresh timestamp here, after the wait loop.
        self.last_step_initiation_time = self._now()

        return self.env.step(action)

    def reset(self, **kwargs):
        """
        Resets the environment. Also resets the step timing mechanism for the first step
        after reset, similar to __init__.
        """
        # Reset last_step_initiation_time so the first step after reset
        # doesn't try to align with the time before the reset call.
        # It will enforce its interval relative to the actual time of reset completion.
        reset_return_value = self.env.reset(**kwargs)
        self.last_step_initiation_time = self._now()
        return reset_return_value
