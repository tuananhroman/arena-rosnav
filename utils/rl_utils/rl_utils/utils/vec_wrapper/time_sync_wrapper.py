import gymnasium as gym
import rclpy
from rclpy.node import Node
from rclpy.time import Time  # Import Time for type hinting


class TimeSyncWrapper(gym.Wrapper):
    def __init__(self, env, node: Node, control_hz: float = 20.0):
        """
        A Gym Wrapper to synchronize step calls to a specific control frequency using ROS 2 time.

        Args:
            env: The Gym environment to wrap.
            node: A rclpy.node.Node instance to access ROS clock and for spinning.
            control_hz: The desired control frequency in Hz.
        """
        super().__init__(env)
        if not isinstance(node, Node):
            raise ValueError("A valid rclpy.node.Node must be provided.")
        self.node = node
        self.clock = node.get_clock()

        if control_hz <= 0:
            raise ValueError("control_hz must be positive.")
        # Store interval in nanoseconds for precise comparison with rclpy.time.Time objects
        self.control_interval_nanosec = int(1e9 / control_hz)

        # Time when the last env.step() was allowed to initiate.
        # Initialized to current time, so the first step call will also adhere to the interval logic.
        self.last_step_initiation_time: Time = self.clock.now()

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

        # Wait if the time elapsed since the last step initiation is less than the control interval.
        # The loop continues as long as the duration since the last step is less than our target interval.
        # Time differences result in rclpy.duration.Duration, which has a nanoseconds attribute.
        while (
            current_time - self.last_step_initiation_time
        ).nanoseconds < self.control_interval_nanosec:
            # Spin the node for a very short duration to allow ROS callbacks
            # and to act as a small sleep for this polling loop.
            # A very small positive timeout is needed for spin_once to not block indefinitely if there are no events.
            rclpy.spin_once(self.node, timeout_sec=0.0001)  # Spin for 0.1ms
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

    # Note: If your environment or the underlying ROS node needs specific shutdown,
    # you might want to add a close() method here.
    # The lifecycle of the passed 'node' is assumed to be managed externally.
