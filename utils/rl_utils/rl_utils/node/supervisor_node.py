import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import threading

from rl_utils.cfg import TrainingCfg


class SupervisorNode(Node):
    # This node is responsible for supervising the training process in a ROS2 environment.
    # stores training parameters and configurations.
    # subscribes to relevant topics and manages the training lifecycle.
    # communicates with other nodes to coordinate training tasks (task reset, curriculum management, etc.).

    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.get_logger().info(f"{node_name} has been started.")
        self._shutdown_event = threading.Event()

        # Use MultiThreadedExecutor to handle callback groups properly
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(target=self._spin_loop)

    def start_spinning(self):
        """Starts the spin loop in a background thread."""
        if not self._spin_thread.is_alive():
            self._shutdown_event.clear()
            self._spin_thread.start()
            self.get_logger().info(
                "SupervisorNode spinning started with MultiThreadedExecutor."
            )

    def stop_spinning(self):
        """Stops the spin loop."""
        if self._spin_thread.is_alive():
            self._shutdown_event.set()
            self._spin_thread.join()
            self.get_logger().info("SupervisorNode spinning stopped.")

    def _spin_loop(self):
        """Continuously spins the ROS2 node in a background thread with MultiThreadedExecutor."""
        while not self._shutdown_event.is_set():
            # Use the executor to handle all callback groups
            self._executor.spin_once(timeout_sec=0.1)

    def destroy_node(self):
        self.stop_spinning()
        # Clean up executor
        if hasattr(self, "_executor"):
            self._executor.remove_node(self)
            self._executor.shutdown()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SupervisorNode("basic_node")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
