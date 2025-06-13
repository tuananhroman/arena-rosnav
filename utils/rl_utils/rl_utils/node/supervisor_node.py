import rclpy
from rclpy.node import Node

from rl_utils.cfg import TrainingCfg


class SupervisorNode(Node):
    # This node is responsible for supervising the training process in a ROS2 environment.
    # stores training parameters and configurations.
    # subscribes to relevant topics and manages the training lifecycle.
    # communicates with other nodes to coordinate training tasks (task reset, curriculum management, etc.).

    def __init__(self, node_name: str, training_cfg: TrainingCfg):
        super().__init__(node_name)
        self.training_cfg = training_cfg
        self.get_logger().info(f"{node_name} has been started.")

    ...


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
