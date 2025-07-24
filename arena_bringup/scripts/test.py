import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException


class SingleTransformListener(Node):
    """!
    @brief A node to look up and print a single TF2 transform and then exit.
    @details This class initializes the necessary TF2 buffer and listener.
             The actual lookup logic is handled outside the class in the main
             function to facilitate a clean, single-shot execution.
    """

    def __init__(self):
        super().__init__('minimal_tf2_single_shot_listener')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)


def main(args=None):
    rclpy.init(args=args)
    node = SingleTransformListener()

    source_frame = 'map'
    target_frame = 'jackal/base_link'
    target_frame = 'jackal/odom'

    node.get_logger().info(
        f"Waiting for transform from '{source_frame}' to '{target_frame}'..."
    )

    try:
        # Wait for and get the transform
        # The timeout will wait for the specified duration for the transform to become available.
        t = node.tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rclpy.time.Time(seconds=0, nanoseconds=0),
            timeout=Duration(seconds=3.0))

        trans = t.transform.translation
        rot = t.transform.rotation
        node.get_logger().info(
            f'Successfully got transform:\n'
            f'Translation: [x: {trans.x:.3f}, y: {trans.y:.3f}, z: {trans.z:.3f}]\n'
            f'Rotation: [x: {rot.x:.3f}, y: {rot.y:.3f}, z: {rot.z:.3f}, w: {rot.w:.3f}]'
        )

    except TransformException as ex:

        node.get_logger().error(
            f'Could not get transform after 3 seconds: {ex}')
    finally:
        # Cleanly shutdown the node and rclpy
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
