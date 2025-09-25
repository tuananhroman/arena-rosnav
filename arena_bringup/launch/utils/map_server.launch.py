import launch
import launch_ros.actions
from arena_bringup.substitutions import LaunchArgument


def generate_launch_description():
    ld_items = []
    LaunchArgument.auto_append(ld_items)

    # Add map_frame parameter to namespace the map frame
    map_frame = LaunchArgument(
        name="map_frame",
        default_value="map",
        description="Map frame id - should be namespaced for multi-environment setups",
    )

    ld = launch.LaunchDescription(
        [
            *ld_items,
            launch_ros.actions.Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_map_server",
                parameters=[
                    {
                        "node_names": ["map_server"],
                        "autostart": True,
                        "use_sim_time": True,
                        "bond_timeout": 0.0,
                    }
                ],
            ),
            launch_ros.actions.Node(
                package="nav2_map_server",
                executable="map_server",
                name="map_server",
                parameters=[
                    {
                        "topic_name": "map",
                        "frame_id": map_frame.substitution,
                        "yaml_filename": "",
                        "use_sim_time": True,
                        "save_map_timeout": 5.0,
                        "free_thresh_default": 0.25,
                        "occupied_thresh_default": 0.65,
                    }
                ],
            ),
        ]
    )

    return ld


if __name__ == "__main__":
    generate_launch_description()
