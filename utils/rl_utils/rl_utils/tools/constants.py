from ament_index_python.packages import get_package_share_directory


class ROS_PACKAGES:
    ARENA_BRINGUP = get_package_share_directory("arena_bringup")


class TRAINING_CONSTANTS:
    class PATHS:
        TRAINING_CONFIGS = (
            lambda config_name: f"{ROS_PACKAGES.ARENA_BRINGUP}/configs/training/{config_name}.yaml"
        )
