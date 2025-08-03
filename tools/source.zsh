#! /usr/bin/env zsh

trap "trap - ERR; return \$?" ERR

if [ -z ${ARENA_SOURCED+x} ] ; then
    export ARENA_WS_DIR="$(pwd)"
    export ARENA_ROS_DISTRO=${ARENA_ROS_DISTRO:-humble}

    # Set Gazebo version if not provided
    export GAZEBO_VERSION=${GAZEBO_VERSION:-harmonic}
    export GZ_VERSION=${GAZEBO_VERSION}

    export FASTRTPS_DEFAULT_PROFILES_FILE=~/.ros/fastdds.xml
    export ROS_DOMAIN_ID=1
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/ros/${ARENA_ROS_DISTRO}/lib/"
    export INSTALLED=$ARENA_WS_DIR/src/arena/arena-rosnav/.installed
    
    # stop rviz from flashing
    export QT_SCREEN_SCALE_FACTORS=1

    pushd src/arena/arena-rosnav > /dev/null
        export VIRTUAL_ENV_DISABLE_PROMPT=1
        venv_path="$(poetry env info -p)"
        source "$venv_path/bin/activate"
        
        dirs=("$venv_path"/lib/*/site-packages)
        export PYTHONPATH="${(j/:/)dirs}:$PYTHONPATH"
        
        unset dirs venv_path
    popd > /dev/null

    r2st() { ros2 service call "$1" "$(ros2 service type "$1")"; }

    export ARENA_SOURCED=1
    export PS1="(arena) $PS1"
    echo 'sourced arena environment'
fi

# ROS setup (use .zsh instead of .bash)
if [ -f "/opt/ros/${ARENA_ROS_DISTRO}/setup.zsh" ] ; then
    source "/opt/ros/${ARENA_ROS_DISTRO}/setup.zsh"
fi
if [ -f install/local_setup.zsh ] ; then
    source install/local_setup.zsh
fi 

# Isaac setup (use .zsh instead of .bash)
if [ -f "$INSTALLED" ] && grep -q "isaac.sh" "$INSTALLED"; then
    if [ -z ${ISAAC_PATH+x} ] ; then
        source "$HOME/isaacsim-4.2.0/setup.zsh"
        source "$HOME/isaacsim-4.2.0/setup.zsh"
    fi
fi

trap - ERR
