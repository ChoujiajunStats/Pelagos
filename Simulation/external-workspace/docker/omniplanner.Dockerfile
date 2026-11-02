# Dedicated ROS 1 backend; the Stonefish ROS 2 image is independent.
ARG OMNI_BASE=ros:noetic-ros-base-focal
FROM ${OMNI_BASE}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
COPY ros.key /tmp/ros.key
# ROS final snapshots are unavailable on this network. Keep signature validation.
RUN gpg --dearmor < /tmp/ros.key > /usr/share/keyrings/ros-archive-keyring.gpg \
    && rm -f /etc/apt/sources.list.d/ros*.list \
    && printf 'deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] https://mirrors.tuna.tsinghua.edu.cn/ros/ubuntu focal main\n' > /etc/apt/sources.list.d/ros.list \
    && sed -i -e 's|http://archive.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' \
              -e 's|http://security.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git python3-catkin-tools python3-numpy python3-yaml \
    libeigen3-dev libgoogle-glog-dev libgflags-dev libprotobuf-dev protobuf-compiler \
    libboost-all-dev libopencv-dev libpcl-dev libsqlite3-dev libzmq3-dev \
    ros-noetic-cv-bridge ros-noetic-pcl-ros ros-noetic-pcl-conversions \
    ros-noetic-tf ros-noetic-tf2-ros ros-noetic-interactive-markers \
    ros-noetic-grid-map-core ros-noetic-grid-map-filters ros-noetic-grid-map-msgs \
    ros-noetic-grid-map-ros ros-noetic-eigen-conversions ros-noetic-tf-conversions ros-noetic-cmake-modules \
    && rm -rf /var/lib/apt/lists/*
COPY src /opt/omniplanner_ws/src
COPY source-lock.omniplanner.json /opt/omniplanner/source-lock.json
COPY integration/patch_sources.py /tmp/patch_sources.py
RUN python3 /tmp/patch_sources.py
RUN touch /opt/omniplanner_ws/src/gbplanner_ros/gbplanner_ui/CATKIN_IGNORE \
    /opt/omniplanner_ws/src/gbplanner_ros/planner_gazebo_sim/CATKIN_IGNORE \
    /opt/omniplanner_ws/src/gbplanner_ros/ugv_simulator/CATKIN_IGNORE \
    /opt/omniplanner_ws/src/gbplanner_ros/planner_control_interface/CATKIN_IGNORE \
    /opt/omniplanner_ws/src/voxblox/voxblox_rviz_plugin/CATKIN_IGNORE
WORKDIR /opt/omniplanner_ws
RUN source /opt/ros/noetic/setup.bash \
    && catkin config --extend /opt/ros/noetic --cmake-args \
       -DCMAKE_BUILD_TYPE=Release -DCATKIN_ENABLE_TESTING=OFF -DBUILD_TESTING=OFF \
       -DBTCPP_BUILD_TESTS=OFF -DBTCPP_EXAMPLES=OFF -DBTCPP_GROOT_INTERFACE=OFF \
       -DBTCPP_SQLITE_LOGGING=OFF -DUSE_SYSTEM_EIGEN=ON -DUSE_SYSTEM_GLOG=ON \
       -DUSE_SYSTEM_PROTOBUF=ON \
    && PYTHONUNBUFFERED=1 catkin build gbplanner --no-status --summarize --jobs 4 --parallel-packages 2
COPY integration/validated_search.cpp /opt/omniplanner_ws/src/gbplanner_ros/gbplanner/src/validated_search.cpp
RUN printf '\ncs_add_executable(uw_omniplanner_node src/validated_search.cpp)\ntarget_link_libraries(uw_omniplanner_node ${PROJECT_NAME} ${planner_common_LIBRARIES})\n' >> /opt/omniplanner_ws/src/gbplanner_ros/gbplanner/CMakeLists.txt \
    && source /opt/ros/noetic/setup.bash \
    && PYTHONUNBUFFERED=1 catkin build gbplanner --no-status --summarize --jobs 4 --parallel-packages 2
COPY integration /opt/omniplanner/integration
RUN chmod +x /opt/omniplanner/integration/entrypoint.sh \
    && dpkg-query -W > /opt/omniplanner/dpkg-packages.txt
ENTRYPOINT ["/opt/omniplanner/integration/entrypoint.sh"]
