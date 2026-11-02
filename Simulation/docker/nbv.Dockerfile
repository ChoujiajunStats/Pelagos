# Reuse the installed ROS1 toolchain and dependencies, but build only original NBV.
ARG NBV_BASE=stonefish-planner:omniplanner
FROM ${NBV_BASE}
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ros-noetic-octomap ros-noetic-octomap-msgs ros-noetic-octomap-ros ros-noetic-image-geometry \
    libgflags-dev libgoogle-glog-dev \
    && rm -rf /var/lib/apt/lists/*
COPY sources/ /opt/pelagos_nbv_ws/src/
COPY source-lock.json /opt/pelagos_nbv/source-lock.json
COPY integration/patch_sources.py /tmp/pelagos-nbv-patch-sources.py
COPY integration/forward_selection.hpp /tmp/pelagos-forward-selection.hpp
RUN python3 /tmp/pelagos-nbv-patch-sources.py \
    && source /opt/ros/noetic/setup.bash \
    && cd /opt/pelagos_nbv_ws \
    && catkin config --extend /opt/ros/noetic --cmake-args \
       -DCMAKE_BUILD_TYPE=Release -DCATKIN_ENABLE_TESTING=OFF -DBUILD_TESTING=OFF \
       -DUSE_SYSTEM_EIGEN=ON -DUSE_SYSTEM_GLOG=ON \
    && catkin build nbvplanner --no-status --summarize --interleave-output --jobs 2 --parallel-packages 1
COPY integration/ /opt/pelagos_nbv_ws/src/pelagos_nbv/
RUN source /opt/ros/noetic/setup.bash \
    && cd /opt/pelagos_nbv_ws \
    && catkin build pelagos_nbv --no-status --summarize --interleave-output --jobs 2 --parallel-packages 1
ENTRYPOINT ["/bin/bash"]
