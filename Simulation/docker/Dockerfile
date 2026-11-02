ARG BASE_IMAGE=surfslam:stereo-v2
FROM ${BASE_IMAGE}
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# The base is the locally built Stonefish/Jazzy image with its existing CUDA
# PyTorch environment. Only the missing OKVIS build dependencies are added.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libeigen3-dev libgoogle-glog-dev libgflags-dev libatlas-base-dev \
    libboost-all-dev libsuitesparse-dev libpcl-dev libopencv-dev \
    ros-jazzy-cv-bridge ros-jazzy-pcl-msgs ros-jazzy-message-filters \
    ros-jazzy-rosbag2-cpp ros-jazzy-rosidl-default-generators \
    && rm -rf /var/lib/apt/lists/*

COPY patches/stonefish-depth-capture-time.patch /tmp/stonefish-depth.patch
COPY patches/stonefish-ros-depth-capture-time.patch /tmp/stonefish-ros-depth.patch
# Retain the existing renderer and sensor; carry its actual render timestamp
# through the depth readback, as the installed color-camera patch already does.
RUN git -C /opt/uw_src/stonefish apply --check /tmp/stonefish-depth.patch \
    && git -C /opt/uw_src/stonefish apply /tmp/stonefish-depth.patch \
    && git -C /opt/uw_src/stonefish_ros2 apply --check /tmp/stonefish-ros-depth.patch \
    && git -C /opt/uw_src/stonefish_ros2 apply /tmp/stonefish-ros-depth.patch \
    && cmake --build /opt/uw_build/stonefish --parallel 2 \
    && cmake --install /opt/uw_build/stonefish && ldconfig \
    && source /opt/ros/jazzy/setup.bash && source /opt/uw_underlay/setup.bash \
    && cmake --build /opt/uw_build/ros/stonefish_ros2 --parallel 2 \
    && cmake --install /opt/uw_build/ros/stonefish_ros2

COPY okvis2/ /tmp/pelagos-okvis2/
COPY patches/okvis-realtime-ingress.patch patches/okvis-realtime-ingress-sha256.json patches/apply-okvis-realtime-ingress.py /tmp/okvis-ingress-patch/
# Patch only the copied build tree; rebuild all ROS clients with the mutex layout.
RUN /usr/bin/python3 /tmp/okvis-ingress-patch/apply-okvis-realtime-ingress.py /tmp/pelagos-okvis2 \
    && source /opt/ros/jazzy/setup.bash \
    && cmake -S /tmp/pelagos-okvis2 -B /tmp/pelagos-okvis-build \
       -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/pelagos/okvis2 \
       -DPython3_EXECUTABLE=/usr/bin/python3 \
       -DBUILD_ROS2=ON -DBUILD_APPS=OFF -DBUILD_TESTS=OFF \
       -DUSE_NN=OFF -DHAVE_LIBREALSENSE=OFF -DBUILD_SHARED_LIBS=OFF \
    && cmake --build /tmp/pelagos-okvis-build --parallel 2 \
    && cmake --install /tmp/pelagos-okvis-build \
    && cp /tmp/okvis-ingress-patch/okvis-realtime-ingress-sha256.json /opt/pelagos/okvis2/share/okvis/ \
    && rm -rf /tmp/pelagos-okvis2 /tmp/pelagos-okvis-build

COPY uw_interfaces/ /tmp/uw_interfaces/
# ROS interface generators need the system Python's ROS/EmPy packages, not
# the CUDA research environment selected by the base image's PATH.
RUN source /opt/ros/jazzy/setup.bash \
    && /usr/bin/python3 -c 'import em; import rosidl_adapter' \
    && /usr/bin/python3 /usr/bin/colcon --log-base /tmp/pelagos-colcon-log build --base-paths /tmp/uw_interfaces \
       --build-base /tmp/pelagos-interfaces-build --install-base /opt/pelagos/interfaces \
       --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 -DPython_EXECUTABLE=/usr/bin/python3 \
    && rm -rf /tmp/uw_interfaces /tmp/pelagos-interfaces-build /tmp/pelagos-colcon-log

# Reuse the timm version used by the local LAS2 evaluation runtime. The
# inherited SurfSLAM timm predates the fasternet_t0 model required by LAS2.
COPY las2_core/ /opt/pelagos/las2/core/
RUN /opt/research/bin/python3 -m pip install --no-cache-dir --no-deps timm==1.0.30 \
    && PYTHONPATH=/opt/pelagos/las2 /opt/research/bin/python3 -c \
       'import timm; assert timm.__version__ == "1.0.30"; from core.models import build_model; build_model("las2", model_size="s", fnet_pretrained=False)'

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1

# Reuse Pelagos's existing ROS2 adapter and the local upstream Wavemap C++
# library. This builds no planner, ROS1 bridge, test suite or GPU mapper.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    liboctomap-dev ros-jazzy-octomap-msgs && rm -rf /var/lib/apt/lists/*
COPY wavemap/ /tmp/pelagos-wavemap/
COPY underwater_wavemap_ros/ /tmp/underwater_wavemap_ros/
RUN source /opt/ros/jazzy/setup.bash \
    && export CMAKE_BUILD_PARALLEL_LEVEL=2 MAKEFLAGS=-j2 \
    && /usr/bin/python3 /usr/bin/colcon --log-base /tmp/wavemap-colcon-log build \
       --base-paths /tmp/underwater_wavemap_ros --build-base /tmp/wavemap-build \
       --install-base /opt/pelagos/mapping --parallel-workers 1 \
       --event-handlers console_direct+ \
       --cmake-args -DCMAKE_BUILD_TYPE=Release \
       -DWAVEMAP_SOURCE_DIR=/tmp/pelagos-wavemap \
       -DPython3_EXECUTABLE=/usr/bin/python3 -DPython_EXECUTABLE=/usr/bin/python3 \
       -DENABLE_TESTING=OFF -DCATKIN_ENABLE_TESTING=OFF -DBUILD_TESTING=OFF \
       -DUSE_SYSTEM_EIGEN=ON -DUSE_SYSTEM_GLOG=ON -DUSE_SYSTEM_BOOST=ON \
    && mkdir -p /opt/pelagos/mapping/share/wavemap \
    && cp /tmp/pelagos-wavemap/LICENSE /opt/pelagos/mapping/share/wavemap/LICENSE \
    && rm -rf /tmp/pelagos-wavemap /tmp/underwater_wavemap_ros /tmp/wavemap-build /tmp/wavemap-colcon-log

# Independent final layer: the mission panel does not rebuild VIO/depth/map.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3-pyqt5 fonts-wqy-microhei && rm -rf /var/lib/apt/lists/*

# ROS2 port of the GBPlanner-style control panel; independent of SLAM/map layers.
COPY rviz_panel/ /tmp/pelagos-rviz-panel/
RUN source /opt/ros/jazzy/setup.bash \
    && cmake -S /tmp/pelagos-rviz-panel -B /tmp/pelagos-rviz-build \
       -DCMAKE_INSTALL_PREFIX=/opt/pelagos/visualization \
       -DPython3_EXECUTABLE=/usr/bin/python3 -DBUILD_TESTING=OFF \
    && cmake --build /tmp/pelagos-rviz-build --parallel 2 \
    && cmake --install /tmp/pelagos-rviz-build \
    && rm -rf /tmp/pelagos-rviz-panel /tmp/pelagos-rviz-build
