FROM dapeng/omniplanner-noetic-base:20261021
COPY src/ /ws/src/
COPY checks/ /checks/
RUN g++ -std=c++17 -O1 -I/checks /checks/test_esdf_grid.cpp -o /checks/test_esdf_grid \
    && /checks/test_esdf_grid
RUN /bin/bash -c 'source /opt/ros/noetic/setup.bash && cd /ws \
    && catkin config --install --extend /opt/ros/noetic \
       --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O1 -DNDEBUG" \
       -DCATKIN_ENABLE_TESTING=OFF \
    && catkin build -j2 -p1 --no-status'
COPY planner.yaml /config/planner.yaml
WORKDIR /ws
