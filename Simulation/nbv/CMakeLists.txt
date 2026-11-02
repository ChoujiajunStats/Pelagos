cmake_minimum_required(VERSION 3.0.2)
project(pelagos_nbv)
add_compile_options(-std=c++14)
find_package(catkin REQUIRED COMPONENTS roscpp nbvplanner nav_msgs sensor_msgs geometry_msgs std_msgs visualization_msgs octomap_msgs message_generation)
add_service_files(FILES Step.srv)
generate_messages(DEPENDENCIES std_msgs sensor_msgs geometry_msgs octomap_msgs)
catkin_package(CATKIN_DEPENDS message_runtime nbvplanner nav_msgs sensor_msgs geometry_msgs octomap_msgs)
include_directories(${catkin_INCLUDE_DIRS})
add_executable(pelagos_nbv_node native.cpp)
add_dependencies(pelagos_nbv_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# nbvplanner is a catkin component above; its export resolves the library's
# full path in the isolated devel space. A bare name would add an unresolved -l.
target_link_libraries(pelagos_nbv_node ${catkin_LIBRARIES})
