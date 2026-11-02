cmake_minimum_required(VERSION 3.16)
project(underwater_wavemap_ros LANGUAGES CXX)

if(NOT DEFINED WAVEMAP_SOURCE_DIR)
  set(WAVEMAP_SOURCE_DIR "" CACHE PATH "Path to the locked TP/Mapping/wavemap checkout")
endif()
if(NOT EXISTS "${WAVEMAP_SOURCE_DIR}/library/cpp/CMakeLists.txt")
  message(FATAL_ERROR "WAVEMAP_SOURCE_DIR must point to TP/Mapping/wavemap")
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(octomap REQUIRED)
find_package(octomap_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(visualization_msgs REQUIRED)

set(GENERATE_WAVEMAP_INSTALL_RULES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(ENABLE_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(${WAVEMAP_SOURCE_DIR}/library/cpp ${CMAKE_CURRENT_BINARY_DIR}/wavemap)

add_executable(wavemap_node src/wavemap_node.cpp)
target_compile_features(wavemap_node PRIVATE cxx_std_17)
target_compile_options(wavemap_node PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(wavemap_node PRIVATE include ${WAVEMAP_SOURCE_DIR}/library/cpp/include ${OCTOMAP_INCLUDE_DIRS})
ament_target_dependencies(wavemap_node
  geometry_msgs nav_msgs octomap_msgs rclcpp sensor_msgs std_msgs std_srvs visualization_msgs)
target_link_libraries(wavemap_node wavemap_core wavemap_io ${OCTOMAP_LIBRARIES})

foreach(target wavemap_to_octomap octomap_projection_check replay_far_rays)
  add_executable(${target} src/${target}.cpp)
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_include_directories(${target} PRIVATE include ${WAVEMAP_SOURCE_DIR}/library/cpp/include ${OCTOMAP_INCLUDE_DIRS})
  target_link_libraries(${target} wavemap_core wavemap_io ${OCTOMAP_LIBRARIES})
endforeach()

install(TARGETS wavemap_node wavemap_to_octomap octomap_projection_check replay_far_rays
  DESTINATION lib/${PROJECT_NAME})

ament_export_dependencies(geometry_msgs nav_msgs rclcpp sensor_msgs std_msgs std_srvs visualization_msgs)
ament_package()
