cmake_minimum_required(VERSION 3.14.0)
project(sequence_controller)

add_definitions(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  dynamic_reconfigure
  tf
  mavros
  mavros_extras
  mavros_msgs
  mavlink
  controller_msgs
  message_filters
  message_generation
  yaml_cpp_catkin
  gazebo_msgs
)

catkin_package(
  LIBRARIES sequence_controller
  CATKIN_DEPENDS roscpp rospy std_msgs mavros_msgs geometry_msgs sensor_msgs tf  message_filters controller_msgs yaml_cpp_catkin gazebo_msgs
)

#############
# LIBRARIES #
#############
include_directories(
  ${Boost_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
)

############
# BINARIES #
############

add_executable(parser
  src/parser.cpp
)

target_link_libraries(parser ${catkin_LIBRARIES})
