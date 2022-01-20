cmake_minimum_required(VERSION 3.0.2)
project(mmseg_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  cv_bridge
  image_transport
)

find_package(OpenCV REQUIRED)

catkin_package(
  CATKIN_DEPENDS
  rospy
  sensor_msgs
  cv_bridge
  image_transport
)

catkin_python_setup()

catkin_install_python(PROGRAMS src/mmseg_ros.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
