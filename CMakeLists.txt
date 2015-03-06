cmake_minimum_required(VERSION 2.8.3)
project(or_trajopt)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  openrave_catkin
)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
catkin_python_setup()

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  LIBRARIES ctrajoptpy
  CATKIN_DEPENDS openrave_catkin
)

###########
## Build ##
###########

include(ExternalProject)
ExternalProject_Add(
  libtrajopt
  BINARY_DIR ${CATKIN_DEVEL_PREFIX}
  GIT_REPOSITORY https://github.com/personalrobotics/trajopt.git
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> -DCMAKE_BUILD_TYPE=Debug
  INSTALL_COMMAND ""
)
ExternalProject_Get_Property(libtrajopt SOURCE_DIR)

add_custom_command(
  TARGET libtrajopt POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${SOURCE_DIR}/trajoptpy/ ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_PYTHON_DESTINATION}/trajoptpy
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_LIB_DESTINATION}/ctrajoptpy.so ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_PYTHON_DESTINATION}/ctrajoptpy.so
)
