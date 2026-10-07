cmake_minimum_required(VERSION 3.8)
project(tianji_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config meshes rviz urdf
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_urdf_vs_sdk_fk test/test_urdf_vs_sdk_fk.py)
endif()

ament_package()
