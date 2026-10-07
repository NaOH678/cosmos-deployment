cmake_minimum_required(VERSION 3.8)
project(marvin_wuji_d435_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY urdf meshes third_party_licenses
  DESTINATION share/${PROJECT_NAME}
)

install(
  FILES README.md preview.png
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
