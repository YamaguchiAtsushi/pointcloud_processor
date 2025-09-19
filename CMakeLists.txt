cmake_minimum_required(VERSION 3.8)
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()
project(pointcloud_processor)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 依存関係の検索
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters)

# GeographicLibの検索
find_package(PkgConfig REQUIRED)
pkg_check_modules(GeographicLib REQUIRED geographiclib)

# インクルードディレクトリの設定
include_directories(
  ${PCL_INCLUDE_DIRS}
  ${GeographicLib_INCLUDE_DIRS}
)

# ライブラリディレクトリの設定
link_directories(
  ${PCL_LIBRARY_DIRS}
  ${GeographicLib_LIBRARY_DIRS}
)

# PCL定義の追加
add_definitions(${PCL_DEFINITIONS})


add_executable(virtual_lidar src/virtual_lidar.cpp)
ament_target_dependencies(virtual_lidar
  rclcpp
  visualization_msgs
  geometry_msgs
  std_msgs
  sensor_msgs
  nav_msgs
  pcl_conversions
  pcl_ros
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_sensor_msgs
  tf2_eigen
)


add_executable(excavated_surface_generator src/excavated_surface_generator.cpp)
ament_target_dependencies(excavated_surface_generator
  rclcpp
  visualization_msgs
  geometry_msgs
  std_msgs
  sensor_msgs
  nav_msgs
  pcl_conversions
  pcl_ros
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_sensor_msgs
  tf2_eigen
)

# pointcloud_merger実行ファイル
add_executable(pointcloud_merger src/pointcloud_merger.cpp)
ament_target_dependencies(pointcloud_merger
  rclcpp
  visualization_msgs
  geometry_msgs
  std_msgs
  sensor_msgs
  nav_msgs
  pcl_conversions
  pcl_ros
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_sensor_msgs
  tf2_eigen
)
# PCLとGeographicLibのリンク
target_link_libraries(pointcloud_merger
  ${PCL_LIBRARIES}
  ${GeographicLib_LIBRARIES}
  Geographic
)

# pointcloud_filter実行ファイル
add_executable(pointcloud_filter src/pointcloud_filter.cpp)
ament_target_dependencies(pointcloud_filter
  rclcpp
  visualization_msgs
  geometry_msgs
  std_msgs
  sensor_msgs
  nav_msgs
  pcl_conversions
  pcl_ros
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_sensor_msgs
  tf2_eigen
)
# PCLライブラリのリンク
target_link_libraries(pointcloud_filter
  ${PCL_LIBRARIES}
)

# calc_heading実行ファイル
add_executable(calc_heading src/calc_heading.cpp)
ament_target_dependencies(calc_heading
  rclcpp
  visualization_msgs
  geometry_msgs
  std_msgs
  sensor_msgs
  nav_msgs
  pcl_conversions
  pcl_ros
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_sensor_msgs
  tf2_eigen
)
# PCLライブラリのリンク
target_link_libraries(calc_heading
  ${PCL_LIBRARIES}
)

# インストール設定
install(TARGETS
  virtual_lidar
  excavated_surface_generator
  pointcloud_merger
  pointcloud_filter
  calc_heading
  DESTINATION lib/${PROJECT_NAME}
)

# 実行時に必要なファイル（ launch ファイルなど）をインストール
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()