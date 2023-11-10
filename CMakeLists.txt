cmake_minimum_required(VERSION 3.8)
project(octomap_mapper)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(Eigen REQUIRED)
find_package(octomap REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(octomap_msgs REQUIRED)



add_executable(pcl_oct src/pcl_oct.cpp)

target_link_libraries(pcl_oct
  ${PCL_LIBRARIES}
  ${OCTOMAP_LIBRARIES}
  ${rclcpp_LIBRARIES}
  ${std_msgs_LIBRARIES}
  ${pcl_ros_LIBRARIES}
  ${tf2_eigen_LIBRARIES}
  ${octomap_msgs_LIBRARIES}
)
target_include_directories(pcl_oct PRIVATE
  ${PCL_INCLUDE_DIRS}
  ${OCTOMAP_INCLUDE_DIRS}
  ${rclcpp_INCLUDE_DIRS}
  ${std_msgs_INCLUDE_DIRS}
  ${pcl_ros_INCLUDE_DIRS}
  ${tf2_eigen_INCLUDE_DIRS}
  ${octomap_msgs_INCLUDE_DIRS}
)

target_compile_features(pcl_oct PUBLIC c_std_99 cxx_std_17) 
ament_target_dependencies(pcl_oct
  "rclcpp"
  "std_msgs"
  "PCL"
  "pcl_ros"
  "pcl_conversions"
  "octomap"
  "tf2_eigen"
"octomap_msgs"
  )


install(TARGETS pcl_oct
  DESTINATION lib/${PROJECT_NAME})


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
