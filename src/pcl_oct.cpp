#define BOOST_BIND_NO_PLACEHOLDERS

#include "iostream"
#include "octomap_msgs/conversions.h"
#include <octomap/octomap_types.h>
#include "octomap_msgs/msg/octomap.hpp"
#include "octomap_ros/conversions.hpp"
#include <octomap/OcTreeKey.h>
#include <octomap/octomap.h>

#include "grid_map_cv/GridMapCvConverter.hpp"
#include "grid_map_ros/grid_map_ros.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include <grid_map_core/GridMap.hpp>
#include <grid_map_octomap/GridMapOctomapConverter.hpp>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "std_msgs/msg/header.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/static_transform_broadcaster.h"
#include "tf2_ros/transform_listener.h"
#include <tf2_eigen/tf2_eigen.hpp>

#include <algorithm>
#include <memory>
#include <stdio.h>
#include <string>

#include <opencv2/opencv.hpp>

typedef octomap::OcTree OcTreeT;
using std::placeholders::_1;

class pcl_oct : public rclcpp::Node
{
private:
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr subscription;

  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_map_publisher;

  std::unique_ptr<OcTreeT> octree_;

  std::unordered_map<octomap::OcTreeKey, double, octomap::OcTreeKey::KeyHash>
      timestamp_map;

  double res_;
  size_t tree_depth_;
  size_t max_tree_depth_;

  std::shared_ptr<tf2_ros::TransformListener> tf_listener{nullptr};
  std::unique_ptr<tf2_ros::Buffer> tf_buffer;
  octomap::KeyRay key_ray_;

  double max_range = 12;

  nav_msgs::msg::OccupancyGrid _grid;
  grid_map::GridMap gridMap;
  cv::Mat map_img;
  cv::Mat gray_img;

public:
  pcl_oct() : Node("pcl_oct")
  {
    subscription = this->create_subscription<sensor_msgs::msg::LaserScan>(
        "/scan", 10, std::bind(&pcl_oct::scan_callback, this, _1));

    grid_map_publisher = this->create_publisher<nav_msgs::msg::OccupancyGrid>("grid_msg", 10);

    tf_buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

    octree_ = std::make_unique<OcTreeT>(0.1);
    octree_->setProbHit(0.95);
    octree_->setProbMiss(0.05);
    octree_->setClampingThresMin(0.02);
    octree_->setClampingThresMax(0.98);
    tree_depth_ = octree_->getTreeDepth();
    max_tree_depth_ = tree_depth_;
  }

public:
  void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
  {
    geometry_msgs::msg::TransformStamped sensor_to_world_transform_stamped;
    try
    {
      sensor_to_world_transform_stamped = tf_buffer->lookupTransform(
          "odom", msg->header.frame_id, tf2::TimePointZero);
    }
    catch (const tf2::TransformException &ex)
    {
      RCLCPP_WARN(this->get_logger(), "%s", ex.what());
      return;
    }

    /*
     * Take the rotation of base_link wrt odom
     * Add the rotation to determine the angle as you iterate from angle_min to angle_max
     * Take sin and cos to get point in rotation corrected frame
     * Add the translation of base_link to obtain the point in odom frame
     * Add it to octomap - see if there's an inbuilt way to deal with 2D
     * If point is infinity, reduce it to max_scan_range and still trace ray and declare as empty without adding point to occupied cells
     */

    const auto &t = sensor_to_world_transform_stamped.transform.translation;
    octomap::point3d sensor_origin(t.x, t.y, t.z);

    tf2::Quaternion q(
        sensor_to_world_transform_stamped.transform.rotation.x,
        sensor_to_world_transform_stamped.transform.rotation.y,
        sensor_to_world_transform_stamped.transform.rotation.z,
        sensor_to_world_transform_stamped.transform.rotation.w);

    tf2::Matrix3x3 m(q);

    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);

    double start_angle = msg->angle_min + yaw;
    int max_ind = (msg->angle_max - msg->angle_min) / msg->angle_increment;

    RCLCPP_WARN(this->get_logger(), "angle range: [%f, %f] -> [%f, %f]",
                msg->angle_min, msg->angle_max, start_angle, start_angle + max_ind * msg->angle_increment);
    // RCLCPP_WARN(this->get_logger(), "(%f, %f) -> (%f, %f)", r, theta, point.x(), point.y());

    octomap::KeySet free_cells, occupied_cells;

    for (int i = 0; i < max_ind; i++)
    {
      double r{msg->ranges[i]}, theta{start_angle + i * msg->angle_increment};

      octomap::point3d point(cos(theta), sin(theta), 0);
      point *= std::min(r, max_range);

      if (octree_->computeRayKeys(sensor_origin, sensor_origin + point, key_ray_))
      {
        free_cells.insert(key_ray_.begin(), key_ray_.end());
      }
      octomap::OcTreeKey key;

      if (octree_->coordToKeyChecked(sensor_origin + point, key))
      {
        if (!(point.x() >= -0.6 && point.x() <= 0) && !(point.y() >= -0.5 && point.y() <= 0.5))
        {
          occupied_cells.insert(key);
        }
        else
        {
          free_cells.insert(key);
        }
      }
      else
      {
        RCLCPP_ERROR_STREAM(get_logger(),
                            "Could not generate Key for endpoint " << sensor_origin + point);
      }
    }
    for (auto it = free_cells.begin(); it != free_cells.end(); ++it)
    {
      // Make sure it doesn't end up in both in any case
      if (occupied_cells.find(*it) == occupied_cells.end())
      {
        octree_->updateNode(*it, false);
      }
    }
    for (auto it = occupied_cells.begin(); it != occupied_cells.end(); it++)
    {
      octree_->updateNode(*it, true);
    }
    octree_->prune();

    bool res = grid_map::GridMapOctomapConverter::fromOctomap(*octree_, "elevation", gridMap);
    if (res)
    {
      for (int r = 0; r < gridMap.get("elevation").rows(); r++)
      {
        for (int c = 0; c < gridMap.get("elevation").cols(); c++)
        {
          if (std::isnan(gridMap.get("elevation")(r, c)))
          {
            gridMap.get("elevation")(r, c) = -1;
          }
        }
      }

      grid_map::GridMapCvConverter::toImage<unsigned char, 4>(
          gridMap, "elevation", CV_8UC4, 0, 100, map_img);
      cv::cvtColor(map_img, gray_img, cv::COLOR_BGR2GRAY);

      // Size is taken as 2n+1: n being number of cells (half bot width: 0.4m -> 4)
      cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * 5, 2 * 5));
      cv::dilate(gray_img, gray_img, element);

      cv::GaussianBlur(gray_img, map_img, cv::Size(3, 3), 0.1, 0.1);

      grid_map::GridMapCvConverter::addLayerFromImage<unsigned char, 4>(
          map_img, "inflation", gridMap, 0, 100);

      for (int r = 0; r < gridMap.get("inflation").rows(); r++)
      {
        for (int c = 0; c < gridMap.get("inflation").cols(); c++)
        {
          if (gridMap.get("elevation")(r, c) == -1)
          {
            gridMap.get("elevation")(r, c) = 0;
          }

          gridMap.get("inflation")(r, c) =
              (gridMap.get("inflation")(r, c) + gridMap.get("elevation")(r, c));
          if (gridMap.get("inflation")(r, c) > 1)
          {
            gridMap.get("inflation")(r, c) = 1;
          }
        }
      }
      grid_map::GridMapRosConverter::toOccupancyGrid(gridMap, "inflation", 0, 1,
                                                     _grid);

      _grid.header.frame_id = "odom";
      this->grid_map_publisher->publish(_grid);
    }
    else
    {
      std::cout << "Error";
    }
  }
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<pcl_oct>());

  return 0;
}
