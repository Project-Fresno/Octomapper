#include <octomap/octomap_types.h>
#define BOOST_BIND_NO_PLACEHOLDERS

#include "iostream"
#include "octomap_msgs/conversions.h"
// #include "octomap_msgs/octomap_msgs/msg/octomap.hpp"
#include "octomap_msgs/msg/octomap.hpp"
#include "octomap_ros/conversions.hpp"
#include <octomap/ColorOcTree.h>
#include <octomap/OcTreeKey.h>
#include <octomap/octomap.h>
// #include "octomap_msgs/"
#include "grid_map_cv/GridMapCvConverter.hpp"
#include "grid_map_ros/grid_map_ros.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include <grid_map_core/GridMap.hpp>
#include <grid_map_octomap/GridMapOctomapConverter.hpp>

#include "pcl/common/transforms.h"
#include "pcl_conversions/pcl_conversions.h"
#include <pcl/features/normal_3d.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/sac_model_perpendicular_plane.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/sac_segmentation.h>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
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
typedef pcl::PointCloud<pcl::Normal> NormalCloud;
typedef pcl::PointXYZ POINT_TYPE;
using std::placeholders::_1;

// class ColorOcTreeNodeStamped : public octomap::ColorOcTreeNode {
// public:
//   ColorOcTreeNodeStamped() : timestamp(0.0) {} // Constructor
//
//   void updateTimestamp() { timestamp = this->get_clock()->now(); }
//
// private:
//   double timestamp;
// };

class pcl_oct : public rclcpp::Node {
private:
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription;

  pcl::PointCloud<POINT_TYPE>::Ptr cloud =
      pcl::make_shared<pcl::PointCloud<POINT_TYPE>>();
  pcl::PointCloud<POINT_TYPE>::Ptr cloud_filtered =
      pcl::make_shared<pcl::PointCloud<POINT_TYPE>>();
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pcl_obs_publisher;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      pcl_ground_publisher;
  rclcpp::Publisher<octomap_msgs::msg::Octomap>::SharedPtr octomap_publisher;
  rclcpp::Publisher<octomap_msgs::msg::Octomap>::SharedPtr ground_publisher;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_map_publisher;

  pcl::ModelCoefficients::Ptr coefficients =
      pcl::make_shared<pcl::ModelCoefficients>();
  pcl::PointCloud<POINT_TYPE>::Ptr cloud_o =
      pcl::make_shared<pcl::PointCloud<POINT_TYPE>>();
  sensor_msgs::msg::PointCloud2 plane;
  sensor_msgs::msg::PointCloud2 obs;
  sensor_msgs::msg::PointCloud2 surf;
  pcl::PointIndices::Ptr inliers = pcl::make_shared<pcl::PointIndices>();
  pcl::ExtractIndices<pcl::PointXYZ> extract;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_p =
      pcl::make_shared<pcl::PointCloud<POINT_TYPE>>();
  pcl::ConditionAnd<POINT_TYPE>::Ptr z_obstacle_cond;
  pcl::ConditionAnd<POINT_TYPE>::Ptr z_obstacle_cond_inv;
  // pcl::PassThrough<pcl::PointXYZ> pass;

  std::unique_ptr<OcTreeT> octree_;
  std::unique_ptr<OcTreeT> octree_ground;

  std::unordered_map<octomap::OcTreeKey, double, octomap::OcTreeKey::KeyHash>
      timestamp_map;

  double res_;
  size_t tree_depth_;
  size_t max_tree_depth_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener{nullptr};
  std::unique_ptr<tf2_ros::Buffer> tf_buffer;
  octomap::KeyRay key_ray_;
  float max_range = 10;
  int occupancy_min_z_ = -100;
  int occupancy_max_z_ = 100;
  octomap_msgs::msg::Octomap msg;
  pcl::ConditionalRemoval<pcl::PointXYZ> condrem =
      pcl::ConditionalRemoval<POINT_TYPE>();
  pcl::ConditionalRemoval<pcl::PointXYZ> condrem_inv =
      pcl::ConditionalRemoval<POINT_TYPE>();
  // octomap_msgs::octomap::ConstPtr oct_msg;
  //
  nav_msgs::msg::OccupancyGrid _grid;
  grid_map::GridMap gridMap;
  cv::Mat map_img;
  cv::Mat gray_img;

public:
  pcl_oct() : Node("pcl_oct") {
    // this->declare_parameter<std::string>("depth_topic",
    // "/depth_camera/points");
    this->declare_parameter<std::string>("depth_topic", "/depth_camera/points");
    // "/depth_camera/points");
    this->declare_parameter<double>("ground_cutoff_height", 0.2);

    subscription = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        this->get_parameter("depth_topic").as_string(), 10,
        std::bind(&pcl_oct::pcl_topic_callback, this, _1));
    // subscription = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    //     "/camera/depth/color/points", 10,
    //     std::bind(&pcl_oct::pcl_topic_callback, this, _1));

    pcl_ground_publisher =
        this->create_publisher<sensor_msgs::msg::PointCloud2>("surfaces", 10);
    pcl_obs_publisher =
        this->create_publisher<sensor_msgs::msg::PointCloud2>("obs", 10);
    octomap_publisher =
        this->create_publisher<octomap_msgs::msg::Octomap>("oct_msg", 20);
    ground_publisher = this->create_publisher<octomap_msgs::msg::Octomap>(
        "ground_oct_msg", 20);
    grid_map_publisher =
        this->create_publisher<nav_msgs::msg::OccupancyGrid>("grid_msg", 10);

    tf_buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

    pcl::ConditionAnd<POINT_TYPE>::Ptr range_cond(
        new pcl::ConditionAnd<POINT_TYPE>());
    pcl::ConditionAnd<POINT_TYPE>::Ptr range_cond_inv(
        new pcl::ConditionAnd<POINT_TYPE>());
    z_obstacle_cond = range_cond;
    z_obstacle_cond_inv = range_cond_inv;
    z_obstacle_cond->addComparison(pcl::FieldComparison<POINT_TYPE>::Ptr(
        new pcl::FieldComparison<POINT_TYPE>(
            "z", pcl::ComparisonOps::GT,
            this->get_parameter("ground_cutoff_height").as_double())));
    z_obstacle_cond_inv->addComparison(pcl::FieldComparison<POINT_TYPE>::Ptr(
        new pcl::FieldComparison<POINT_TYPE>(
            "z", pcl::ComparisonOps::LT,
            this->get_parameter("ground_cutoff_height").as_double())));

    octree_ = std::make_unique<OcTreeT>(0.1);
    octree_->setProbHit(0.65);
    octree_->setProbMiss(0.45);
    octree_->setClampingThresMin(0.12);
    octree_->setClampingThresMax(0.95);
    tree_depth_ = octree_->getTreeDepth();
    max_tree_depth_ = tree_depth_;
    // octree_ground = std::make_unique<OcTreeT>(0.05);
    // octree_ground->setProbHit(0.7);
    // octree_ground->setProbMiss(0.4);
    // octree_ground->setClampingThresMin(0.12);
    // octree_ground->setClampingThresMax(0.97);
    // tree_depth_ = octree_ground->getTreeDepth();
    // max_tree_depth_ = 10;
    //
  }

public:
  void pcl_topic_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    unsigned int num_points = msg->width;
    RCLCPP_INFO(this->get_logger(),
                "The number of points in the input pointcloud is %i",
                num_points);
    pcl::fromROSMsg(*msg, *this->cloud);
    voxel_downsample(this->cloud, cloud_filtered);
    // this->cloud_filtered = this->cloud;

    geometry_msgs::msg::TransformStamped sensor_to_world_transform_stamped;
    try {
      sensor_to_world_transform_stamped = tf_buffer->lookupTransform(
          "odom", cloud->header.frame_id, tf2::TimePointZero);
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN(this->get_logger(), "%s", ex.what());
      return;
    }
    Eigen::Matrix4f sensor_to_world =
        tf2::transformToEigen(sensor_to_world_transform_stamped.transform)
            .matrix()
            .cast<float>();
    pcl::transformPointCloud(*this->cloud_filtered, *this->cloud_filtered,
                             sensor_to_world);
    const auto &t = sensor_to_world_transform_stamped.transform.translation;
    condrem.setInputCloud(this->cloud_filtered);
    condrem.setCondition(z_obstacle_cond);
    condrem.filter(*this->cloud_o);
    // condrem.setNegative(true);
    condrem_inv.setInputCloud(this->cloud_filtered);
    condrem_inv.setCondition(z_obstacle_cond_inv);
    condrem_inv.filter(*this->cloud_p);

    pcl::toROSMsg(*this->cloud_p, plane);
    plane.header.frame_id = "odom";
    this->pcl_ground_publisher->publish(plane);

    tf2::Vector3 sensor_to_world_vec3{t.x, t.y, t.z};
    pcl_conv_oct(sensor_to_world_vec3, this->cloud_o, this->cloud_p);
  }

public:
  void voxel_downsample(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered) {

    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(0.1f, 0.1f, 0.1f);
    sor.filter(*cloud_filtered);
  }

public:
  void pcl_conv_oct(const tf2::Vector3 &sensor_origin_tf,
                    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_obs,
                    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_ground) {
    const auto sensor_origin = octomap::pointTfToOctomap(sensor_origin_tf);
    octomap::KeySet free_cells, occupied_cells;
    // For ground pcl, mark all cells free
    // for (pcl::PointCloud<POINT_TYPE>::const_iterator it =
    // cloud_ground->begin();
    //      it != cloud_ground->end(); it++) {
    //   octomap::point3d point(it->x, it->y, it->z);
    //   if (it->x != std::numeric_limits<double>::infinity()) {
    //     if ((max_range < 0.0) ||
    //         ((point - sensor_origin).norm() <= max_range)) {
    //       octomap::OcTreeKey key;
    //       if (octree_->coordToKeyChecked(point, key)) {
    //         free_cells.insert(key);
    //         // td::cout << key;
    //         octree_->averageNodeColor(key, 0, 255, 0);
    //       }
    //     }
    //   }
    // }
    //      if (octree_->computeRayKeys(sensor_origin, point, key_ray_)) {
    //         free_cells.insert(key_ray_.begin(), key_ray_.end());
    //       }
    //       octomap::OcTreeKey key;
    //       if (octree_->coordToKeyChecked(point, key)) {
    //         free_cells.insert(key);
    //       }
    //     } else {
    //       octomap::point3d new_end =
    //           sensor_origin + (point - sensor_origin).normalized() *
    //           max_range;
    //       if (octree_->computeRayKeys(sensor_origin, new_end,
    //       key_ray_)) {
    //         free_cells.insert(key_ray_.begin(), key_ray_.end());
    //
    //         octomap::point3d new_end =
    //             sensor_origin +
    //             (point - sensor_origin).normalized() * max_range;
    //         octomap::OcTreeKey end_key;
    //
    //         if (octree_->coordToKeyChecked(new_end, end_key)) {
    //           free_cells.insert(end_key);
    //         } else {
    //           RCLCPP_ERROR_STREAM(get_logger(),
    //                               "Could not generate Key for endpoint
    //                               "
    //                                   << new_end);
    //         }
    //       }
    //     }
    //   }
    // }
    // For Obstacle pcl
    for (pcl::PointCloud<pcl::PointXYZ>::const_iterator it = cloud_obs->begin();
         it != cloud_obs->end(); it++) {
      octomap::point3d point(it->x, it->y, it->z);
      // std::cout << it->x << std::endl;
      // if (it->x != std::numeric_limits<double>::infinity()) {
      if ((max_range < 0.0) || ((point - sensor_origin).norm() <= max_range)) {
        if (octree_->computeRayKeys(sensor_origin, point, key_ray_)) {
          free_cells.insert(key_ray_.begin(), key_ray_.end());
        }
        octomap::OcTreeKey key;
        if (octree_->coordToKeyChecked(point, key)) {
          occupied_cells.insert(key);
        }
      } else {
        octomap::point3d new_end =
            sensor_origin + (point - sensor_origin).normalized() * max_range;
        if (octree_->computeRayKeys(sensor_origin, new_end, key_ray_)) {
          free_cells.insert(key_ray_.begin(), key_ray_.end());

          octomap::point3d new_end =
              sensor_origin + (point - sensor_origin).normalized() * max_range;
          octomap::OcTreeKey end_key;

          if (octree_->coordToKeyChecked(new_end, end_key)) {
            free_cells.insert(end_key);
          } else {
            RCLCPP_ERROR_STREAM(get_logger(),
                                "Could not generate Key for endpoint "
                                    << new_end);
          }
        }
      }
    }
    for (auto it = free_cells.begin(), end = free_cells.end(); it != end;
         ++it) {
      if (occupied_cells.find(*it) == occupied_cells.end()) {
        octree_->updateNode(*it, false);
      }
    }
    std::cout << free_cells.size() << std::endl;
    this->octomap_publisher->publish(msg);
    // now mark all occupied cells:
    for (auto it = occupied_cells.begin(), end = occupied_cells.end();
         it != end; it++) {
      octree_->updateNode(*it, true);
      timestamp_map[*it] = this->get_clock()->now().nanoseconds();
    }
    // msg.header.stamp = this->get_clock();
    for (OcTreeT::iterator it = octree_->begin_leafs(),
                           end = octree_->end_leafs();
         it != end; ++it) {
      auto time_it = timestamp_map.find(it.getKey());
      if (time_it != timestamp_map.end()) {
        std::cout << it.getKey()[0];
        std::cout << this->get_clock()->now().nanoseconds() - time_it->second
                  << "\n";
        if ((this->get_clock()->now().nanoseconds() - time_it->second) /
                100000 >
            100000) {
          it->setLogOdds(octomap::logodds(0.0));
          timestamp_map.erase(time_it);
        }
        // if (isSpeckleNode(it.getKey())) {
        //   it->setLogOdds(octomap::logodds(0.0));
        // }
      }
    }
    std::cout << occupied_cells.size() << std::endl;
    octree_->prune();
    RCLCPP_ERROR_STREAM(get_logger(), "size:" << octree_->size());
    octomap_msgs::fullMapToMsg(*this->octree_, msg);
    msg.header.frame_id = "odom";

    //   if (octree_->isNodeOccupied(*it)) {
    //     double z = it.getZ();
    //     double half_size = it.getSize() / 2.0;
    //     if (z + half_size > occupancy_min_z_ &&
    //         z - half_size < occupancy_max_z_) {
    //       double x = it.getX();
    //       double y = it.getY();
    //     }
    //     POINT_TYPE _point = ;
    //     _point.x = x;
    //     _point.y = y;
    //     _point.z = z;
    //   }
    bool res = grid_map::GridMapOctomapConverter::fromOctomap(
        *octree_, "elevation", gridMap);
    // std::cout << gridM std::endl;
    if (res) {
      // grid_map::GridMapRosConverter::toOccupancyGrid(gridMap, "elevation",
      // 100.0, -1.0, _grid);

      for (int r = 0; r < gridMap.get("elevation").rows(); r++) {
        for (int c = 0; c < gridMap.get("elevation").cols(); c++) {
          if (std::isnan(gridMap.get("elevation")(r, c))) {
            gridMap.get("elevation")(r, c) = -1;
          }
        }
      }
      // replaceNan(gridMap.get("elevation"), -1);
      grid_map::GridMapCvConverter::toImage<unsigned char, 4>(
          gridMap, "elevation", CV_8UC4, 0, 100, map_img);
      // cv::imshow("orginal", map_img);
      // cv::waitKey(0);
      cv::cvtColor(map_img, gray_img, cv::COLOR_BGR2GRAY);

      // Size is taken as 2n+1: n being number of cells (half bot width: 0.4m ->
      // 4 cells)
      cv::Mat element = cv::getStructuringElement(
          cv::MORPH_RECT, cv::Size(2 * 4 + 1, 2 * 4 + 1));
      cv::dilate(gray_img, gray_img, element);

      cv::GaussianBlur(gray_img, map_img, cv::Size(11, 11), 0, 0);

      grid_map::GridMapCvConverter::addLayerFromImage<unsigned char, 4>(
          map_img, "inflation", gridMap, 0, 100);
      // std::cout << gridMap.get("elevation") << "\n";
      for (int r = 0; r < gridMap.get("inflation").rows(); r++) {
        for (int c = 0; c < gridMap.get("inflation").cols(); c++) {
          if (gridMap.get("elevation")(r, c) == -1) {
            gridMap.get("elevation")(r, c) = 0;
          }

          gridMap.get("inflation")(r, c) =
              (gridMap.get("inflation")(r, c) + gridMap.get("elevation")(r, c));
          if (gridMap.get("inflation")(r, c) > 1) {
            gridMap.get("inflation")(r, c) = 1;
          }
        }
      }
      std::cout << gridMap.get("inflation") << "\n";
      grid_map::GridMapRosConverter::toOccupancyGrid(gridMap, "inflation", 0, 1,
                                                     _grid);
      // cv::imshow("Blured", map_img);
      // cv::waitKey(0);

      _grid.header.frame_id = "odom";
      this->grid_map_publisher->publish(_grid);
      std::cout << "grid_size " << _grid.info.width << "\n";
    } else {
      std::cout << "Error";
    }
  }

public:
  void plane_seg(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);
    std::cerr << "Model inliers: " << inliers->indices.size() << std::endl;
    // for (const auto &idx : inliers->indices)
    //   std::cerr << idx << "    " << cloud->points[idx].x << " "
    //             << cloud->points[idx].y << " " << cloud->points[idx].z
    //             << std::endl;
    std::cerr << "Model coefficients: " << coefficients->values[0] << " "
              << coefficients->values[1] << " " << coefficients->values[2]
              << " " << coefficients->values[3] << std::endl;
    Eigen::Vector3f plane_normal(coefficients->values[0],
                                 coefficients->values[1],
                                 coefficients->values[2]);
    std::cout << "Normal vector: (" << plane_normal[0] << ", "
              << plane_normal[1] << ", " << plane_normal[2] << ")" << std::endl;
    Eigen::Vector3f normalized_normal = plane_normal.normalized();
    float dot_product = normalized_normal[0] * 0 + normalized_normal[1] * 0 +
                        normalized_normal[2] * 1;
    float slope = std::acos(dot_product);
    float theta = std::atan(slope) * 180 / M_PI;
    std::cout << "Theta: " << theta << std::endl;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*this->cloud_p);
    extract.setNegative(true);
    extract.filter(*this->cloud_o);
    pcl::toROSMsg(*this->cloud_p, plane);
    this->pcl_ground_publisher->publish(plane);
    pcl::toROSMsg(*this->cloud_o, obs);
    this->pcl_obs_publisher->publish(obs);
  }
  bool isSpeckleNode(const octomap::OcTreeKey &n_key) const {
    octomap::OcTreeKey key;
    bool neighbor_found = false;
    for (key[2] = n_key[2] - 1; !neighbor_found && key[2] <= n_key[2] + 1;
         ++key[2]) {
      for (key[1] = n_key[1] - 1; !neighbor_found && key[1] <= n_key[1] + 1;
           ++key[1]) {
        for (key[0] = n_key[0] - 1; !neighbor_found && key[0] <= n_key[0] + 1;
             ++key[0]) {
          if (key != n_key) {
            octomap::OcTreeNode *node = octree_->search(key);
            if (node && octree_->isNodeOccupied(node)) {
              // we have a neighbor=> break!
              neighbor_found = true;
            }
          }
        }
      }
    }

    return neighbor_found;
  }
  grid_map::Matrix replaceNan(grid_map::Matrix &m, const double newValue) {

    return m;
  }
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<pcl_oct>());

  return 0;
}
