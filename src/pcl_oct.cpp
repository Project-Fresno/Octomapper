#include <octomap/octomap_types.h>
#define BOOST_BIND_NO_PLACEHOLDERS

#include "iostream"
#include "octomap_msgs/conversions.h"
// #include "octomap_msgs/octomap_msgs/msg/octomap.hpp"
#include "octomap_msgs/msg/octomap.hpp"
#include "octomap_ros/conversions.hpp"
#include <octomap/OcTreeKey.h>
#include <octomap/octomap.h>
// #include "octomap_msgs/"

#include "pcl/common/transforms.h"
#include "pcl_conversions/pcl_conversions.h"
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter.h>
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

typedef octomap::OcTree OcTreeT;
typedef pcl::PointCloud<pcl::Normal> NormalCloud;
typedef pcl::PointXYZ POINT_TYPE;
using std::placeholders::_1;
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
  std::unique_ptr<OcTreeT> octree_;
  double res_;
  size_t tree_depth_;
  size_t max_tree_depth_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener{nullptr};
  std::unique_ptr<tf2_ros::Buffer> tf_buffer;
  octomap::KeyRay key_ray_;
  float max_range = 20;
  int occupancy_min_z_ = -100;
  int occupancy_max_z_ = 100;
  octomap_msgs::msg::Octomap msg;

  // octomap_msgs::octomap::ConstPtr oct_msg;

public:
  pcl_oct() : Node("pcl_oct") {
    subscription = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/depth_camera/points", 10,
        std::bind(&pcl_oct::pcl_topic_callback, this, _1));
    pcl_ground_publisher =
        this->create_publisher<sensor_msgs::msg::PointCloud2>("surfaces", 10);
    pcl_obs_publisher =
        this->create_publisher<sensor_msgs::msg::PointCloud2>("obs", 10);
    octomap_publisher =
        this->create_publisher<octomap_msgs::msg::Octomap>("oct_msg", 20);

    tf_buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

    octree_ = std::make_unique<OcTreeT>(0.05);
    octree_->setProbHit(0.7);
    octree_->setProbMiss(0.4);
    octree_->setClampingThresMin(0.12);
    octree_->setClampingThresMax(0.97);
    tree_depth_ = octree_->getTreeDepth();
    max_tree_depth_ = tree_depth_;
  }

public:
  void pcl_topic_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    unsigned int num_points = msg->width;
    RCLCPP_INFO(this->get_logger(),
                "The number of points in the input pointcloud is %i",
                num_points);
    pcl::fromROSMsg(*msg, *this->cloud);
    voxel_downsample(this->cloud, cloud_filtered);
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
    // calcSurfaceNormals_normal_method(this->cloud, cloud_normals);
    // findClusters(this->cloud, cloud_normals, surfaces);
    // voxel_downsample(this->cloud, cloud_filtered);
    pcl::transformPointCloud(*this->cloud_filtered, *this->cloud_filtered,
                             sensor_to_world);
    const auto &t = sensor_to_world_transform_stamped.transform.translation;
    tf2::Vector3 sensor_to_world_vec3{t.x, t.y, t.z};
    // pcl_conv_oct(this->cloud, sensor_to_world_vec3);
    plane_seg(this->cloud_filtered);
    pcl_conv_oct(sensor_to_world_vec3, this->cloud_p, this->cloud_o);

    // num_points = surfaces->width;
    // RCLCPP_INFO(this->get_logger(),
    // "The number of points in the output pointcloud is %i",
    // num_points);
    // pcl::toROSMsg(*this->cloud_filtered, surf);
    // this->pcl_ground_publisher->publish(surf);
    // pcl::visualization::PCLVisualizer viewer("PCL Viewer");
    // viewer.setBackgroundColor(0.0, 0.0, 0.5);
    // viewer->setBackgroundColor(0.0, 0.0, 0.5);
    // viewer.addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(
    // cloud, cloud_normals, 1, 0.1);
    // viewer.addPointCloud<pcl::PointXYZRGB>(cloud);
    // while (!viewer.wasStopped()) {
    // viewer.spinOnce();
  }

public:
  void voxel_downsample(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered) {
    // pcl::toPCLPointCloud2(*cloud, cloudpcl);

    pcl::VoxelGrid<pcl::PointXYZ> sor;

    sor.setInputCloud(cloud);
    sor.setLeafSize(0.08f, 0.08f, 0.08f);
    sor.filter(*cloud_filtered);
    // pcl::fromPCLPointCloud2(*cloud_filteredpcl, cloud_filtered);
  }

public:
  void pcl_conv_oct(const tf2::Vector3 &sensor_origin_tf,
                    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_obs,
                    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_ground) {
    const auto sensor_origin = octomap::pointTfToOctomap(sensor_origin_tf);
    octomap::KeySet free_cells, occupied_cells;
    // For ground pcl, mark all cells free
    for (pcl::PointCloud<POINT_TYPE>::const_iterator it = cloud_ground->begin();
         it != cloud_ground->end(); it++) {
      octomap::point3d point(it->x, it->y, it->z);
      if (it->x != std::numeric_limits<double>::infinity()) {
        if ((max_range < 0.0) ||
            ((point - sensor_origin).norm() <= max_range)) {
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
                sensor_origin +
                (point - sensor_origin).normalized() * max_range;
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
    }
    // For Obstacle pcl
    for (pcl::PointCloud<pcl::PointXYZ>::const_iterator it = cloud_obs->begin();
         it != cloud_obs->end(); it++) {
      octomap::point3d point(it->x, it->y, it->z);
      // std::cout << it->x << std::endl;
      if (it->x != std::numeric_limits<double>::infinity()) {
        if ((max_range < 0.0) ||
            ((point - sensor_origin).norm() <= max_range)) {
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
                sensor_origin +
                (point - sensor_origin).normalized() * max_range;
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
    }
    for (auto it = free_cells.begin(), end = free_cells.end(); it != end;
         ++it) {
      if (occupied_cells.find(*it) == occupied_cells.end()) {
        octree_->updateNode(*it, false);
      }
    }
    std::cout << free_cells.size() << std::endl;
    // now mark all occupied cells:
    for (auto it = occupied_cells.begin(), end = occupied_cells.end();
         it != end; it++) {
      octree_->updateNode(*it, true);
    }
    std::cout << occupied_cells.size() << std::endl;
    octree_->prune();
    RCLCPP_ERROR_STREAM(get_logger(), "size:" << octree_->size());
    octomap_msgs::binaryMapToMsg(*this->octree_, msg);
    msg.header.frame_id = "odom";
    // msg.header.stamp = this->get_clock();
    this->octomap_publisher->publish(msg);
    // for (OcTreeT::iterator it = octree_->begin(max_tree_depth_),
    //                        end = octree_->end();
    //      it != end; ++it) {
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
    // }
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
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<pcl_oct>());

  return 0;
}
