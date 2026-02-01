#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_box.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <cmath>
#include <algorithm>
#include <set>

class ExcavationTerrainGenerator : public rclcpp::Node {
public:
    ExcavationTerrainGenerator() : Node("excavation_terrain_generator") {
        // 基本パラメータ
        this->declare_parameter("excavation.depth", 1.0);
        this->declare_parameter("excavation.slope_angle", 75.0);
        this->declare_parameter("excavation.offset_x", 5.0);
        this->declare_parameter("excavation.offset_y", 0.0);
        this->declare_parameter("excavation.point_density", 0.05);
        this->declare_parameter("excavation.enabled", true);
        this->declare_parameter("excavation.terrain_search_radius", 0.5);
        
        // L字型掘削パラメータ
        this->declare_parameter("excavation.l_shape_enabled", true);
        this->declare_parameter("excavation.arm1_length", 2.0);
        this->declare_parameter("excavation.arm1_width", 1.2);
        this->declare_parameter("excavation.arm2_length", 2.0);
        this->declare_parameter("excavation.arm2_width", 1.2);
        
        // 通常の掘削パラメータ
        this->declare_parameter("excavation.width", 1.2);
        this->declare_parameter("excavation.length", 1.8);
        
        updateParameters();
        
        // TF2 Buffer and Listener
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        
        // Subscriber
        matched_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/matched_point_cloud", 10,
            std::bind(&ExcavationTerrainGenerator::matchedCloudCallback, this, std::placeholders::_1));
        
        // Publishers
        excavated_terrain_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/excavated_terrain", 10);
        excavation_area_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/excavation_area", 10);
        excavation_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/excavation_markers", 10);
        
        // Parameter update timer
        param_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&ExcavationTerrainGenerator::updateParameters, this));
            
        RCLCPP_INFO(this->get_logger(), "Excavation Terrain Generator initialized");
        if (l_shape_enabled_) {
            RCLCPP_INFO(this->get_logger(), 
                "L-Shape Mode - Arm1: %.2fm x %.2fm, Arm2: %.2fm x %.2fm, Depth: %.2fm",
                arm1_length_, arm1_width_, arm2_length_, arm2_width_, depth_);
        } else {
            RCLCPP_INFO(this->get_logger(), 
                "Rectangle Mode - Length: %.2fm, Width: %.2fm, Depth: %.2fm",
                length_, width_, depth_);
        }
    }
    
private:
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr matched_cloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr excavated_terrain_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr excavation_area_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr excavation_marker_pub_;
    
    rclcpp::TimerBase::SharedPtr param_timer_;
    
    // パラメータ
    double width_, length_, depth_;
    double slope_angle_deg_, slope_angle_rad_;
    double offset_x_, offset_y_;
    double point_density_;
    bool enabled_;
    double terrain_search_radius_;
    
    // L字型パラメータ
    bool l_shape_enabled_;
    double arm1_length_, arm1_width_;
    double arm2_length_, arm2_width_;
    
    // 掘削領域の構造体
    struct ExcavationBox {
        double center_x_local;
        double center_y_local;
        double length;
        double width;
        double min_x, max_x, min_y, max_y;  // 境界ボックス
    };
    
    void updateParameters() {
        depth_ = this->get_parameter("excavation.depth").as_double();
        slope_angle_deg_ = this->get_parameter("excavation.slope_angle").as_double();
        slope_angle_rad_ = slope_angle_deg_ * M_PI / 180.0;
        offset_x_ = this->get_parameter("excavation.offset_x").as_double();
        offset_y_ = this->get_parameter("excavation.offset_y").as_double();
        point_density_ = this->get_parameter("excavation.point_density").as_double();
        enabled_ = this->get_parameter("excavation.enabled").as_bool();
        terrain_search_radius_ = this->get_parameter("excavation.terrain_search_radius").as_double();
        
        l_shape_enabled_ = this->get_parameter("excavation.l_shape_enabled").as_bool();
        arm1_length_ = this->get_parameter("excavation.arm1_length").as_double();
        arm1_width_ = this->get_parameter("excavation.arm1_width").as_double();
        arm2_length_ = this->get_parameter("excavation.arm2_length").as_double();
        arm2_width_ = this->get_parameter("excavation.arm2_width").as_double();
        
        width_ = this->get_parameter("excavation.width").as_double();
        length_ = this->get_parameter("excavation.length").as_double();
    }
    
    std::vector<ExcavationBox> getExcavationBoxes() {
        std::vector<ExcavationBox> boxes;
        
        if (l_shape_enabled_) {
            // L字型: 2つの長方形で構成
            // 縦アーム（下方向に伸びる）
            ExcavationBox arm1;
            arm1.center_x_local = 0.0;
            arm1.center_y_local = -arm1_length_ / 2.0;
            arm1.length = arm1_width_;
            arm1.width = arm1_length_;
            arm1.min_x = arm1.center_x_local - arm1.length / 2.0;
            arm1.max_x = arm1.center_x_local + arm1.length / 2.0;
            arm1.min_y = arm1.center_y_local - arm1.width / 2.0;
            arm1.max_y = arm1.center_y_local + arm1.width / 2.0;
            boxes.push_back(arm1);
            
            // 横アーム（右方向に伸びる）
            ExcavationBox arm2;
            arm2.center_x_local = arm2_length_ / 2.0;
            arm2.center_y_local = -arm1_length_ + arm2_width_ / 2.0;
            arm2.length = arm2_length_;
            arm2.width = arm2_width_;
            arm2.min_x = arm2.center_x_local - arm2.length / 2.0;
            arm2.max_x = arm2.center_x_local + arm2.length / 2.0;
            arm2.min_y = arm2.center_y_local - arm2.width / 2.0;
            arm2.max_y = arm2.center_y_local + arm2.width / 2.0;
            boxes.push_back(arm2);
        } else {
            // 通常の長方形
            ExcavationBox box;
            box.center_x_local = 0.0;
            box.center_y_local = 0.0;
            box.length = length_;
            box.width = width_;
            box.min_x = -length_ / 2.0;
            box.max_x = length_ / 2.0;
            box.min_y = -width_ / 2.0;
            box.max_y = width_ / 2.0;
            boxes.push_back(box);
        }
        
        return boxes;
    }
    
    double getTerrainHeight(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
                           double x, double y) {
        if (cloud->empty()) return 0.0;
        
        pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
        kdtree.setInputCloud(cloud);
        
        pcl::PointXYZRGB search_point;
        search_point.x = x;
        search_point.y = y;
        search_point.z = 0;
        
        std::vector<int> point_indices;
        std::vector<float> point_distances;
        
        if (kdtree.radiusSearch(search_point, terrain_search_radius_, 
                               point_indices, point_distances) > 0) {
            double sum_z = 0.0;
            int valid_count = 0;
            
            for (int idx : point_indices) {
                double dx = cloud->points[idx].x - x;
                double dy = cloud->points[idx].y - y;
                double distance_2d = sqrt(dx*dx + dy*dy);
                
                if (distance_2d <= terrain_search_radius_) {
                    sum_z += cloud->points[idx].z;
                    valid_count++;
                }
            }
            
            if (valid_count > 0) {
                return sum_z / valid_count;
            }
        }
        
        std::vector<int> k_indices(1);
        std::vector<float> k_distances(1);
        if (kdtree.nearestKSearch(search_point, 1, k_indices, k_distances) > 0) {
            return cloud->points[k_indices[0]].z;
        }
        
        return 0.0;
    }
    
    // L字領域内の点かどうかを判定
    bool isInsideAnyBox(double x_local, double y_local, const std::vector<ExcavationBox>& boxes) {
        for (const auto& box : boxes) {
            if (x_local >= box.min_x && x_local <= box.max_x &&
                y_local >= box.min_y && y_local <= box.max_y) {
                return true;
            }
        }
        return false;
    }
    
    // 外壁エッジかどうかを判定（L字の外周のみ）
    bool isOuterEdge(double x_local, double y_local, const std::vector<ExcavationBox>& boxes, 
                     double tolerance = 0.01) {
        // この点が少なくとも1つのボックス内にある必要がある
        if (!isInsideAnyBox(x_local, y_local, boxes)) {
            return false;
        }
        
        // 4方向をチェック（上下左右）
        double delta = tolerance;
        bool has_outside_neighbor = false;
        
        // 上
        if (!isInsideAnyBox(x_local + delta, y_local, boxes)) has_outside_neighbor = true;
        // 下
        if (!isInsideAnyBox(x_local - delta, y_local, boxes)) has_outside_neighbor = true;
        // 右
        if (!isInsideAnyBox(x_local, y_local + delta, boxes)) has_outside_neighbor = true;
        // 左
        if (!isInsideAnyBox(x_local, y_local - delta, boxes)) has_outside_neighbor = true;
        
        return has_outside_neighbor;
    }
    
    void matchedCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        if (!enabled_) {
            excavated_terrain_pub_->publish(*msg);
            return;
        }
        
        geometry_msgs::msg::TransformStamped zx120_transform;
        try {
            zx120_transform = tf_buffer_->lookupTransform(
                "map", "zx120/base_link", 
                tf2::TimePointZero, tf2::durationFromSec(0.1));
        } catch (tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "Could not get zx120 transform: %s", ex.what());
            excavated_terrain_pub_->publish(*msg);
            return;
        }
        
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::fromROSMsg(*msg, *input_cloud);
        
        tf2::Transform zx120_tf;
        tf2::fromMsg(zx120_transform.transform, zx120_tf);
        
        tf2::Vector3 excavation_center_local(offset_x_, offset_y_, 0);
        tf2::Vector3 excavation_center_2d = zx120_tf * excavation_center_local;
        
        double terrain_height = getTerrainHeight(input_cloud, 
                                               excavation_center_2d.x(), 
                                               excavation_center_2d.y());
        
        tf2::Vector3 excavation_center(excavation_center_2d.x(), 
                                      excavation_center_2d.y(), 
                                      terrain_height);
        
        tf2::Quaternion zx120_rotation = zx120_tf.getRotation();
        tf2::Matrix3x3 m(zx120_rotation);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);
        
        // 掘削領域の可視化
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr excavation_area(new pcl::PointCloud<pcl::PointXYZRGB>);
        generateExcavationArea(excavation_area, excavation_center, yaw, input_cloud);
        
        // 掘削処理
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr result_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        processExcavation(input_cloud, result_cloud, excavation_center, yaw);
        
        // Publish
        sensor_msgs::msg::PointCloud2 output_msg;
        pcl::toROSMsg(*result_cloud, output_msg);
        output_msg.header = msg->header;
        output_msg.header.frame_id = "map";
        excavated_terrain_pub_->publish(output_msg);
        
        sensor_msgs::msg::PointCloud2 area_msg;
        pcl::toROSMsg(*excavation_area, area_msg);
        area_msg.header = msg->header;
        area_msg.header.frame_id = "map";
        excavation_area_pub_->publish(area_msg);
        
        publishExcavationMarkers(excavation_center, yaw, msg->header.stamp);
    }
    
    bool isInsideExcavationArea(double x_local, double y_local, double z_relative, 
                               const std::vector<ExcavationBox>& boxes) {
        if (z_relative < -depth_ || z_relative > 0) return false;
        
        double slope_offset = depth_ / tan(slope_angle_rad_);
        double slope_factor = (depth_ + z_relative) / depth_;
        double current_offset = slope_offset * slope_factor;
        
        for (const auto& box : boxes) {
            double dx = x_local - box.center_x_local;
            double dy = y_local - box.center_y_local;
            
            double half_length = box.length / 2.0 + current_offset;
            double half_width = box.width / 2.0 + current_offset;
            
            if (std::abs(dx) <= half_length && std::abs(dy) <= half_width) {
                return true;
            }
        }
        
        return false;
    }
    
    void generateExcavationArea(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                                const tf2::Vector3& center,
                                double yaw,
                                const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& terrain_cloud) {
        auto boxes = getExcavationBoxes();
        double slope_offset = depth_ / tan(slope_angle_rad_);
        
        // 全領域を細かくグリッド化
        double overall_min_x = std::numeric_limits<double>::max();
        double overall_max_x = std::numeric_limits<double>::lowest();
        double overall_min_y = std::numeric_limits<double>::max();
        double overall_max_y = std::numeric_limits<double>::lowest();
        
        for (const auto& box : boxes) {
            overall_min_x = std::min(overall_min_x, box.min_x);
            overall_max_x = std::max(overall_max_x, box.max_x);
            overall_min_y = std::min(overall_min_y, box.min_y);
            overall_max_y = std::max(overall_max_y, box.max_y);
        }
        
        int n_x = static_cast<int>((overall_max_x - overall_min_x) / point_density_) + 1;
        int n_y = static_cast<int>((overall_max_y - overall_min_y) / point_density_) + 1;
        int n_depth = static_cast<int>(depth_ / point_density_);
        
        for (int i = 0; i <= n_x; ++i) {
            for (int j = 0; j <= n_y; ++j) {
                double x_local = overall_min_x + i * point_density_;
                double y_local = overall_min_y + j * point_density_;
                
                // この点がL字領域内にあるかチェック
                if (!isInsideAnyBox(x_local, y_local, boxes)) {
                    continue;
                }
                
                double x_global = center.x() + x_local * cos(yaw) - y_local * sin(yaw);
                double y_global = center.y() + x_local * sin(yaw) + y_local * cos(yaw);
                
                double local_terrain_height = getTerrainHeight(terrain_cloud, x_global, y_global);
                
                // 底面の点を追加
                pcl::PointXYZRGB bottom_point;
                bottom_point.x = x_global;
                bottom_point.y = y_global;
                bottom_point.z = local_terrain_height - depth_;
                bottom_point.r = 255; bottom_point.g = 255; bottom_point.b = 0;
                cloud->push_back(bottom_point);
                
                // 外壁エッジの場合のみ斜面を生成
                if (isOuterEdge(x_local, y_local, boxes, point_density_)) {
                    for (int k = 1; k < n_depth; ++k) {
                        double z_ratio = static_cast<double>(k) / n_depth;
                        double z_local = local_terrain_height - depth_ + k * point_density_;
                        
                        pcl::PointXYZRGB slope_point;
                        slope_point.x = x_global;
                        slope_point.y = y_global;
                        slope_point.z = z_local;
                        slope_point.r = 200; slope_point.g = 200; slope_point.b = 0;
                        cloud->push_back(slope_point);
                    }
                }
            }
        }
    }
    
    void processExcavation(pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud,
                           pcl::PointCloud<pcl::PointXYZRGB>::Ptr output_cloud,
                           const tf2::Vector3& center,
                           double yaw) {
        auto boxes = getExcavationBoxes();
        
        for (const auto& point : input_cloud->points) {
            double dx = point.x - center.x();
            double dy = point.y - center.y();
            
            double x_local = dx * cos(-yaw) - dy * sin(-yaw);
            double y_local = dx * sin(-yaw) + dy * cos(-yaw);
            
            double local_terrain_height = getTerrainHeight(input_cloud, point.x, point.y);
            double z_relative_to_terrain = point.z - local_terrain_height;
            
            bool inside = isInsideExcavationArea(x_local, y_local, z_relative_to_terrain, boxes);
            
            if (!inside) {
                output_cloud->push_back(point);
            }
        }
        
        generateExcavatedSurface(output_cloud, center, yaw, input_cloud);
        
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
            "Excavation processed - Input: %zu, Output: %zu points",
            input_cloud->size(), output_cloud->size());
    }
    
    void generateExcavatedSurface(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                                  const tf2::Vector3& center,
                                  double yaw,
                                  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& terrain_cloud) {
        auto boxes = getExcavationBoxes();
        double slope_offset = depth_ / tan(slope_angle_rad_);
        
        // 全領域の範囲を計算
        double overall_min_x = std::numeric_limits<double>::max();
        double overall_max_x = std::numeric_limits<double>::lowest();
        double overall_min_y = std::numeric_limits<double>::max();
        double overall_max_y = std::numeric_limits<double>::lowest();
        
        for (const auto& box : boxes) {
            overall_min_x = std::min(overall_min_x, box.min_x);
            overall_max_x = std::max(overall_max_x, box.max_x);
            overall_min_y = std::min(overall_min_y, box.min_y);
            overall_max_y = std::max(overall_max_y, box.max_y);
        }
        
        int n_x = static_cast<int>((overall_max_x - overall_min_x) / point_density_) + 1;
        int n_y = static_cast<int>((overall_max_y - overall_min_y) / point_density_) + 1;
        
        // 底面を生成
        for (int i = 0; i <= n_x; ++i) {
            for (int j = 0; j <= n_y; ++j) {
                double x_local = overall_min_x + i * point_density_;
                double y_local = overall_min_y + j * point_density_;
                
                if (!isInsideAnyBox(x_local, y_local, boxes)) {
                    continue;
                }
                
                double x_global = center.x() + x_local * cos(yaw) - y_local * sin(yaw);
                double y_global = center.y() + x_local * sin(yaw) + y_local * cos(yaw);
                
                double local_terrain_height = getTerrainHeight(terrain_cloud, x_global, y_global);
                
                pcl::PointXYZRGB bottom_point;
                bottom_point.x = x_global;
                bottom_point.y = y_global;
                bottom_point.z = local_terrain_height - depth_;
                bottom_point.r = 0; bottom_point.g = 139; bottom_point.b = 0;
                cloud->push_back(bottom_point);
            }
        }
        
        // 外壁の斜面を生成
        int n_slope = static_cast<int>(slope_offset / point_density_) + 1;
        
        for (int i = 0; i <= n_x; ++i) {
            for (int j = 0; j <= n_y; ++j) {
                double x_local = overall_min_x + i * point_density_;
                double y_local = overall_min_y + j * point_density_;
                
                // 外壁エッジのみ斜面を生成
                if (!isOuterEdge(x_local, y_local, boxes, point_density_)) {
                    continue;
                }
                
                for (int k = 0; k <= n_slope; ++k) {
                    double z_ratio = static_cast<double>(k) / n_slope;
                    double offset = slope_offset * z_ratio;
                    
                    // 外側方向を計算（簡易版：4方向チェック）
                    double offset_x = 0.0, offset_y = 0.0;
                    
                    if (!isInsideAnyBox(x_local + point_density_, y_local, boxes)) {
                        offset_x = offset;
                    } else if (!isInsideAnyBox(x_local - point_density_, y_local, boxes)) {
                        offset_x = -offset;
                    }
                    
                    if (!isInsideAnyBox(x_local, y_local + point_density_, boxes)) {
                        offset_y = offset;
                    } else if (!isInsideAnyBox(x_local, y_local - point_density_, boxes)) {
                        offset_y = -offset;
                    }
                    
                    double x_slope = x_local + offset_x;
                    double y_slope = y_local + offset_y;
                    
                    double x_global = center.x() + x_slope * cos(yaw) - y_slope * sin(yaw);
                    double y_global = center.y() + x_slope * sin(yaw) + y_slope * cos(yaw);
                    
                    double local_terrain_height = getTerrainHeight(terrain_cloud, x_global, y_global);
                    double z_local = local_terrain_height - depth_ * (1.0 - z_ratio);
                    
                    pcl::PointXYZRGB slope_point;
                    slope_point.x = x_global;
                    slope_point.y = y_global;
                    slope_point.z = z_local;
                    slope_point.r = 144; slope_point.g = 238; slope_point.b = 144;
                    cloud->push_back(slope_point);
                }
            }
        }
    }
    
    void publishExcavationMarkers(const tf2::Vector3& center, double yaw, rclcpp::Time stamp) {
        auto boxes = getExcavationBoxes();
        visualization_msgs::msg::MarkerArray marker_array;
        
        int id = 0;
        for (const auto& box : boxes) {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "map";
            marker.header.stamp = stamp;
            marker.ns = "excavation";
            marker.id = id++;
            marker.type = visualization_msgs::msg::Marker::CUBE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            
            double x_global = center.x() + box.center_x_local * cos(yaw) - box.center_y_local * sin(yaw);
            double y_global = center.y() + box.center_x_local * sin(yaw) + box.center_y_local * cos(yaw);
            
            marker.pose.position.x = x_global;
            marker.pose.position.y = y_global;
            marker.pose.position.z = center.z() - depth_/2;
            
            tf2::Quaternion q;
            q.setRPY(0, 0, yaw);
            marker.pose.orientation.x = q.x();
            marker.pose.orientation.y = q.y();
            marker.pose.orientation.z = q.z();
            marker.pose.orientation.w = q.w();
            
            marker.scale.x = box.length;
            marker.scale.y = box.width;
            marker.scale.z = depth_;
            
            marker.color.r = 0.5;
            marker.color.g = 0.25;
            marker.color.b = 0.0;
            marker.color.a = 0.3;
            
            marker.lifetime = rclcpp::Duration::from_seconds(0.5);
            
            marker_array.markers.push_back(marker);
        }
        
        excavation_marker_pub_->publish(marker_array);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ExcavationTerrainGenerator>());
    rclcpp::shutdown();
    return 0;
}