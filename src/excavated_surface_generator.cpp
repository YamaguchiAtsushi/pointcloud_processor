#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>  // For getYaw function
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_box.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <visualization_msgs/msg/marker.hpp>
#include <cmath>
#include <algorithm>

class ExcavationTerrainGenerator : public rclcpp::Node {
public:
    ExcavationTerrainGenerator() : Node("excavation_terrain_generator") {
        // Declare parameters for excavation geometry
        this->declare_parameter("excavation.width", 1.2);          // 掘削幅（バケット幅）
        this->declare_parameter("excavation.length", 1.8);         // 掘削長さ
        this->declare_parameter("excavation.depth", 1.0);          // 掘削深さ
        this->declare_parameter("excavation.slope_angle", 75.0);   // 斜面角度（度）
        this->declare_parameter("excavation.offset_x", 5.0);       // バックホウ前方のオフセット
        this->declare_parameter("excavation.offset_y", 0.0);       // 横方向オフセット
        this->declare_parameter("excavation.point_density", 0.05); // 点群の密度（メートル）
        this->declare_parameter("excavation.enabled", true);       // 掘削シミュレーションの有効/無効
        this->declare_parameter("excavation.terrain_search_radius", 0.5); // 地形高度検索半径
        
        // Get parameters
        updateParameters();
        
        // TF2 Buffer and Listener
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        
        // Subscriber for matched point cloud
        matched_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/matched_point_cloud", 10,
            std::bind(&ExcavationTerrainGenerator::matchedCloudCallback, this, std::placeholders::_1));
        
        // Publishers
        excavated_terrain_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/excavated_terrain", 10);
        excavation_area_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/excavation_area", 10);
        excavation_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
            "/excavation_marker", 10);
        
        // Parameter update timer
        param_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&ExcavationTerrainGenerator::updateParameters, this));
            
        RCLCPP_INFO(this->get_logger(), "Excavation Terrain Generator initialized");
        RCLCPP_INFO(this->get_logger(), 
            "Excavation params - Width: %.2fm, Length: %.2fm, Depth: %.2fm, Slope: %.1f°",
            width_, length_, depth_, slope_angle_deg_);
    }
    
private:
    // TF related
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    // Subscribers and Publishers
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr matched_cloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr excavated_terrain_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr excavation_area_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr excavation_marker_pub_;
    
    // Timer
    rclcpp::TimerBase::SharedPtr param_timer_;
    
    // Excavation parameters
    double width_;           // 掘削幅
    double length_;          // 掘削長さ  
    double depth_;           // 掘削深さ
    double slope_angle_deg_; // 斜面角度（度）
    double slope_angle_rad_; // 斜面角度（ラジアン）
    double offset_x_;        // X方向オフセット
    double offset_y_;        // Y方向オフセット
    double point_density_;   // 点群密度
    bool enabled_;           // 掘削有効/無効
    double terrain_search_radius_; // 地形高度検索半径
    
    void updateParameters() {
        width_ = this->get_parameter("excavation.width").as_double();
        length_ = this->get_parameter("excavation.length").as_double();
        depth_ = this->get_parameter("excavation.depth").as_double();
        slope_angle_deg_ = this->get_parameter("excavation.slope_angle").as_double();
        slope_angle_rad_ = slope_angle_deg_ * M_PI / 180.0;
        offset_x_ = this->get_parameter("excavation.offset_x").as_double();
        offset_y_ = this->get_parameter("excavation.offset_y").as_double();
        point_density_ = this->get_parameter("excavation.point_density").as_double();
        enabled_ = this->get_parameter("excavation.enabled").as_bool();
        terrain_search_radius_ = this->get_parameter("excavation.terrain_search_radius").as_double();
    }
    
    // 地形の指定した位置の半径1mの平均値の高度を取得する関数
    double getTerrainHeight(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
                           double x, double y) {
        if (cloud->empty()) return 0.0;
        
        // KdTreeを使って最近傍点を探索
        pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
        kdtree.setInputCloud(cloud);
        
        pcl::PointXYZRGB search_point;
        search_point.x = x;
        search_point.y = y;
        search_point.z = 0; // Z座標は無視
        
        // 半径内の点を検索
        std::vector<int> point_indices;
        std::vector<float> point_distances;
        
        if (kdtree.radiusSearch(search_point, terrain_search_radius_, 
                               point_indices, point_distances) > 0) {
            // 検索された点のZ座標の平均を返す
            double sum_z = 0.0;
            int valid_count = 0;
            
            for (int idx : point_indices) {
                // XY距離をチェック（Z座標を除いた2D距離）
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
        
        // 見つからない場合は最近傍点のZ座標を使用
        std::vector<int> k_indices(1);
        std::vector<float> k_distances(1);
        if (kdtree.nearestKSearch(search_point, 1, k_indices, k_distances) > 0) {
            return cloud->points[k_indices[0]].z;
        }
        
        return 0.0; // デフォルト値
    }
    
    void matchedCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        if (!enabled_) {
            // 掘削が無効の場合は元の点群をそのまま転送
            excavated_terrain_pub_->publish(*msg);
            return;
        }
        
        // Get zx120/base_link to map transform
        geometry_msgs::msg::TransformStamped zx120_transform;
        try {
            zx120_transform = tf_buffer_->lookupTransform(
                "map", "zx120/base_link", 
                tf2::TimePointZero, tf2::durationFromSec(0.1));
        } catch (tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "Could not get zx120 transform: %s", ex.what());
            // Transform取得失敗時は元の点群をそのまま転送
            excavated_terrain_pub_->publish(*msg);
            return;
        }
        
        // Convert to PCL
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::fromROSMsg(*msg, *input_cloud);
        
        // Calculate excavation area in map frame
        tf2::Transform zx120_tf;
        tf2::fromMsg(zx120_transform.transform, zx120_tf);
        
        // 掘削領域の中心位置（zx120/base_link座標系）
        tf2::Vector3 excavation_center_local(offset_x_, offset_y_, 0);
        tf2::Vector3 excavation_center_2d = zx120_tf * excavation_center_local;
        
        // 掘削中心位置の地形高度を取得
        double terrain_height = getTerrainHeight(input_cloud, 
                                               excavation_center_2d.x(), 
                                               excavation_center_2d.y());
        
        // 掘削領域の実際の中心位置（地形の高さを考慮）
        tf2::Vector3 excavation_center(excavation_center_2d.x(), 
                                      excavation_center_2d.y(), 
                                      terrain_height);
        
        // 掘削領域の向き（zx120の向き）
        tf2::Quaternion zx120_rotation = zx120_tf.getRotation();
        // Manually calculate yaw from quaternion
        tf2::Matrix3x3 m(zx120_rotation);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);
        
        // Generate excavation area visualization
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr excavation_area(new pcl::PointCloud<pcl::PointXYZRGB>);
        generateExcavationArea(excavation_area, excavation_center, yaw, input_cloud);
        
        // Remove points inside excavation area and add excavated terrain
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr result_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        processExcavation(input_cloud, result_cloud, excavation_center, yaw);
        
        // Publish excavated terrain
        sensor_msgs::msg::PointCloud2 output_msg;
        pcl::toROSMsg(*result_cloud, output_msg);
        output_msg.header = msg->header;
        output_msg.header.frame_id = "map";
        excavated_terrain_pub_->publish(output_msg);
        
        // Publish excavation area visualization
        sensor_msgs::msg::PointCloud2 area_msg;
        pcl::toROSMsg(*excavation_area, area_msg);
        area_msg.header = msg->header;
        area_msg.header.frame_id = "map";
        excavation_area_pub_->publish(area_msg);
        
        // Publish excavation marker
        publishExcavationMarker(excavation_center, yaw);
    }
    
    void generateExcavationArea(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                                const tf2::Vector3& center,
                                double yaw,
                                const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& terrain_cloud) {
        // 斜面のオフセット（上部が広がる分）
        // 掘削深さと斜面角度から「斜面が上に行くほどどれだけ広がるか」を計算
        double slope_offset = depth_ / tan(slope_angle_rad_);
        
        // 掘削領域の点群を生成（台形断面 - 上が広く、下が狭い）
        // 掘削領域を点群で「どれくらい細かく」作るかを決定、point_density_ が 0.05m なら、50cm の幅は 10分割される
        int n_length = static_cast<int>(length_ / point_density_);
        int n_width = static_cast<int>(width_ / point_density_);
        int n_depth = static_cast<int>(depth_ / point_density_);
        
        for (int i = 0; i <= n_length; ++i) {
            double x_local = -length_/2 + i * point_density_;
            
            for (int j = 0; j <= n_width; ++j) {
                double y_local = -width_/2 + j * point_density_;
                
                // グローバル座標での位置を計算
                double x_global = center.x() + x_local * cos(yaw) - y_local * sin(yaw);
                double y_global = center.y() + x_local * sin(yaw) + y_local * cos(yaw);
                
                // その位置の地形高度を取得
                double local_terrain_height = getTerrainHeight(terrain_cloud, x_global, y_global);
                
                // 底面の点を追加（地形の高さから掘削深度分下）
                pcl::PointXYZRGB bottom_point;
                bottom_point.x = x_global;
                bottom_point.y = y_global;
                bottom_point.z = local_terrain_height - depth_;
                bottom_point.r = 255; bottom_point.g = 255; bottom_point.b = 0; // Yellow
                cloud->push_back(bottom_point);
                
                // 斜面の点を追加（4辺）
                for (int k = 1; k < n_depth; ++k) {
                    double z_ratio = static_cast<double>(k) / n_depth; // 0 at bottom, 1 at top
                    double z_local = local_terrain_height - depth_ + k * point_density_;
                    double slope_factor = z_ratio; // 0 at bottom, 1 at top
                    
                    // Front and back slopes (along length)
                    if (i == 0 || i == n_length) {
                        double x_slope_local = (i == 0) ? 
                            x_local - slope_offset * slope_factor :  // 前方斜面は外側へ
                            x_local + slope_offset * slope_factor;   // 後方斜面は外側へ
                        
                        pcl::PointXYZRGB slope_point;
                        slope_point.x = center.x() + x_slope_local * cos(yaw) - y_local * sin(yaw);
                        slope_point.y = center.y() + x_slope_local * sin(yaw) + y_local * cos(yaw);
                        slope_point.z = z_local;
                        slope_point.r = 200; slope_point.g = 200; slope_point.b = 0;
                        cloud->push_back(slope_point);
                    }
                    
                    // Side slopes (along width)
                    if (j == 0 || j == n_width) {
                        double y_slope_local = (j == 0) ?
                            y_local - slope_offset * slope_factor :  // 左斜面は外側へ
                            y_local + slope_offset * slope_factor;   // 右斜面は外側へ
                        
                        pcl::PointXYZRGB slope_point;
                        slope_point.x = center.x() + x_local * cos(yaw) - y_slope_local * sin(yaw);
                        slope_point.y = center.y() + x_local * sin(yaw) + y_slope_local * cos(yaw);
                        slope_point.z = z_local;
                        slope_point.r = 200; slope_point.g = 200; slope_point.b = 0;
                        cloud->push_back(slope_point);
                    }
                }
            }
        }
    }
    
    // 指定された掘削領域を点群から削除し、掘削後の地形を生成する処理
    void processExcavation(pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud,
                           pcl::PointCloud<pcl::PointXYZRGB>::Ptr output_cloud,
                           const tf2::Vector3& center,
                           double yaw) {
        double slope_offset = depth_ / tan(slope_angle_rad_);
        
        // Process each point
        for (const auto& point : input_cloud->points) {
            // Transform point to excavation local coordinates
            double dx = point.x - center.x();
            double dy = point.y - center.y();
            
            // Rotate to align with excavation
            double x_local = dx * cos(-yaw) - dy * sin(-yaw);
            double y_local = dx * sin(-yaw) + dy * cos(-yaw);
            
            // その位置の地形高度を取得
            double local_terrain_height = getTerrainHeight(input_cloud, point.x, point.y);
            double z_relative_to_terrain = point.z - local_terrain_height;
            
            // Check if point is inside excavation area (trapezoid check - 上が広く、下が狭い)
            bool inside_excavation = false;
            
            if (z_relative_to_terrain >= -depth_ && z_relative_to_terrain <= 0) {
                // Calculate boundaries at this depth (上部ほど広い)
                double slope_factor = (depth_ + z_relative_to_terrain) / depth_; // 0 at bottom, 1 at top
                double current_offset = slope_offset * slope_factor;
                
                double x_min = -length_/2 - current_offset;
                double x_max = length_/2 + current_offset;
                double y_min = -width_/2 - current_offset;
                double y_max = width_/2 + current_offset;
                
                if (x_local >= x_min && x_local <= x_max &&
                    y_local >= y_min && y_local <= y_max) {
                    inside_excavation = true;
                }
            }
            
            // Keep point if outside excavation area
            if (!inside_excavation) {
                output_cloud->push_back(point);
            }
        }
        
        // Add excavated terrain surface (sloped walls and bottom)
        generateExcavatedSurface(output_cloud, center, yaw, input_cloud);
        
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
            "Processed excavation - Input: %zu points, Output: %zu points",
            input_cloud->size(), output_cloud->size());
    }
    
    void generateExcavatedSurface(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                                  const tf2::Vector3& center,
                                  double yaw,
                                  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& terrain_cloud) {
        double slope_offset = depth_ / tan(slope_angle_rad_);
        
        // Generate dense surface mesh for excavated area
        int n_length = static_cast<int>(length_ / point_density_);
        int n_width = static_cast<int>(width_ / point_density_);
        int n_slope = static_cast<int>(slope_offset / point_density_) + 1;
        
        // Bottom surface (最も狭い部分)
        for (int i = 0; i <= n_length; ++i) {
            for (int j = 0; j <= n_width; ++j) {
                double x_local = -length_/2 + i * point_density_;
                double y_local = -width_/2 + j * point_density_;
                
                // グローバル座標での位置
                double x_global = center.x() + x_local * cos(yaw) - y_local * sin(yaw);
                double y_global = center.y() + x_local * sin(yaw) + y_local * cos(yaw);
                
                // その位置の地形高度を取得
                double local_terrain_height = getTerrainHeight(terrain_cloud, x_global, y_global);
                
                pcl::PointXYZRGB bottom_point;
                bottom_point.x = x_global;
                bottom_point.y = y_global;
                bottom_point.z = local_terrain_height - depth_;
                bottom_point.r = 0; bottom_point.g = 139; bottom_point.b = 0; // Green
                cloud->push_back(bottom_point);
            }
        }
        
        // Front slope (前方斜面 - 底面から地表面へ)
        for (int i = 0; i <= n_slope; ++i) {
            double z_ratio = static_cast<double>(i) / n_slope;  // 0 at bottom, 1 at top
            double x_offset = slope_offset * z_ratio;    // 0 at bottom, slope_offset at top
            
            for (int j = 0; j <= n_width; ++j) {
                double y_ratio = static_cast<double>(j) / n_width;
                double y_local = -width_/2 + width_ * y_ratio;
                double x_local = -length_/2 - x_offset;
                
                // グローバル座標での位置
                double x_global = center.x() + x_local * cos(yaw) - y_local * sin(yaw);
                double y_global = center.y() + x_local * sin(yaw) + y_local * cos(yaw);
                
                // その位置の地形高度を取得
                double local_terrain_height = getTerrainHeight(terrain_cloud, x_global, y_global);
                double z_local = local_terrain_height - depth_ * (1.0 - z_ratio);
                
                pcl::PointXYZRGB slope_point;
                slope_point.x = x_global;
                slope_point.y = y_global;
                slope_point.z = z_local;
                slope_point.r = 144; slope_point.g = 238; slope_point.b = 144; // Dark Green
                cloud->push_back(slope_point);
            }
        }
        
        // Back slope (後方斜面 - 底面から地表面へ)
        for (int i = 0; i <= n_slope; ++i) {
            double z_ratio = static_cast<double>(i) / n_slope;
            double x_offset = slope_offset * z_ratio;
            
            for (int j = 0; j <= n_width; ++j) {
                double y_ratio = static_cast<double>(j) / n_width;
                double y_local = -width_/2 + width_ * y_ratio;
                double x_local = length_/2 + x_offset;
                
                // グローバル座標での位置
                double x_global = center.x() + x_local * cos(yaw) - y_local * sin(yaw);
                double y_global = center.y() + x_local * sin(yaw) + y_local * cos(yaw);
                
                // その位置の地形高度を取得
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
        
        // Left slope (左側斜面 - 底面から地表面へ)
        for (int i = 0; i <= n_slope; ++i) {
            double z_ratio = static_cast<double>(i) / n_slope;
            double y_offset = slope_offset * z_ratio;
            
            for (int j = 0; j <= n_length; ++j) {
                double x_ratio = static_cast<double>(j) / n_length;
                double x_local = -length_/2 + length_ * x_ratio;
                double y_local = -width_/2 - y_offset;
                
                // グローバル座標での位置
                double x_global = center.x() + x_local * cos(yaw) - y_local * sin(yaw);
                double y_global = center.y() + x_local * sin(yaw) + y_local * cos(yaw);
                
                // その位置の地形高度を取得
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
        
        // Right slope (右側斜面 - 底面から地表面へ)
        for (int i = 0; i <= n_slope; ++i) {
            double z_ratio = static_cast<double>(i) / n_slope;
            double y_offset = slope_offset * z_ratio;
            
            for (int j = 0; j <= n_length; ++j) {
                double x_ratio = static_cast<double>(j) / n_length;
                double x_local = -length_/2 + length_ * x_ratio;
                double y_local = width_/2 + y_offset;
                
                // グローバル座標での位置
                double x_global = center.x() + x_local * cos(yaw) - y_local * sin(yaw);
                double y_global = center.y() + x_local * sin(yaw) + y_local * cos(yaw);
                
                // その位置の地形高度を取得
                double local_terrain_height = getTerrainHeight(terrain_cloud, x_global, y_global);
                double z_local = local_terrain_height - depth_ * (1.0 - z_ratio);
                
                pcl::PointXYZRGB slope_point;
                slope_point.x = x_global;
                slope_point.y = y_global;
                slope_point.z = z_local;
                slope_point.r = 0; slope_point.g = 100; slope_point.b = 0;
                cloud->push_back(slope_point);
            }
        }
    }
    
    void publishExcavationMarker(const tf2::Vector3& center, double yaw) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = this->now();
        marker.ns = "excavation";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::CUBE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        // Position
        marker.pose.position.x = center.x();
        marker.pose.position.y = center.y();
        marker.pose.position.z = center.z() - depth_/2;
        
        // Orientation
        tf2::Quaternion q;
        q.setRPY(0, 0, yaw);
        marker.pose.orientation.x = q.x();
        marker.pose.orientation.y = q.y();
        marker.pose.orientation.z = q.z();
        marker.pose.orientation.w = q.w();
        
        // Scale
        marker.scale.x = length_;
        marker.scale.y = width_;
        marker.scale.z = depth_;
        
        // Color (semi-transparent brown)
        marker.color.r = 0.5;
        marker.color.g = 0.25;
        marker.color.b = 0.0;
        marker.color.a = 0.3;
        
        marker.lifetime = rclcpp::Duration::from_seconds(0.5);
        
        excavation_marker_pub_->publish(marker);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ExcavationTerrainGenerator>());
    rclcpp::shutdown();
    return 0;
}