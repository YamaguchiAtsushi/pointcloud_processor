#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <vector>
#include <limits>
#include <cmath>

class SimplifiedScanMatcher : public rclcpp::Node
{
public:
    SimplifiedScanMatcher() : Node("simplified_scan_matcher")
    {
        // パラメータの宣言
        this->declare_parameter("grid_resolution", 1.0);  // グリッドの解像度 (m)
        this->declare_parameter("map_width", 100.0);      // マップの幅 (m)
        this->declare_parameter("map_height", 100.0);     // マップの高さ (m)
        this->declare_parameter("max_gradient", 0.3);     // 最大勾配 (走行可能な限界)
        this->declare_parameter("min_points_per_cell", 10); // セル内の最小点数
        this->declare_parameter("start_clear_radius", 3.0);  // スタート地点周辺のクリア半径 (m)
        
        grid_resolution_ = this->get_parameter("grid_resolution").as_double();
        map_width_ = this->get_parameter("map_width").as_double();
        map_height_ = this->get_parameter("map_height").as_double();
        max_gradient_ = this->get_parameter("max_gradient").as_double();
        min_points_per_cell_ = this->get_parameter("min_points_per_cell").as_int();
        start_clear_radius_ = this->get_parameter("start_clear_radius").as_double();
        
        // グリッドのサイズを計算
        grid_width_ = static_cast<int>(map_width_ / grid_resolution_);
        grid_height_ = static_cast<int>(map_height_ / grid_resolution_);
        
        // TF2の初期化
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        
        // 初回のロボット位置を取得（スタート地点として記録）
        start_position_initialized_ = false;
        
        // Subscriber
        robot_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/four_wheel_robot/filtered_points", 1,
            std::bind(&SimplifiedScanMatcher::robotCloudCallback, this, std::placeholders::_1));
        
        // Publisher
        occupancy_grid_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
            "/occupancy_grid", 10);
        
        RCLCPP_INFO(this->get_logger(), "Occupancy Grid Map Generator initialized");
        RCLCPP_INFO(this->get_logger(), "Grid size: %d x %d, Resolution: %.2f m", 
                    grid_width_, grid_height_, grid_resolution_);
        RCLCPP_INFO(this->get_logger(), "Start clear radius: %.2f m", start_clear_radius_);
    }

private:
    // グリッドセルの情報を保持する構造体
    struct GridCell
    {
        std::vector<float> z_values;  // セル内の点のZ座標
        bool has_points = false;
    };
    
    void robotCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // 点群をmapフレームに変換
        sensor_msgs::msg::PointCloud2 transformed_cloud;
        try
        {
            // TF変換が利用可能か確認
            std::string target_frame = "map";
            if (!tf_buffer_->canTransform(target_frame, msg->header.frame_id, 
                                         tf2::TimePointZero, tf2::durationFromSec(0.5)))
            {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                    "Transform from %s to %s not available yet", 
                    msg->header.frame_id.c_str(), target_frame.c_str());
                return;
            }
            
            // 点群をmapフレームに変換
            auto transform = tf_buffer_->lookupTransform(
                target_frame, msg->header.frame_id, 
                tf2::TimePointZero, tf2::durationFromSec(0.5));
            
            tf2::doTransform(*msg, transformed_cloud, transform);
            
            RCLCPP_DEBUG(this->get_logger(), "Transformed point cloud from %s to %s",
                        msg->header.frame_id.c_str(), target_frame.c_str());
        }
        catch (tf2::TransformException& ex)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "Could not transform point cloud: %s", ex.what());
            return;
        }
        
        // PointCloud2をPCLに変換
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(transformed_cloud, *cloud);
        
        if (cloud->points.empty())
        {
            RCLCPP_WARN(this->get_logger(), "Received empty point cloud");
            return;
        }
        
        // 点群の中心位置を計算（マップ原点として使用）
        geometry_msgs::msg::TransformStamped robot_transform;
        try
        {
            robot_transform = tf_buffer_->lookupTransform(
                "map", "four_wheel_robot/base_link",
                tf2::TimePointZero, tf2::durationFromSec(0.1));
        }
        catch (tf2::TransformException& ex)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "Could not get robot transform: %s", ex.what());
            return;
        }
        
        double robot_x = robot_transform.transform.translation.x;
        double robot_y = robot_transform.transform.translation.y;
        
        // 初回のロボット位置をスタート地点として記録
        if (!start_position_initialized_)
        {
            start_x_ = robot_x;
            start_y_ = robot_y;
            start_position_initialized_ = true;
            RCLCPP_INFO(this->get_logger(), "Start position set at (%.2f, %.2f)", start_x_, start_y_);
        }
        
        // マップの原点をロボット位置中心に設定
        double map_origin_x = robot_x - map_width_ / 2.0;
        double map_origin_y = robot_y - map_height_ / 2.0;
        
        // グリッドマップの初期化
        std::vector<GridCell> grid_map(grid_width_ * grid_height_);
        
        // 点群をグリッドに分類
        for (const auto& point : cloud->points)
        {
            // 無効な点をスキップ
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
                continue;
            
            // map座標をグリッド座標に変換
            int grid_x = static_cast<int>((point.x - map_origin_x) / grid_resolution_);
            int grid_y = static_cast<int>((point.y - map_origin_y) / grid_resolution_);
            
            // グリッド範囲内かチェック
            if (grid_x >= 0 && grid_x < grid_width_ && grid_y >= 0 && grid_y < grid_height_)
            {
                int index = grid_y * grid_width_ + grid_x;
                grid_map[index].z_values.push_back(point.z);
                grid_map[index].has_points = true;
            }
        }
        
        // Occupancy Gridを生成
        nav_msgs::msg::OccupancyGrid occupancy_grid;
        occupancy_grid.header.stamp = this->now();
        occupancy_grid.header.frame_id = "map";
        
        occupancy_grid.info.resolution = grid_resolution_;
        occupancy_grid.info.width = grid_width_;
        occupancy_grid.info.height = grid_height_;
        occupancy_grid.info.origin.position.x = map_origin_x;
        occupancy_grid.info.origin.position.y = map_origin_y;
        occupancy_grid.info.origin.position.z = 0.0;
        occupancy_grid.info.origin.orientation.w = 1.0;
        
        occupancy_grid.data.resize(grid_width_ * grid_height_);
        
        // 各セルの勾配を計算して占有率を決定
        for (int y = 0; y < grid_height_; ++y)
        {
            for (int x = 0; x < grid_width_; ++x)
            {
                int index = y * grid_width_ + x;
                
                // セルの実際のmap座標を計算
                double cell_x = map_origin_x + (x + 0.5) * grid_resolution_;
                double cell_y = map_origin_y + (y + 0.5) * grid_resolution_;
                
                // スタート地点からの距離を計算
                double dist_to_start = std::sqrt(
                    std::pow(cell_x - start_x_, 2) + std::pow(cell_y - start_y_, 2)
                );
                
                // スタート地点周辺は強制的に自由空間（0）に設定
                if (dist_to_start <= start_clear_radius_)
                {
                    occupancy_grid.data[index] = 0;// 走行可能 (自由空間)
                }
                else if (!grid_map[index].has_points || 
                         grid_map[index].z_values.size() < static_cast<size_t>(min_points_per_cell_))// 点群が少ない場合
                {
                    // 未知領域
                    occupancy_grid.data[index] = -1;
                }
                else
                {
                    // セル内の勾配を計算
                    float gradient = calculateGradient(grid_map[index].z_values);
                    
                    if (gradient > max_gradient_)
                    {
                        // 走行不可 (障害物) - 確実に100に設定
                        occupancy_grid.data[index] = 100;
                    }
                    else
                    {
                        // 走行可能 (自由空間)
                        occupancy_grid.data[index] = 0;
                    }
                }
            }
        }
        
        // パブリッシュ
        occupancy_grid_pub_->publish(occupancy_grid);
        
        RCLCPP_DEBUG(this->get_logger(), "Published occupancy grid at (%.2f, %.2f) with %zu points", 
                     robot_x, robot_y, cloud->points.size());
    }
    
    // セル内の点群から勾配を計算
    float calculateGradient(const std::vector<float>& z_values)
    {
        if (z_values.size() < 2)
            return 0.0f;
        
        // 最大値と最小値の差を勾配として使用
        float min_z = *std::min_element(z_values.begin(), z_values.end());
        float max_z = *std::max_element(z_values.begin(), z_values.end());
        float gradient = (max_z - min_z) / grid_resolution_;
        
        return gradient;
    }
    
    // メンバ変数
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr robot_cloud_sub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr occupancy_grid_pub_;
    
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    double grid_resolution_;
    double map_width_;
    double map_height_;
    double max_gradient_;
    int min_points_per_cell_;
    int grid_width_;
    int grid_height_;
    
    // スタート地点関連
    double start_clear_radius_;  // スタート地点周辺のクリア半径
    double start_x_;             // スタート地点のX座標
    double start_y_;             // スタート地点のY座標
    bool start_position_initialized_;  // スタート位置が初期化されたか
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SimplifiedScanMatcher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}