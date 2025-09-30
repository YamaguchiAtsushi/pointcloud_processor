#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <geometry_msgs/msg/quaternion_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/gicp.h>
#include <GeographicLib/LocalCartesian.hpp>

class GnssGicpMatcher : public rclcpp::Node {
public:
    GnssGicpMatcher() : Node("gnss_gicp_matcher") {
        // TF2 Buffer and Listener
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        
        // Initialize local cartesian converter (origin will be set at first GNSS fix)
        origin_set_ = false;
        
        // Subscribers
        robot_gnss_sub_ = this->create_subscription<sensor_msgs::msg::NavSatFix>(
            "/four_wheel_robot/gnss_compass_front/fix", 10,
            std::bind(&GnssGicpMatcher::robotGnssCallback, this, std::placeholders::_1));
            
        robot_heading_sub_ = this->create_subscription<geometry_msgs::msg::QuaternionStamped>(
            "/four_wheel_robot/gnss_compass/calc_heading", 10,
            std::bind(&GnssGicpMatcher::robotHeadingCallback, this, std::placeholders::_1));
            
        backhoe_gnss_sub_ = this->create_subscription<sensor_msgs::msg::NavSatFix>(
            "/zx120/gnss_compass_front/fix", 10,
            std::bind(&GnssGicpMatcher::backhoeGnssCallback, this, std::placeholders::_1));
            
        backhoe_heading_sub_ = this->create_subscription<geometry_msgs::msg::QuaternionStamped>(
            "/zx120/gnss_compass/calc_heading", 10,
            std::bind(&GnssGicpMatcher::backhoeHeadingCallback, this, std::placeholders::_1));
            
        robot_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/four_wheel_robot/filtered_points", 10,
            std::bind(&GnssGicpMatcher::robotCloudCallback, this, std::placeholders::_1));
            
        backhoe_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/zx120/filtered_points", 10,
            std::bind(&GnssGicpMatcher::backhoeCloudCallback, this, std::placeholders::_1));
        
        // Publishers
        matched_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/matched_point_cloud", 10);
        robot_colored_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/robot_colored_point_cloud", 10);
        backhoe_colored_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/backhoe_colored_point_cloud", 10);
        
        // Timer for periodic processing
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&GnssGicpMatcher::processPointClouds, this));
            
        RCLCPP_INFO(this->get_logger(), "GNSS GICP Matcher node initialized");
    }
    
private:
    // TF related
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    
    // Geographic conversion
    GeographicLib::LocalCartesian local_cartesian_;//GNSSを局所直交座標系（ENU座標系） に変換するためのクラス
    bool origin_set_;
    
    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr robot_gnss_sub_;
    rclcpp::Subscription<geometry_msgs::msg::QuaternionStamped>::SharedPtr robot_heading_sub_;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr backhoe_gnss_sub_;
    rclcpp::Subscription<geometry_msgs::msg::QuaternionStamped>::SharedPtr backhoe_heading_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr robot_cloud_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr backhoe_cloud_sub_;
    
    // Publishers
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr matched_cloud_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr robot_colored_cloud_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr backhoe_colored_cloud_pub_;
    
    // Timer
    rclcpp::TimerBase::SharedPtr timer_;
    
    // Latest data storage
    sensor_msgs::msg::NavSatFix::SharedPtr robot_gnss_data_;
    geometry_msgs::msg::QuaternionStamped::SharedPtr robot_heading_data_;
    sensor_msgs::msg::NavSatFix::SharedPtr backhoe_gnss_data_;
    geometry_msgs::msg::QuaternionStamped::SharedPtr backhoe_heading_data_;
    sensor_msgs::msg::PointCloud2::SharedPtr robot_cloud_data_;
    sensor_msgs::msg::PointCloud2::SharedPtr backhoe_cloud_data_;
    
    // Callbacks for GNSS data
    void robotGnssCallback(const sensor_msgs::msg::NavSatFix::SharedPtr msg) {
        robot_gnss_data_ = msg;
        
        // Set origin at first GNSS fix from zx120 (優先) or four_wheel_robot
        if (!origin_set_ && msg->status.status >= 0) {
            // zx120のGNSSデータがあればそれを原点に、なければfour_wheel_robotを原点に
            if (backhoe_gnss_data_ && backhoe_gnss_data_->status.status >= 0) {
                local_cartesian_.Reset(backhoe_gnss_data_->latitude, 
                                      backhoe_gnss_data_->longitude, 
                                      backhoe_gnss_data_->altitude);
                RCLCPP_INFO(this->get_logger(), 
                    "Origin set at zx120 position: lat=%f, lon=%f, alt=%f",
                    backhoe_gnss_data_->latitude, backhoe_gnss_data_->longitude, 
                    backhoe_gnss_data_->altitude);
            } else {
                local_cartesian_.Reset(msg->latitude, msg->longitude, msg->altitude);
                RCLCPP_INFO(this->get_logger(), 
                    "Origin set at four_wheel_robot position: lat=%f, lon=%f, alt=%f",
                    msg->latitude, msg->longitude, msg->altitude);
            }
            origin_set_ = true;
        }
        
        // Update TF from GNSS
        if (origin_set_ && robot_heading_data_) {
            publishGnssTransform("four_wheel_robot", msg, robot_heading_data_);
        }
    }
    
    void robotHeadingCallback(const geometry_msgs::msg::QuaternionStamped::SharedPtr msg) {
        robot_heading_data_ = msg;
        
        // Update TF if GNSS data is available
        if (origin_set_ && robot_gnss_data_) {
            publishGnssTransform("four_wheel_robot", robot_gnss_data_, msg);
        }
    }
    
    void backhoeGnssCallback(const sensor_msgs::msg::NavSatFix::SharedPtr msg) {
        backhoe_gnss_data_ = msg;
        
        // Set origin at zx120 position if not already set
        if (!origin_set_ && msg->status.status >= 0) {
            //zx120のGNSS位置を原点(0,0,0)に設定
            //Resetは一度だけ実行される
            local_cartesian_.Reset(msg->latitude, msg->longitude, msg->altitude);
            origin_set_ = true;
            RCLCPP_INFO(this->get_logger(), 
                "Origin set at zx120 position: lat=%f, lon=%f, alt=%f",
                msg->latitude, msg->longitude, msg->altitude);
        }
        
        // Update TF from GNSS (zx120は常に原点に固定)
        if (origin_set_ && backhoe_heading_data_) {
            publishGnssTransform("zx120", msg, backhoe_heading_data_);
        }
    }
    
    void backhoeHeadingCallback(const geometry_msgs::msg::QuaternionStamped::SharedPtr msg) {
        backhoe_heading_data_ = msg;
        
        // Update TF if GNSS data is available
        if (origin_set_ && backhoe_gnss_data_) {
            publishGnssTransform("zx120", backhoe_gnss_data_, msg);
        }
    }
    
    // Callbacks for point cloud data
    void robotCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        robot_cloud_data_ = msg;
    }
    
    void backhoeCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        backhoe_cloud_data_ = msg;
    }
    
    // Publish transform from map to base_link based on GNSS
    void publishGnssTransform(const std::string& robot_name,
                              const sensor_msgs::msg::NavSatFix::SharedPtr gnss_msg,
                              const geometry_msgs::msg::QuaternionStamped::SharedPtr heading_msg) {
        if (!origin_set_) return;
        
        geometry_msgs::msg::TransformStamped transform;
        transform.header.stamp = this->now();
        transform.header.frame_id = "map";
        transform.child_frame_id = robot_name + "/base_link";
        
        // zx120のbase_linkからgnss_linkへのオフセットを取得
        tf2::Vector3 zx120_gnss_offset(0.0, 0.0, 0.0);
        try {
            geometry_msgs::msg::TransformStamped zx120_base_to_gnss = 
                tf_buffer_->lookupTransform("zx120/base_link", "zx120/gnss_link", 
                                          tf2::TimePointZero, tf2::durationFromSec(0.1));
            
            zx120_gnss_offset.setValue(
                zx120_base_to_gnss.transform.translation.x,
                zx120_base_to_gnss.transform.translation.y,
                zx120_base_to_gnss.transform.translation.z
            );
            
            RCLCPP_INFO_ONCE(this->get_logger(), 
                "ZX120 GNSS offset from base_link: [%.3f, %.3f, %.3f]",
                zx120_gnss_offset.x(), zx120_gnss_offset.y(), zx120_gnss_offset.z());
                
        } catch (tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "Could not get zx120 base_link to gnss_link transform: %s", ex.what());
        }
        
        // zx120を常にmap原点に固定（変更なし）
        if (robot_name == "zx120") {
            // 位置: map原点に固定
            transform.transform.translation.x = 0.0;
            transform.transform.translation.y = 0.0;
            transform.transform.translation.z = 0.0;
            
            // 回転: 単位クォータニオン（回転なし）
            transform.transform.rotation.x = 0.0;
            transform.transform.rotation.y = 0.0;
            transform.transform.rotation.z = 0.0;
            transform.transform.rotation.w = 1.0;
            
            RCLCPP_INFO_ONCE(this->get_logger(), 
                "zx120/base_link fixed at map origin (0,0,0) with no rotation");
        } 
        // four_wheel_robotは相対位置を計算してzx120_gnss_offsetで補正
        else if (robot_name == "four_wheel_robot") {
            // four_wheel_robotのGNSS位置を取得
            double robot_x, robot_y, robot_z;
            // GNSSから位置をENUに変換
            // zx120を原点(0,0,0)からの相対位置として計算
            local_cartesian_.Forward(gnss_msg->latitude, gnss_msg->longitude, 
                                     gnss_msg->altitude, robot_x, robot_y, robot_z);
            
            // zx120のGNSS位置を取得（もし利用可能なら）
            double zx120_x = 0.0, zx120_y = 0.0, zx120_z = 0.0;
            if (backhoe_gnss_data_ && backhoe_gnss_data_->status.status >= 0) {
                // zx120の位置もENUに変換（使える場合）
                local_cartesian_.Forward(backhoe_gnss_data_->latitude, 
                                        backhoe_gnss_data_->longitude,
                                        backhoe_gnss_data_->altitude, 
                                        zx120_x, zx120_y, zx120_z);
            }
            
            // Get transform from gnss_link to base_link for four_wheel_robot
            geometry_msgs::msg::TransformStamped gnss_to_base_transform;
            try {
                gnss_to_base_transform = tf_buffer_->lookupTransform(
                    robot_name + "/gnss_link",
                    robot_name + "/base_link", 
                    tf2::TimePointZero);
            } catch (tf2::TransformException& ex) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                    "Could not get transform: %s", ex.what());
                return;
            }
            
            // four_wheel_robotの相対位置を計算（zx120を原点として）
            tf2::Quaternion rotation(heading_msg->quaternion.x, heading_msg->quaternion.y,
                                     heading_msg->quaternion.z, heading_msg->quaternion.w);
            tf2::Vector3 base_to_gnss_offset(-gnss_to_base_transform.transform.translation.x,
                                            -gnss_to_base_transform.transform.translation.y,
                                            -gnss_to_base_transform.transform.translation.z);
            tf2::Vector3 rotated_offset = tf2::quatRotate(rotation, base_to_gnss_offset);
            
            // zx120からの相対位置にzx120のGNSSオフセット補正を適用（逆方向）
            transform.transform.translation.x = (robot_x - zx120_x) + rotated_offset.x() + zx120_gnss_offset.x();
            transform.transform.translation.y = (robot_y - zx120_y) + rotated_offset.y() + zx120_gnss_offset.y();
            transform.transform.translation.z = (robot_z - zx120_z) + rotated_offset.z() + zx120_gnss_offset.z();

            
            // zx120のヘディングを基準とした相対回転
            if (backhoe_heading_data_) {
                // zx120の回転の逆変換を適用してから、four_wheel_robotの回転を適用
                tf2::Quaternion zx120_rotation(
                    backhoe_heading_data_->quaternion.x,
                    backhoe_heading_data_->quaternion.y,
                    backhoe_heading_data_->quaternion.z,
                    backhoe_heading_data_->quaternion.w);
                tf2::Quaternion relative_rotation = zx120_rotation.inverse() * rotation;
                
                transform.transform.rotation.x = relative_rotation.x();
                transform.transform.rotation.y = relative_rotation.y();
                transform.transform.rotation.z = relative_rotation.z();
                transform.transform.rotation.w = relative_rotation.w();
            } else {
                transform.transform.rotation = heading_msg->quaternion;
            }
            
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "four_wheel_robot relative to zx120 (with ZX120 GNSS offset compensation) - Pos: [%.2f, %.2f, %.2f]",
                transform.transform.translation.x, 
                transform.transform.translation.y, 
                transform.transform.translation.z);
        }
        
        tf_broadcaster_->sendTransform(transform);
    }
    
    // Process and merge point clouds
    void processPointClouds() {
        if (!origin_set_) return;
        
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr merged_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr robot_colored(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr backhoe_colored(new pcl::PointCloud<pcl::PointXYZRGB>);
        
        // Process robot point cloud
        if (robot_cloud_data_) {
            processRobotCloud(robot_cloud_data_, "four_wheel_robot", 
                             merged_cloud, robot_colored, 255, 0, 0); // Red color
        }
        
        // Process backhoe point cloud
        if (backhoe_cloud_data_) {
            processRobotCloud(backhoe_cloud_data_, "zx120",
                             merged_cloud, backhoe_colored, 0, 0, 255); // Blue color
        }
        
        // Publish merged cloud
        if (!merged_cloud->empty()) {
            sensor_msgs::msg::PointCloud2 output_msg;
            pcl::toROSMsg(*merged_cloud, output_msg);
            output_msg.header.frame_id = "map";
            output_msg.header.stamp = this->now();
            matched_cloud_pub_->publish(output_msg);
        }
        
        // Publish individual colored clouds
        if (!robot_colored->empty()) {
            sensor_msgs::msg::PointCloud2 robot_msg;
            pcl::toROSMsg(*robot_colored, robot_msg);
            robot_msg.header.frame_id = "map";
            robot_msg.header.stamp = this->now();
            robot_colored_cloud_pub_->publish(robot_msg);
        }
        
        if (!backhoe_colored->empty()) {
            sensor_msgs::msg::PointCloud2 backhoe_msg;
            pcl::toROSMsg(*backhoe_colored, backhoe_msg);
            backhoe_msg.header.frame_id = "map";
            backhoe_msg.header.stamp = this->now();
            backhoe_colored_cloud_pub_->publish(backhoe_msg);
        }
    }
    
    void processRobotCloud(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg,
                          const std::string& robot_name,
                          pcl::PointCloud<pcl::PointXYZRGB>::Ptr merged_cloud,
                          pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud,
                          uint8_t r, uint8_t g, uint8_t b) {
        try {
            // Transform point cloud from velodyne_link to map frame
            geometry_msgs::msg::TransformStamped transform_stamped;
            transform_stamped = tf_buffer_->lookupTransform(
                "map",
                robot_name + "/velodyne_link",
                tf2::TimePointZero,
                tf2::durationFromSec(0.1));
            
            // Transform the point cloud
            sensor_msgs::msg::PointCloud2 transformed_cloud;
            tf2::doTransform(*cloud_msg, transformed_cloud, transform_stamped);
            
            // Convert to PCL and add color
            pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromROSMsg(transformed_cloud, *temp_cloud);
            
            for (const auto& point : temp_cloud->points) {
                pcl::PointXYZRGB colored_point;
                colored_point.x = point.x;
                colored_point.y = point.y;
                colored_point.z = point.z;
                colored_point.r = r;
                colored_point.g = g;
                colored_point.b = b;
                
                merged_cloud->push_back(colored_point);
                colored_cloud->push_back(colored_point);
            }
            
        } catch (tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "Could not transform point cloud for %s: %s", 
                robot_name.c_str(), ex.what());
        }
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GnssGicpMatcher>());
    rclcpp::shutdown();
    return 0;
}