#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

class SimplifiedScanMatcher : public rclcpp::Node
{
public:
    SimplifiedScanMatcher() : Node("simplified_scan_matcher")
    {
        // Subscribers
        robot_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/four_wheel_robot/velodyne_points", 1,
            std::bind(&SimplifiedScanMatcher::robotCloudCallback, this, std::placeholders::_1));
        
        backhoe_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/zx120/velodyne_points", 1,
            std::bind(&SimplifiedScanMatcher::backhoeCloudCallback, this, std::placeholders::_1));

        // Publishers
        robot_filtered_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/four_wheel_robot/filtered_points", 1);
        
        backhoe_filtered_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/zx120/filtered_points", 1);

        // パラメータ設定 - 前方範囲のみ
        this->declare_parameter("robot_front_range", 15.0);
        this->declare_parameter("robot_side_range", 10.0);
        this->declare_parameter("robot_height_range", 10.0);
        
        this->declare_parameter("backhoe_front_range", 15.0);
        this->declare_parameter("backhoe_side_range", 10.0);
        this->declare_parameter("backhoe_height_range", 10.0);
        
        // ダウンサンプリングパラメータ
        this->declare_parameter("voxel_leaf_size", 0.2);

        RCLCPP_INFO(this->get_logger(), "Simplified Scan Matcher initialized");
        RCLCPP_INFO(this->get_logger(), "Function: Front area cropping and downsampling only");
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr robot_cloud_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr backhoe_cloud_sub_;
    
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr robot_filtered_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr backhoe_filtered_pub_;

    void robotCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        auto filtered_cloud = processCloudSimple(msg, "robot");
        robot_filtered_pub_->publish(*filtered_cloud);
    }

    void backhoeCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        auto filtered_cloud = processCloudSimple(msg, "backhoe");
        backhoe_filtered_pub_->publish(*filtered_cloud);
    }

    sensor_msgs::msg::PointCloud2::SharedPtr processCloudSimple(
        const sensor_msgs::msg::PointCloud2::SharedPtr& input_msg,
        const std::string& vehicle_type)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*input_msg, *cloud);

        // 1. 前方範囲の切り出し
        auto cropped_cloud = cropFrontArea(cloud, vehicle_type);
        
        // 2. ダウンサンプリング
        auto downsampled_cloud = downsampleCloud(cropped_cloud);

        auto output_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
        pcl::toROSMsg(*downsampled_cloud, *output_msg);
        output_msg->header = input_msg->header;

        RCLCPP_DEBUG(this->get_logger(), "%s cloud: %zu -> %zu -> %zu points", 
                    vehicle_type.c_str(), cloud->size(), cropped_cloud->size(), downsampled_cloud->size());

        return output_msg;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cropFrontArea(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud, 
        const std::string& vehicle_type)
    {
        double front_range, side_range, height_range;
        
        if (vehicle_type == "robot") {
            front_range = this->get_parameter("robot_front_range").as_double();
            side_range = this->get_parameter("robot_side_range").as_double();
            height_range = this->get_parameter("robot_height_range").as_double();
        } else {
            front_range = this->get_parameter("backhoe_front_range").as_double();
            side_range = this->get_parameter("backhoe_side_range").as_double();
            height_range = this->get_parameter("backhoe_height_range").as_double();
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr cropped_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        cropped_cloud->reserve(input_cloud->size() / 4); // メモリ事前確保

        for (const auto& point : *input_cloud) {
            // 前方範囲のフィルタリング
            // X軸: 0 < x < front_range (前方)
            // Y軸: -side_range < y < side_range (左右)
            // Z軸: -1.5 < z < height_range (上下)
            if (point.x > 0.0 && point.x < front_range &&
                point.y > -side_range && point.y < side_range &&
                point.z > -1.5 && point.z < height_range) {
                cropped_cloud->push_back(point);
            }
        }

        cropped_cloud->header = input_cloud->header;
        return cropped_cloud;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampleCloud(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud)
    {
        if (input_cloud->empty()) {
            return input_cloud;
        }

        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setInputCloud(input_cloud);
        
        double leaf_size = this->get_parameter("voxel_leaf_size").as_double();
        voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);

        pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        voxel_filter.filter(*downsampled_cloud);

        return downsampled_cloud;
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SimplifiedScanMatcher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}