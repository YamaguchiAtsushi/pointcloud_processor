#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <geometry_msgs/msg/quaternion_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <cmath>

class GnssCompassNode : public rclcpp::Node
{
public:
    GnssCompassNode() : Node("gnss_compass_node")
    {
        // Publishers
        heading_pub_fw_ = this->create_publisher<geometry_msgs::msg::QuaternionStamped>(
            "/four_wheel_robot/gnss_compass/calc_heading", 10);
        heading_pub_zx_ = this->create_publisher<geometry_msgs::msg::QuaternionStamped>(
            "/zx120/gnss_compass/calc_heading", 10);

        // Subscribers - four_wheel_robot
        gnss_fw_front_sub_ = this->create_subscription<sensor_msgs::msg::NavSatFix>(
            "/four_wheel_robot/gnss_compass_front/fix", 10,
            std::bind(&GnssCompassNode::gnss_fw_front_callback, this, std::placeholders::_1));
        gnss_fw_back_sub_ = this->create_subscription<sensor_msgs::msg::NavSatFix>(
            "/four_wheel_robot/gnss_compass_back/fix", 10,
            std::bind(&GnssCompassNode::gnss_fw_back_callback, this, std::placeholders::_1));

        // Subscribers - zx120
        gnss_zx_front_sub_ = this->create_subscription<sensor_msgs::msg::NavSatFix>(
            "/zx120/gnss_compass_front/fix", 10,
            std::bind(&GnssCompassNode::gnss_zx_front_callback, this, std::placeholders::_1));
        gnss_zx_back_sub_ = this->create_subscription<sensor_msgs::msg::NavSatFix>(
            "/zx120/gnss_compass_back/fix", 10,
            std::bind(&GnssCompassNode::gnss_zx_back_callback, this, std::placeholders::_1));

        // Initialize flags
        fw_front_received_ = fw_back_received_ = false;
        zx_front_received_ = zx_back_received_ = false;

        RCLCPP_INFO(this->get_logger(), "GNSS Compass Node started (four_wheel_robot & zx120)");
    }

private:
    // Publishers
    rclcpp::Publisher<geometry_msgs::msg::QuaternionStamped>::SharedPtr heading_pub_fw_;
    rclcpp::Publisher<geometry_msgs::msg::QuaternionStamped>::SharedPtr heading_pub_zx_;

    // Subscribers - four_wheel_robot
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gnss_fw_front_sub_;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gnss_fw_back_sub_;
    // Subscribers - zx120
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gnss_zx_front_sub_;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gnss_zx_back_sub_;

    // Data storage
    sensor_msgs::msg::NavSatFix fw_front_fix_, fw_back_fix_;
    sensor_msgs::msg::NavSatFix zx_front_fix_, zx_back_fix_;
    bool fw_front_received_, fw_back_received_;
    bool zx_front_received_, zx_back_received_;

    // four_wheel_robot callbacks
    void gnss_fw_front_callback(const sensor_msgs::msg::NavSatFix::SharedPtr msg)
    {
        fw_front_fix_ = *msg;
        fw_front_received_ = true;
        if (fw_front_received_ && fw_back_received_)
            calculate_and_publish_heading(fw_back_fix_, fw_front_fix_, heading_pub_fw_, "four_wheel_robot/gnss_link");
    }
    void gnss_fw_back_callback(const sensor_msgs::msg::NavSatFix::SharedPtr msg)
    {
        fw_back_fix_ = *msg;
        fw_back_received_ = true;
        if (fw_front_received_ && fw_back_received_)
            calculate_and_publish_heading(fw_back_fix_, fw_front_fix_, heading_pub_fw_, "four_wheel_robot/gnss_link");
    }

    // zx120 callbacks
    void gnss_zx_front_callback(const sensor_msgs::msg::NavSatFix::SharedPtr msg)
    {
        zx_front_fix_ = *msg;
        zx_front_received_ = true;
        if (zx_front_received_ && zx_back_received_)
            calculate_and_publish_heading(zx_back_fix_, zx_front_fix_, heading_pub_zx_, "zx120/gnss_link");
    }
    void gnss_zx_back_callback(const sensor_msgs::msg::NavSatFix::SharedPtr msg)
    {
        zx_back_fix_ = *msg;
        zx_back_received_ = true;
        if (zx_front_received_ && zx_back_received_)
            calculate_and_publish_heading(zx_back_fix_, zx_front_fix_, heading_pub_zx_, "zx120/gnss_link");
    }

    // 共通計算関数
    // 球面三角法の公式を使って「大円航法による方位角」を求めてい
    void calculate_and_publish_heading(
        const sensor_msgs::msg::NavSatFix &back_fix,
        const sensor_msgs::msg::NavSatFix &front_fix,
        rclcpp::Publisher<geometry_msgs::msg::QuaternionStamped>::SharedPtr pub,
        const std::string &frame_id)
    {
        if (back_fix.status.status < sensor_msgs::msg::NavSatStatus::STATUS_FIX ||
            front_fix.status.status < sensor_msgs::msg::NavSatStatus::STATUS_FIX) {
            RCLCPP_WARN(this->get_logger(), "[%s] Invalid GNSS fix status", frame_id.c_str());
            return;
        }

        double lat1 = deg_to_rad(back_fix.latitude);
        double lon1 = deg_to_rad(back_fix.longitude);
        double lat2 = deg_to_rad(front_fix.latitude);
        double lon2 = deg_to_rad(front_fix.longitude);

        double delta_lon = lon2 - lon1;
        double y = sin(delta_lon) * cos(lat2);
        double x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(delta_lon);
        double bearing = atan2(y, x);
        if (bearing < 0) bearing += 2 * M_PI;


        //bearingは北を0度とする時計回りの角度
        //yawは東を0度とする反時計回りの角度

        // 真北からのbearingをROSのyaw（東から反時計回り）に変換
        double yaw = M_PI/2 - bearing;  // 90度回転 + 符号反転
        if (yaw < 0) yaw += 2 * M_PI;

        tf2::Quaternion quat;
        quat.setRPY(0, 0, yaw);

        geometry_msgs::msg::QuaternionStamped heading_msg;
        heading_msg.header.stamp = this->get_clock()->now();
        heading_msg.header.frame_id = frame_id;
        heading_msg.quaternion = tf2::toMsg(quat);

        pub->publish(heading_msg);

        RCLCPP_INFO(this->get_logger(), "[%s] Heading: %.2f deg", frame_id.c_str(), rad_to_deg(bearing));
    }

    inline double deg_to_rad(double deg) const { return deg * M_PI / 180.0; }
    inline double rad_to_deg(double rad) const { return rad * 180.0 / M_PI; }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GnssCompassNode>());
    rclcpp::shutdown();
    return 0;
}
