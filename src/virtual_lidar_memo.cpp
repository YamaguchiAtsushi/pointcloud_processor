#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>
#include <random>

struct GridCell {
    double x, y, z;
    bool is_valid;
    double score_zx120;
    double score_mobile;
    double combined_score;
    bool visible_from_zx120;
    bool visible_from_mobile;
    bool in_fov_zx120;
    bool in_fov_mobile;
    bool in_range_zx120;
    bool in_range_mobile;
    pcl::Normal surface_normal;
    
    GridCell(double x=0, double y=0, double z=0) 
        : x(x), y(y), z(z), is_valid(true), 
          score_zx120(0.0), score_mobile(0.0), combined_score(0.0),
          visible_from_zx120(false), visible_from_mobile(false),
          in_fov_zx120(false), in_fov_mobile(false),
          in_range_zx120(false), in_range_mobile(false) {
        surface_normal.normal_x = 0.0;
        surface_normal.normal_y = 0.0;
        surface_normal.normal_z = 1.0;
    }
};

struct LidarPosition {
    double x, y, z, pitch, yaw, total_score;
    
    LidarPosition(double x=0, double y=0, double z=10, double pitch=-M_PI/2, double yaw=0) 
        : x(x), y(y), z(z), pitch(pitch), yaw(yaw), total_score(0.0) {}
};

struct SimplifiedEvaluation {
    double total_score;
    double coverage_ratio;
    int covered_cells;
    int total_cells;
};

class SimplifiedDualLidarOptimizer : public rclcpp::Node {
public:
    SimplifiedDualLidarOptimizer() : Node("simplified_dual_lidar_optimizer") {
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        
        this->declare_parameter("grid_resolution", 0.1);
        this->declare_parameter("sensor_height", 1.1);
        this->declare_parameter("search_radius", 3.0);
        this->declare_parameter("max_distance", 15.0);
        this->declare_parameter("num_candidates", 100);
        this->declare_parameter("vertical_layers", 10);
        
        updateParameters();
        
        excavation_area_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/excavation_area", 10,
            std::bind(&SimplifiedDualLidarOptimizer::excavationAreaCallback, this, std::placeholders::_1));
            
        terrain_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/excavated_terrain", 10,
            std::bind(&SimplifiedDualLidarOptimizer::terrainCallback, this, std::placeholders::_1));
        
        zx120_points_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/zx120/filtered_points", 10,
            std::bind(&SimplifiedDualLidarOptimizer::zx120PointsCallback, this, std::placeholders::_1));
        
        optimal_position_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>(
            "/optimal_mobile_lidar_position", 10);
        candidate_positions_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/mobile_lidar_candidate_positions", 10);
        grid_visualization_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/excavation_grid_visualization", 10);
        
        optimization_timer_ = this->create_wall_timer(
            std::chrono::seconds(3),
            std::bind(&SimplifiedDualLidarOptimizer::runOptimization, this));
    }
    
private:
    static constexpr double ALPHA = 1.0;
    static constexpr double BETA = 1.0;
    static constexpr double MIN_DISTANCE = 1.0;
    static constexpr double ZX120_OFFSET_X = 0.4;
    static constexpr double ZX120_OFFSET_Y = 0.5;
    static constexpr double ZX120_OFFSET_Z = 3.5;
    static constexpr double ZX120_PITCH = -M_PI/6;
    static constexpr double ZX120_YAW = 0.0;
    static constexpr double FOV_HORIZONTAL = 360.0 * M_PI / 180.0;
    static constexpr double FOV_VERTICAL = 180.0 * M_PI / 180.0;
    static constexpr double NORMAL_SEARCH_RADIUS = 1.5;
    static constexpr double RAY_STEP_SIZE = 0.2;
    static constexpr double VISIBILITY_RADIUS = 0.05;
    static constexpr double MIN_ELEVATION = -80.0 * M_PI / 180.0;
    static constexpr double MAX_ELEVATION = 85.0 * M_PI / 180.0;
    
    double grid_resolution_;
    double sensor_height_;
    double search_radius_;
    double max_distance_;
    int num_candidates_;
    int vertical_layers_;
    
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr excavation_area_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr terrain_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr zx120_points_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr optimal_position_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr candidate_positions_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr grid_visualization_pub_;
    
    rclcpp::TimerBase::SharedPtr optimization_timer_;
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr excavation_area_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr terrain_cloud_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr zx120_cloud_;
    pcl::KdTreeFLANN<pcl::PointXYZRGB>::Ptr terrain_kdtree_;
    pcl::KdTreeFLANN<pcl::PointXYZRGB>::Ptr zx120_kdtree_;
    pcl::KdTreeFLANN<pcl::PointXYZRGB>::Ptr excavation_kdtree_;
    pcl::PointCloud<pcl::Normal>::Ptr terrain_normals_;
    
    std::vector<GridCell> excavation_grid_3d_;
    double grid_min_x_, grid_max_x_, grid_min_y_, grid_max_y_;
    double excavation_min_z_, excavation_max_z_;
    int grid_width_, grid_height_;
    
    std::mt19937 rng_;
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> normal_estimator_;
    
    LidarPosition best_mobile_position_;
    LidarPosition zx120_lidar_position_;
    std::vector<LidarPosition> candidate_positions_;
    
    void updateParameters() {
        grid_resolution_ = this->get_parameter("grid_resolution").as_double();
        sensor_height_ = this->get_parameter("sensor_height").as_double();
        search_radius_ = this->get_parameter("search_radius").as_double();
        max_distance_ = this->get_parameter("max_distance").as_double();
        num_candidates_ = this->get_parameter("num_candidates").as_int();
        vertical_layers_ = this->get_parameter("vertical_layers").as_int();
    }
    
    void excavationAreaCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        excavation_area_.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::fromROSMsg(*msg, *excavation_area_);
        
        if (excavation_area_->empty()) return;
        
        excavation_kdtree_.reset(new pcl::KdTreeFLANN<pcl::PointXYZRGB>);
        try {
            excavation_kdtree_->setInputCloud(excavation_area_);
            computeTerrainNormals();
            generateExcavationGrid3D();
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to process excavation area: %s", e.what());
        }
    }
    
    void terrainCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        terrain_cloud_.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::fromROSMsg(*msg, *terrain_cloud_);
        
        if (!terrain_cloud_->empty()) {
            terrain_kdtree_.reset(new pcl::KdTreeFLANN<pcl::PointXYZRGB>);
            try {
                terrain_kdtree_->setInputCloud(terrain_cloud_);
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "Failed to build terrain KD-tree: %s", e.what());
            }
        }
    }
    
    void zx120PointsCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        zx120_cloud_.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::fromROSMsg(*msg, *zx120_cloud_);
        
        if (!zx120_cloud_->empty()) {
            zx120_kdtree_.reset(new pcl::KdTreeFLANN<pcl::PointXYZRGB>);
            try {
                zx120_kdtree_->setInputCloud(zx120_cloud_);
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "Failed to build ZX120 KD-tree: %s", e.what());
            }
        }
    }
    
    void computeTerrainNormals() {
        if (!excavation_area_ || excavation_area_->empty()) return;
        
        try {
            pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGB>());
            kdtree->setInputCloud(excavation_area_);
            
            normal_estimator_.setInputCloud(excavation_area_);
            normal_estimator_.setSearchMethod(kdtree);
            normal_estimator_.setRadiusSearch(NORMAL_SEARCH_RADIUS);
            
            terrain_normals_.reset(new pcl::PointCloud<pcl::Normal>);
            normal_estimator_.compute(*terrain_normals_);
            
            for (auto& normal : terrain_normals_->points) {
                if (normal.normal_z < 0) {
                    normal.normal_x = -normal.normal_x;
                    normal.normal_y = -normal.normal_y;
                    normal.normal_z = -normal.normal_z;
                }
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to compute terrain normals: %s", e.what());
            terrain_normals_.reset(new pcl::PointCloud<pcl::Normal>);
        }
    }
    
    void generateExcavationGrid3D() {
        if (!excavation_area_ || excavation_area_->empty()) return;
        
        grid_min_x_ = grid_min_y_ = excavation_min_z_ = std::numeric_limits<double>::max();
        grid_max_x_ = grid_max_y_ = excavation_max_z_ = std::numeric_limits<double>::lowest();
        
        for (const auto& point : excavation_area_->points) {
            grid_min_x_ = std::min(grid_min_x_, static_cast<double>(point.x));
            grid_max_x_ = std::max(grid_max_x_, static_cast<double>(point.x));
            grid_min_y_ = std::min(grid_min_y_, static_cast<double>(point.y));
            grid_max_y_ = std::max(grid_max_y_, static_cast<double>(point.y));
            excavation_min_z_ = std::min(excavation_min_z_, static_cast<double>(point.z));
            excavation_max_z_ = std::max(excavation_max_z_, static_cast<double>(point.z));
        }
        
        double margin = grid_resolution_;
        grid_min_x_ -= margin; grid_max_x_ += margin;
        grid_min_y_ -= margin; grid_max_y_ += margin;
        excavation_min_z_ -= margin; excavation_max_z_ += margin;
        
        grid_width_ = static_cast<int>(std::ceil((grid_max_x_ - grid_min_x_) / grid_resolution_)) + 1;
        grid_height_ = static_cast<int>(std::ceil((grid_max_y_ - grid_min_y_) / grid_resolution_)) + 1;
        
        excavation_grid_3d_.clear();
        
        double z_range = excavation_max_z_ - excavation_min_z_;
        double z_step = z_range / std::max(1, vertical_layers_);
        
        int valid_cells = 0;
        
        for (int i = 0; i < grid_height_; ++i) {
            for (int j = 0; j < grid_width_; ++j) {
                double x = grid_min_x_ + j * grid_resolution_;
                double y = grid_min_y_ + i * grid_resolution_;
                
                for (int k = 0; k < vertical_layers_; ++k) {
                    double z = excavation_min_z_ + k * z_step + z_step / 2.0;
                    
                    if (isPointNearExcavation(x, y, z, grid_resolution_ * 1.5)) {
                        GridCell cell(x, y, z);
                        computeCellSurfaceNormal(cell);
                        excavation_grid_3d_.push_back(cell);
                        valid_cells++;
                    }
                }
            }
        }
        
        RCLCPP_INFO(this->get_logger(), "Generated 3D grid: %d valid cells across %d vertical layers", 
                   valid_cells, vertical_layers_);
        publishGridVisualization();
    }
    
    bool isPointNearExcavation(double x, double y, double z, double radius) {
        pcl::PointXYZRGB search_point;
        search_point.x = x;
        search_point.y = y;
        search_point.z = z;
        
        std::vector<int> point_indices;
        std::vector<float> point_distances;
        
        return excavation_kdtree_->radiusSearch(search_point, radius, point_indices, point_distances) > 0;
    }
    
    void computeCellSurfaceNormal(GridCell& cell) {
        if (!terrain_normals_ || terrain_normals_->empty()) return;
        
        pcl::PointXYZRGB search_point;
        search_point.x = cell.x;
        search_point.y = cell.y;
        search_point.z = cell.z;
        
        std::vector<int> point_indices;
        std::vector<float> point_distances;
        
        if (excavation_kdtree_->radiusSearch(search_point, NORMAL_SEARCH_RADIUS, 
                                           point_indices, point_distances) > 0) {
            
            double sum_nx = 0.0, sum_ny = 0.0, sum_nz = 0.0;
            int valid_count = 0;
            
            for (int idx : point_indices) {
                if (idx < static_cast<int>(terrain_normals_->size())) {
                    const auto& normal = terrain_normals_->points[idx];
                    if (std::isfinite(normal.normal_x) && std::isfinite(normal.normal_y) && 
                        std::isfinite(normal.normal_z)) {
                        sum_nx += normal.normal_x;
                        sum_ny += normal.normal_y;
                        sum_nz += normal.normal_z;
                        valid_count++;
                    }
                }
            }
            
            if (valid_count > 0) {
                double norm = sqrt(sum_nx*sum_nx + sum_ny*sum_ny + sum_nz*sum_nz);
                if (norm > 1e-6) {
                    cell.surface_normal.normal_x = sum_nx / norm;
                    cell.surface_normal.normal_y = sum_ny / norm;
                    cell.surface_normal.normal_z = sum_nz / norm;
                }
            }
        }
    }
    
    bool getZX120Position() {
        try {
            geometry_msgs::msg::TransformStamped transform = 
                tf_buffer_->lookupTransform("map", "zx120/base_link", tf2::TimePointZero, 
                                          tf2::durationFromSec(0.1));
            
            zx120_lidar_position_.x = transform.transform.translation.x + ZX120_OFFSET_X;
            zx120_lidar_position_.y = transform.transform.translation.y + ZX120_OFFSET_Y;
            zx120_lidar_position_.z = transform.transform.translation.z + ZX120_OFFSET_Z;
            zx120_lidar_position_.pitch = ZX120_PITCH;
            zx120_lidar_position_.yaw = ZX120_YAW;
            
            return true;
        } catch (tf2::TransformException& ex) {
            return false;
        }
    }
    
    void evaluateZX120Only() {
        double total_score_zx120 = 0.0;
        int total_cells = 0;
        int green_cells_zx120 = 0;
        int red_cells_zx120 = 0;
        int blue_cells_zx120 = 0;
        int yellow_cells_zx120 = 0;
        
        for (auto& cell : excavation_grid_3d_) {
            if (!cell.is_valid) continue;
            
            total_cells++;
            
            // ZX120からのスコア評価
            cell.score_zx120 = evaluateCellScore(zx120_lidar_position_, cell, true);
            
            if (cell.score_zx120 > 0) {
                total_score_zx120 += cell.score_zx120;
            }
            
            // セルの分類（ZX120のみ）
            if (!cell.in_range_zx120) {
                blue_cells_zx120++;
            } else if (!cell.in_fov_zx120) {
                yellow_cells_zx120++;
            } else if (!cell.visible_from_zx120) {
                red_cells_zx120++;
            } else {
                green_cells_zx120++;
            }
        }
        
        // 比率計算
        double green_ratio_zx120 = total_cells > 0 ? 
            static_cast<double>(green_cells_zx120) / total_cells * 100.0 : 0.0;
        double red_ratio_zx120 = total_cells > 0 ? 
            static_cast<double>(red_cells_zx120) / total_cells * 100.0 : 0.0;
        double blue_ratio_zx120 = total_cells > 0 ? 
            static_cast<double>(blue_cells_zx120) / total_cells * 100.0 : 0.0;
        double yellow_ratio_zx120 = total_cells > 0 ? 
            static_cast<double>(yellow_cells_zx120) / total_cells * 100.0 : 0.0;
        
        double red_to_green_ratio_zx120 = green_cells_zx120 > 0 ? 
            static_cast<double>(red_cells_zx120) / green_cells_zx120 : 
            (red_cells_zx120 > 0 ? std::numeric_limits<double>::infinity() : 0.0);
        
        int unobservable_zx120 = red_cells_zx120 + blue_cells_zx120 + yellow_cells_zx120;
        double unobservable_ratio_zx120 = total_cells > 0 ? 
            static_cast<double>(unobservable_zx120) / total_cells * 100.0 : 0.0;
        
        // ZX120単独の結果を出力
        RCLCPP_INFO(this->get_logger(), "========================================");
        RCLCPP_INFO(this->get_logger(), "ZX120 LiDAR Only Evaluation");
        RCLCPP_INFO(this->get_logger(), "========================================");
        RCLCPP_INFO(this->get_logger(), "ZX120 Position: (%.2f, %.2f, %.2f)", 
                   zx120_lidar_position_.x, zx120_lidar_position_.y, zx120_lidar_position_.z);
        RCLCPP_INFO(this->get_logger(), "Total Score (ZX120 only): %.2f", total_score_zx120);
        RCLCPP_INFO(this->get_logger(), "----------------------------------------");
        RCLCPP_INFO(this->get_logger(), "Color-based Area Analysis (ZX120 only):");
        RCLCPP_INFO(this->get_logger(), "  Total cells: %d", total_cells);
        RCLCPP_INFO(this->get_logger(), "  Green (Observable): %d cells (%.1f%%)", 
                   green_cells_zx120, green_ratio_zx120);
        RCLCPP_INFO(this->get_logger(), "  Red (Occluded): %d cells (%.1f%%)", 
                   red_cells_zx120, red_ratio_zx120);
        RCLCPP_INFO(this->get_logger(), "  Blue (Out of range): %d cells (%.1f%%)", 
                   blue_cells_zx120, blue_ratio_zx120);
        RCLCPP_INFO(this->get_logger(), "  Yellow (Out of FOV): %d cells (%.1f%%)", 
                   yellow_cells_zx120, yellow_ratio_zx120);
        RCLCPP_INFO(this->get_logger(), "  ---");
        RCLCPP_INFO(this->get_logger(), "  Red/Green Ratio: %.3f", red_to_green_ratio_zx120);
        RCLCPP_INFO(this->get_logger(), "  Total Unobservable: %d cells (%.1f%%)", 
                   unobservable_zx120, unobservable_ratio_zx120);
        RCLCPP_INFO(this->get_logger(), "========================================");
        RCLCPP_INFO(this->get_logger(), "");
    }
    
    void runOptimization() {
        if (excavation_grid_3d_.empty() || !terrain_cloud_ || !getZX120Position()) return;
        
        updateParameters();
        
        // ========== ZX120単独での評価 ==========
        evaluateZX120Only();
        
        auto candidates = generateCandidatePositions();
        
        double best_score = -std::numeric_limits<double>::infinity();
        LidarPosition best_candidate;
        
        for (auto& candidate : candidates) {
            auto evaluation = evaluatePosition(candidate);
            candidate.total_score = evaluation.total_score;
            
            if (candidate.total_score > best_score) {
                best_score = candidate.total_score;
                best_candidate = candidate;
            }
        }
        
        best_mobile_position_ = best_candidate;
        candidate_positions_ = candidates;
        
        // ========== 色別エリア統計（デュアル構成） ==========
        int total_cells = 0;
        int green_cells = 0;   // 観測可能（緑）
        int red_cells = 0;     // オクルージョン（赤）
        int blue_cells = 0;    // 距離範囲外（青）
        int yellow_cells = 0;  // FOV外（黄）
        
        for (const auto& cell : excavation_grid_3d_) {
            if (!cell.is_valid) continue;
            
            total_cells++;
            
            if (!cell.in_range_zx120 && !cell.in_range_mobile) {
                blue_cells++;
            } else if (!cell.in_fov_zx120 && !cell.in_fov_mobile) {
                yellow_cells++;
            } else if (!cell.visible_from_zx120 && !cell.visible_from_mobile) {
                red_cells++;
            } else {
                green_cells++;
            }
        }
        
        // 比率計算
        double green_ratio = total_cells > 0 ? 
            static_cast<double>(green_cells) / total_cells * 100.0 : 0.0;
        double red_ratio = total_cells > 0 ? 
            static_cast<double>(red_cells) / total_cells * 100.0 : 0.0;
        double blue_ratio = total_cells > 0 ? 
            static_cast<double>(blue_cells) / total_cells * 100.0 : 0.0;
        double yellow_ratio = total_cells > 0 ? 
            static_cast<double>(yellow_cells) / total_cells * 100.0 : 0.0;
        
        double red_to_green_ratio = green_cells > 0 ? 
            static_cast<double>(red_cells) / green_cells : 
            (red_cells > 0 ? std::numeric_limits<double>::infinity() : 0.0);
        
        int unobservable = red_cells + blue_cells + yellow_cells;
        double unobservable_ratio = total_cells > 0 ? 
            static_cast<double>(unobservable) / total_cells * 100.0 : 0.0;
        
        // 結果を出力
        RCLCPP_INFO(this->get_logger(), "========================================");
        RCLCPP_INFO(this->get_logger(), "Dual LiDAR Configuration (ZX120 + Mobile)");
        RCLCPP_INFO(this->get_logger(), "========================================");
        RCLCPP_INFO(this->get_logger(), "Best Mobile LiDAR Position: (%.2f, %.2f, %.2f)", 
                   best_mobile_position_.x, best_mobile_position_.y, best_mobile_position_.z);
        RCLCPP_INFO(this->get_logger(), "Total Score: %.2f", best_score);
        RCLCPP_INFO(this->get_logger(), "========================================");
        RCLCPP_INFO(this->get_logger(), "Color-based Area Analysis:");
        RCLCPP_INFO(this->get_logger(), "  Total cells: %d", total_cells);
        RCLCPP_INFO(this->get_logger(), "  Green (Observable): %d cells (%.1f%%)", 
                   green_cells, green_ratio);
        RCLCPP_INFO(this->get_logger(), "  Red (Occluded): %d cells (%.1f%%)", 
                   red_cells, red_ratio);
        RCLCPP_INFO(this->get_logger(), "  Blue (Out of range): %d cells (%.1f%%)", 
                   blue_cells, blue_ratio);
        RCLCPP_INFO(this->get_logger(), "  Yellow (Out of FOV): %d cells (%.1f%%)", 
                   yellow_cells, yellow_ratio);
        RCLCPP_INFO(this->get_logger(), "  ---");
        RCLCPP_INFO(this->get_logger(), "  Red/Green Ratio: %.3f", red_to_green_ratio);
        RCLCPP_INFO(this->get_logger(), "  Total Unobservable: %d cells (%.1f%%)", 
                   unobservable, unobservable_ratio);
        RCLCPP_INFO(this->get_logger(), "========================================");
        
        publishOptimalPosition();
        publishCandidatePositions();
        publishGridVisualization();
    }
    
    std::vector<LidarPosition> generateCandidatePositions() {
        std::vector<LidarPosition> candidates;
        
        double expanded_min_x = grid_min_x_ - search_radius_;
        double expanded_max_x = grid_max_x_ + search_radius_;
        double expanded_min_y = grid_min_y_ - search_radius_;
        double expanded_max_y = grid_max_y_ + search_radius_;
        
        double center_x = (grid_min_x_ + grid_max_x_) / 2.0;
        double center_y = (grid_min_y_ + grid_max_y_) / 2.0;
        double center_z = (excavation_min_z_ + excavation_max_z_) / 2.0;
        
        int grid_size = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(num_candidates_))));
        double x_step = (expanded_max_x - expanded_min_x) / (grid_size - 1);
        double y_step = (expanded_max_y - expanded_min_y) / (grid_size - 1);
        
        for (int i = 0; i < grid_size; ++i) {
            for (int j = 0; j < grid_size; ++j) {
                double x = expanded_min_x + i * x_step;
                double y = expanded_min_y + j * y_step;
                
                double dist_to_zx120 = sqrt(pow(x - zx120_lidar_position_.x, 2) + 
                                          pow(y - zx120_lidar_position_.y, 2));
                if (dist_to_zx120 < 0.5) continue;
                
                if (x >= grid_min_x_ && x <= grid_max_x_ && y >= grid_min_y_ && y <= grid_max_y_) continue;
                
                double ground_z = getGroundHeight(x, y);
                double z = ground_z + sensor_height_;
                
                double dx = center_x - x;
                double dy = center_y - y;
                double dz = center_z - z;
                double horizontal_distance = sqrt(dx*dx + dy*dy);
                
                if (horizontal_distance < 0.1) continue;
                
                double elevation_angle = atan2(-dz, horizontal_distance);
                
                if (elevation_angle >= MIN_ELEVATION && elevation_angle <= MAX_ELEVATION) {
                    double pitch = -M_PI/2 + elevation_angle;
                    double yaw = atan2(dy, dx);
                    candidates.emplace_back(x, y, z, pitch, yaw);
                }
            }
        }
        
        return candidates;
    }
    
    double getGroundHeight(double x, double y) {
        if (!terrain_cloud_ || terrain_cloud_->empty()) return 0.0;
        
        pcl::PointXYZRGB search_point;
        search_point.x = x;
        search_point.y = y;
        search_point.z = 0;
        
        std::vector<int> point_indices;
        std::vector<float> point_distances;
        
        if (terrain_kdtree_->radiusSearch(search_point, 2.0, point_indices, point_distances) > 0) {
            double max_z = std::numeric_limits<double>::lowest();
            for (int idx : point_indices) {
                const auto& point = terrain_cloud_->points[idx];
                double dx = point.x - x;
                double dy = point.y - y;
                if (sqrt(dx*dx + dy*dy) < 1.0) {
                    max_z = std::max(max_z, static_cast<double>(point.z));
                }
            }
            if (max_z != std::numeric_limits<double>::lowest()) return max_z;
        }
        
        return 0.0;
    }
    
    SimplifiedEvaluation evaluatePosition(const LidarPosition& mobile_pos) {
        SimplifiedEvaluation evaluation = {};
        
        double total_score = 0.0;
        int covered_cells = 0;
        int total_valid_cells = excavation_grid_3d_.size();
        
        for (auto& cell : excavation_grid_3d_) {
            if (!cell.is_valid) continue;
            
            cell.score_zx120 = evaluateCellScore(zx120_lidar_position_, cell, true);
            cell.score_mobile = evaluateCellScore(mobile_pos, cell, false);
            cell.combined_score = std::max(cell.score_zx120, cell.score_mobile);
            
            if (cell.combined_score > 0) {
                covered_cells++;
                total_score += cell.combined_score;
            }
        }
        
        evaluation.total_score = total_score;
        evaluation.covered_cells = covered_cells;
        evaluation.total_cells = total_valid_cells;
        evaluation.coverage_ratio = total_valid_cells > 0 ? 
            static_cast<double>(covered_cells) / total_valid_cells : 0.0;
        
        return evaluation;
    }
    
    double evaluateCellScore(const LidarPosition& lidar_pos, GridCell& cell, bool is_zx120) {
        double dx = cell.x - lidar_pos.x;
        double dy = cell.y - lidar_pos.y;
        double dz = cell.z - lidar_pos.z;
        double L = sqrt(dx*dx + dy*dy + dz*dz);
        
        bool in_range = (L >= MIN_DISTANCE && L <= max_distance_);
        if (is_zx120) {
            cell.in_range_zx120 = in_range;
        } else {
            cell.in_range_mobile = in_range;
        }
        
        if (!in_range) return 0.0;
        
        bool in_fov = isInFieldOfView(lidar_pos, cell, dx, dy, dz, L);
        if (is_zx120) {
            cell.in_fov_zx120 = in_fov;
        } else {
            cell.in_fov_mobile = in_fov;
        }
        
        if (!in_fov) return 0.0;
        
        bool visible = checkVisibility(lidar_pos, cell, is_zx120);
        if (is_zx120) {
            cell.visible_from_zx120 = visible;
        } else {
            cell.visible_from_mobile = visible;
        }
        
        if (!visible) return 0.0;
        
        double beam_x = dx / L;
        double beam_y = dy / L;
        double beam_z = dz / L;
        
        double dot_product = beam_x * cell.surface_normal.normal_x + 
                           beam_y * cell.surface_normal.normal_y + 
                           beam_z * cell.surface_normal.normal_z;
        
        double theta = acos(std::max(0.0, std::min(1.0, std::abs(dot_product))));
        double score = ALPHA * sin(M_PI/2 - theta) + BETA * (1.0 / L);
        
        return std::max(0.0, score);
    }
    
    bool isInFieldOfView(const LidarPosition& lidar_pos, const GridCell& cell, 
        double dx, double dy, double dz, double distance) {
        double azimuth = atan2(dy, dx);
        double elevation = atan2(dz, sqrt(dx*dx + dy*dy));

        double azimuth_diff = fmod(azimuth - lidar_pos.yaw + M_PI, 2*M_PI) - M_PI;
        double elevation_diff = elevation - lidar_pos.pitch;

        const double FOV_HORIZONTAL_LOCAL = 180.0 * M_PI / 180.0;
        const double FOV_VERTICAL_LOCAL = 90.0 * M_PI / 180.0;

        return (std::abs(azimuth_diff) <= FOV_HORIZONTAL_LOCAL / 2.0) &&
               (std::abs(elevation_diff) <= FOV_VERTICAL_LOCAL / 2.0);
    }
        
    bool checkVisibility(const LidarPosition& lidar_pos, const GridCell& cell, bool is_zx120) {
        if (is_zx120) {
            if (!zx120_kdtree_ || !zx120_cloud_ || zx120_cloud_->empty()) {
                return false;
            }
            return checkVisibilityWithPointCloud(lidar_pos, cell, zx120_kdtree_);
        } else {
            if (!terrain_kdtree_) return true;
            return checkVisibilityWithRaycasting(lidar_pos, cell, terrain_kdtree_);
        }
    }
    
    bool checkVisibilityWithPointCloud(const LidarPosition& lidar_pos, const GridCell& cell,
                                       pcl::KdTreeFLANN<pcl::PointXYZRGB>::Ptr kdtree) {
        pcl::PointXYZRGB search_point;
        search_point.x = cell.x;
        search_point.y = cell.y;
        search_point.z = cell.z;
        
        std::vector<int> point_indices;
        std::vector<float> point_distances;
        
        if (kdtree->radiusSearch(search_point, VISIBILITY_RADIUS, point_indices, point_distances) > 0) {
            return true;
        }
        
        return false;
    }
    
    bool checkVisibilityWithRaycasting(const LidarPosition& lidar_pos, const GridCell& cell,
                                       pcl::KdTreeFLANN<pcl::PointXYZRGB>::Ptr kdtree) {
        double dx = cell.x - lidar_pos.x;
        double dy = cell.y - lidar_pos.y;
        double dz = cell.z - lidar_pos.z;
        double distance = sqrt(dx*dx + dy*dy + dz*dz);
        
        double norm_dx = dx / distance;
        double norm_dy = dy / distance;
        double norm_dz = dz / distance;
        
        double start_offset = 0.5;
        double step_distance = start_offset;
        double end_distance = distance - VISIBILITY_RADIUS;
        
        while (step_distance < end_distance) {
            double check_x = lidar_pos.x + norm_dx * step_distance;
            double check_y = lidar_pos.y + norm_dy * step_distance;
            double check_z = lidar_pos.z + norm_dz * step_distance;
            
            pcl::PointXYZRGB search_point;
            search_point.x = check_x;
            search_point.y = check_y;
            search_point.z = check_z;
            
            std::vector<int> point_indices;
            std::vector<float> point_distances;
            
            if (kdtree->radiusSearch(search_point, VISIBILITY_RADIUS * 0.7, 
                                    point_indices, point_distances) > 0) {
                bool is_blocked = false;
                for (size_t i = 0; i < point_indices.size(); ++i) {
                    if (point_distances[i] < VISIBILITY_RADIUS * 0.5) {
                        is_blocked = true;
                        break;
                    }
                }
                if (is_blocked) {
                    return false;
                }
            }
            
            step_distance += RAY_STEP_SIZE;
        }
        
        return true;
    }
    
    void publishOptimalPosition() {
        geometry_msgs::msg::PointStamped position_msg;
        position_msg.header.stamp = this->now();
        position_msg.header.frame_id = "map";
        position_msg.point.x = best_mobile_position_.x;
        position_msg.point.y = best_mobile_position_.y;
        position_msg.point.z = best_mobile_position_.z;
        
        optimal_position_pub_->publish(position_msg);
    }
    
    void publishCandidatePositions() {
        visualization_msgs::msg::MarkerArray marker_array;
        
        visualization_msgs::msg::Marker clear_marker;
        clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        marker_array.markers.push_back(clear_marker);
        
        visualization_msgs::msg::Marker zx120_marker;
        zx120_marker.header.stamp = this->now();
        zx120_marker.header.frame_id = "map";
        zx120_marker.ns = "zx120_lidar";
        zx120_marker.id = 0;
        zx120_marker.type = visualization_msgs::msg::Marker::CUBE;
        zx120_marker.action = visualization_msgs::msg::Marker::ADD;
        
        zx120_marker.pose.position.x = zx120_lidar_position_.x;
        zx120_marker.pose.position.y = zx120_lidar_position_.y;
        zx120_marker.pose.position.z = zx120_lidar_position_.z;
        zx120_marker.pose.orientation.w = 1.0;
        
        zx120_marker.scale.x = 0.5;
        zx120_marker.scale.y = 0.5;
        zx120_marker.scale.z = 0.5;
        
        zx120_marker.color.r = 0.0;
        zx120_marker.color.g = 1.0;
        zx120_marker.color.b = 1.0;
        zx120_marker.color.a = 1.0;
        
        zx120_marker.lifetime = rclcpp::Duration::from_seconds(10.0);
        marker_array.markers.push_back(zx120_marker);
        
        double min_score = std::numeric_limits<double>::max();
        double max_score = std::numeric_limits<double>::lowest();
        
        for (const auto& candidate : candidate_positions_) {
            min_score = std::min(min_score, candidate.total_score);
            max_score = std::max(max_score, candidate.total_score);
        }
        
        for (size_t i = 0; i < candidate_positions_.size(); ++i) {
            visualization_msgs::msg::Marker marker;
            marker.header.stamp = this->now();
            marker.header.frame_id = "map";
            marker.ns = "mobile_lidar_candidates";
            marker.id = static_cast<int>(i);
            marker.type = visualization_msgs::msg::Marker::SPHERE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            
            marker.pose.position.x = candidate_positions_[i].x;
            marker.pose.position.y = candidate_positions_[i].y;
            marker.pose.position.z = candidate_positions_[i].z;
            marker.pose.orientation.w = 1.0;
            
            marker.scale.x = 0.3;
            marker.scale.y = 0.3;
            marker.scale.z = 0.3;
            
            marker.color.r = 1.0;
            marker.color.g = 1.0;
            marker.color.b = 0.0;
            marker.color.a = 0.7;
            
            marker.lifetime = rclcpp::Duration::from_seconds(8.0);
            marker_array.markers.push_back(marker);
        }
        
        visualization_msgs::msg::Marker best_marker;
        best_marker.header.stamp = this->now();
        best_marker.header.frame_id = "map";
        best_marker.ns = "optimal_mobile_lidar";
        best_marker.id = 0;
        best_marker.type = visualization_msgs::msg::Marker::CYLINDER;
        best_marker.action = visualization_msgs::msg::Marker::ADD;
        
        best_marker.pose.position.x = best_mobile_position_.x;
        best_marker.pose.position.y = best_mobile_position_.y;
        best_marker.pose.position.z = best_mobile_position_.z;
        best_marker.pose.orientation.w = 1.0;
        
        best_marker.scale.x = 1.0;
        best_marker.scale.y = 1.0;
        best_marker.scale.z = 2.0;
        
        best_marker.color.r = 0.0;
        best_marker.color.g = 0.0;
        best_marker.color.b = 1.0;
        best_marker.color.a = 0.9;
        
        best_marker.lifetime = rclcpp::Duration::from_seconds(10.0);
        marker_array.markers.push_back(best_marker);
        
        candidate_positions_pub_->publish(marker_array);
    }
    
    void publishGridVisualization() {
        visualization_msgs::msg::MarkerArray marker_array;
        
        visualization_msgs::msg::Marker clear_marker;
        clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        marker_array.markers.push_back(clear_marker);
        
        int marker_id = 0;
        for (const auto& cell : excavation_grid_3d_) {
            if (!cell.is_valid) continue;
            
            visualization_msgs::msg::Marker marker;
            marker.header.stamp = this->now();
            marker.header.frame_id = "map";
            marker.ns = "excavation_grid_3d";
            marker.id = marker_id++;
            marker.type = visualization_msgs::msg::Marker::CUBE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            
            marker.pose.position.x = cell.x;
            marker.pose.position.y = cell.y;
            marker.pose.position.z = cell.z;
            marker.pose.orientation.w = 1.0;
            
            marker.scale.x = grid_resolution_ * 0.6;
            marker.scale.y = grid_resolution_ * 0.6;
            marker.scale.z = grid_resolution_ * 0.6;
            
            if (!cell.in_range_zx120 && !cell.in_range_mobile) {
                marker.color.r = 0.0;
                marker.color.g = 0.0;
                marker.color.b = 1.0;
                marker.color.a = 0.5;
            } else if (!cell.in_fov_zx120 && !cell.in_fov_mobile) {
                marker.color.r = 1.0;
                marker.color.g = 1.0;
                marker.color.b = 0.0;
                marker.color.a = 0.5;
            } else if (!cell.visible_from_zx120 && !cell.visible_from_mobile) {
                marker.color.r = 1.0;
                marker.color.g = 0.0;
                marker.color.b = 0.0;
                marker.color.a = 0.5;
            } else {
                marker.color.r = 0.0;
                marker.color.g = 1.0;
                marker.color.b = 0.0;
                marker.color.a = 0.5;
            }
            marker.lifetime = rclcpp::Duration::from_seconds(15.0);
            marker_array.markers.push_back(marker);
        }
        
        grid_visualization_pub_->publish(marker_array);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SimplifiedDualLidarOptimizer>());
    rclcpp::shutdown();
    return 0;
}