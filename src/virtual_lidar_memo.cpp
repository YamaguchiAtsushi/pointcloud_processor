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
#include <pcl/surface/gp3.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>
#include <random>
#include <unordered_map>

struct GridCell {
    double x, y, z;
    double weight;
    bool is_valid;
    double coverage_score;
    double redundancy_score;
    double angle_score;
    double distance_score;
    double slope_score;
    double total_score;
    
    pcl::Normal surface_normal;
    double slope_angle;
    double slope_direction;
    double surface_roughness;
    bool is_steep_slope;
    
    GridCell(double x=0, double y=0, double z=0) 
        : x(x), y(y), z(z), weight(1.0), is_valid(true), 
          coverage_score(0.0), redundancy_score(0.0), 
          angle_score(0.0), distance_score(0.0), slope_score(0.0), total_score(0.0),
          slope_angle(0.0), slope_direction(0.0), surface_roughness(0.0), is_steep_slope(false) {
        surface_normal.normal_x = 0.0;
        surface_normal.normal_y = 0.0;
        surface_normal.normal_z = 1.0;
    }
};

struct LidarPosition {
    double x, y, z;
    double pitch, yaw;
    double total_score;
    
    LidarPosition(double x=0, double y=0, double z=10, double pitch=-M_PI/2, double yaw=0) 
        : x(x), y(y), z(z), pitch(pitch), yaw(yaw), total_score(0.0) {}
};

struct DualLidarEvaluation {
    double coverage_score;
    double redundancy_score;
    double complementary_score;
    double slope_adaptation_score;
    double weighted_total_score;
    int covered_cells;
    int redundant_cells;
    int total_cells;
    int steep_slope_cells;
    int well_covered_steep_cells;
};

struct OptimizationParams {
    double distance_weight = 2.0;
    double angle_weight = 1.5;
    double visibility_weight = 4.0;
    double coverage_weight = 5.0;
    double redundancy_weight = 1.5;
    double complementary_weight = 6.0;
    double slope_adaptation_weight = 4.0;
    double min_distance = 1.0;
    double max_distance = 15.0;
    double optimal_angle = 60 * M_PI / 180.0;
    
    double grid_resolution = 0.3;
    double cell_weight_distance_threshold = 3.0;
    
    double steep_slope_threshold = 45.0 * M_PI / 180.0;
    double optimal_incidence_angle = 60.0 * M_PI / 180.0;
    double min_incidence_angle = 15.0 * M_PI / 180.0;
    double max_incidence_angle = 75.0 * M_PI / 180.0;
    double surface_normal_search_radius = 1.5;
    double slope_weight_multiplier = 3.0;
};

class GridBasedDualLidarOptimizer : public rclcpp::Node {
public:
    GridBasedDualLidarOptimizer() : Node("grid_based_dual_lidar_optimizer") {
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        
        declareParameters();
        updateParameters();
        
        excavation_area_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/excavation_area", 10,
            std::bind(&GridBasedDualLidarOptimizer::excavationAreaCallback, this, std::placeholders::_1));
            
        terrain_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/excavated_terrain", 10,
            std::bind(&GridBasedDualLidarOptimizer::terrainCallback, this, std::placeholders::_1));
        
        optimal_position_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>(
            "/optimal_mobile_lidar_position", 10);
        candidate_positions_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/mobile_lidar_candidate_positions", 10);
        grid_visualization_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/excavation_grid_visualization", 10);
        
        optimization_timer_ = this->create_wall_timer(
            std::chrono::seconds(3),
            std::bind(&GridBasedDualLidarOptimizer::runOptimization, this));
            
        rng_.seed(std::chrono::system_clock::now().time_since_epoch().count());
    }
    
private:
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr excavation_area_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr terrain_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr optimal_position_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr candidate_positions_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr grid_visualization_pub_;
    
    rclcpp::TimerBase::SharedPtr optimization_timer_;
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr excavation_area_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr terrain_cloud_;
    pcl::KdTreeFLANN<pcl::PointXYZRGB>::Ptr terrain_kdtree_;
    pcl::KdTreeFLANN<pcl::PointXYZRGB>::Ptr excavation_kdtree_;  // 追加
    pcl::PointCloud<pcl::Normal>::Ptr terrain_normals_;
    
    std::vector<std::vector<GridCell>> excavation_grid_;
    double grid_min_x_, grid_max_x_, grid_min_y_, grid_max_y_;
    double excavation_min_z_, excavation_max_z_;  // 追加
    int grid_width_, grid_height_;
    
    int num_candidates_;
    int max_iterations_;
    double search_radius_;
    double sensor_height_;
    double min_elevation_angle_;
    double max_elevation_angle_;
    OptimizationParams opt_params_;
    bool optimization_enabled_;
    
    double zx120_offset_x_, zx120_offset_y_, zx120_offset_z_;
    double zx120_pitch_, zx120_yaw_;
    
    double fov_horizontal_, fov_vertical_;
    double max_range_, angular_resolution_;
    double ray_step_size_, lidar_search_radius_;
    
    std::mt19937 rng_;
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> normal_estimator_;
    
    LidarPosition best_mobile_position_;
    LidarPosition zx120_lidar_position_;
    std::vector<LidarPosition> candidate_positions_;
    std::vector<DualLidarEvaluation> candidate_evaluations_;
    
    void declareParameters() {
        this->declare_parameter("optimization.num_candidates", 100);
        this->declare_parameter("optimization.max_iterations", 100);
        this->declare_parameter("optimization.search_radius", 5.0);
        this->declare_parameter("optimization.sensor_height", 1.0);
        this->declare_parameter("optimization.min_elevation_angle", -45.0);
        this->declare_parameter("optimization.max_elevation_angle", 30.0);
        this->declare_parameter("optimization.distance_weight", 2.0);
        this->declare_parameter("optimization.angle_weight", 2.0);
        this->declare_parameter("optimization.visibility_weight", 4.0);
        this->declare_parameter("optimization.coverage_weight", 5.0);
        this->declare_parameter("optimization.redundancy_weight", 1.0);
        this->declare_parameter("optimization.complementary_weight", 4.0);
        this->declare_parameter("optimization.slope_adaptation_weight", 4.0);
        this->declare_parameter("optimization.enabled", true);
        this->declare_parameter("optimization.grid_resolution", 0.3);
        
        this->declare_parameter("slope.steep_threshold_degrees", 20.0);
        this->declare_parameter("slope.optimal_incidence_degrees", 60.0);
        this->declare_parameter("slope.min_incidence_degrees", 15.0);
        this->declare_parameter("slope.max_incidence_degrees", 85.0);
        this->declare_parameter("slope.weight_multiplier", 3.0);
        this->declare_parameter("slope.normal_search_radius", 1.5);
        
        this->declare_parameter("zx120_lidar.offset_x", 0.4);
        this->declare_parameter("zx120_lidar.offset_y", 0.5);
        this->declare_parameter("zx120_lidar.offset_z", 3.5);
        this->declare_parameter("zx120_lidar.pitch", -M_PI/6);
        this->declare_parameter("zx120_lidar.yaw", 0.0);
        
        this->declare_parameter("lidar.fov_horizontal", 120.0);
        this->declare_parameter("lidar.fov_vertical", 90.0);
        this->declare_parameter("lidar.max_range", 50.0);
        this->declare_parameter("lidar.angular_resolution", 1.0);
        this->declare_parameter("lidar.ray_step_size", 0.1);
        this->declare_parameter("lidar.search_radius", 0.5);
    }
    
    void updateParameters() {
        num_candidates_ = this->get_parameter("optimization.num_candidates").as_int();
        max_iterations_ = this->get_parameter("optimization.max_iterations").as_int();
        search_radius_ = this->get_parameter("optimization.search_radius").as_double();
        sensor_height_ = this->get_parameter("optimization.sensor_height").as_double();
        min_elevation_angle_ = this->get_parameter("optimization.min_elevation_angle").as_double() * M_PI / 180.0;
        max_elevation_angle_ = this->get_parameter("optimization.max_elevation_angle").as_double() * M_PI / 180.0;
        opt_params_.distance_weight = this->get_parameter("optimization.distance_weight").as_double();
        opt_params_.angle_weight = this->get_parameter("optimization.angle_weight").as_double();
        opt_params_.visibility_weight = this->get_parameter("optimization.visibility_weight").as_double();
        opt_params_.coverage_weight = this->get_parameter("optimization.coverage_weight").as_double();
        opt_params_.redundancy_weight = this->get_parameter("optimization.redundancy_weight").as_double();
        opt_params_.complementary_weight = this->get_parameter("optimization.complementary_weight").as_double();
        opt_params_.slope_adaptation_weight = this->get_parameter("optimization.slope_adaptation_weight").as_double();
        opt_params_.grid_resolution = this->get_parameter("optimization.grid_resolution").as_double();
        optimization_enabled_ = this->get_parameter("optimization.enabled").as_bool();
        
        opt_params_.steep_slope_threshold = this->get_parameter("slope.steep_threshold_degrees").as_double() * M_PI / 180.0;
        opt_params_.optimal_incidence_angle = this->get_parameter("slope.optimal_incidence_degrees").as_double() * M_PI / 180.0;
        opt_params_.min_incidence_angle = this->get_parameter("slope.min_incidence_degrees").as_double() * M_PI / 180.0;
        opt_params_.max_incidence_angle = this->get_parameter("slope.max_incidence_degrees").as_double() * M_PI / 180.0;
        opt_params_.slope_weight_multiplier = this->get_parameter("slope.weight_multiplier").as_double();
        opt_params_.surface_normal_search_radius = this->get_parameter("slope.normal_search_radius").as_double();
        
        zx120_offset_x_ = this->get_parameter("zx120_lidar.offset_x").as_double();
        zx120_offset_y_ = this->get_parameter("zx120_lidar.offset_y").as_double();
        zx120_offset_z_ = this->get_parameter("zx120_lidar.offset_z").as_double();
        zx120_pitch_ = this->get_parameter("zx120_lidar.pitch").as_double();
        zx120_yaw_ = this->get_parameter("zx120_lidar.yaw").as_double();
        
        fov_horizontal_ = this->get_parameter("lidar.fov_horizontal").as_double() * M_PI / 180.0;
        fov_vertical_ = this->get_parameter("lidar.fov_vertical").as_double() * M_PI / 180.0;
        max_range_ = this->get_parameter("lidar.max_range").as_double();
        angular_resolution_ = this->get_parameter("lidar.angular_resolution").as_double() * M_PI / 180.0;
        ray_step_size_ = this->get_parameter("lidar.ray_step_size").as_double();
        lidar_search_radius_ = this->get_parameter("lidar.search_radius").as_double();
    }
    
    void excavationAreaCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        excavation_area_.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::fromROSMsg(*msg, *excavation_area_);
        
        if (excavation_area_->empty()) return;
        
        // 掘削エリアのKDTreeを構築
        excavation_kdtree_.reset(new pcl::KdTreeFLANN<pcl::PointXYZRGB>);
        try {
            excavation_kdtree_->setInputCloud(excavation_area_);
            RCLCPP_INFO(this->get_logger(), "Built excavation KD-tree with %zu points", excavation_area_->size());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to build excavation KD-tree: %s", e.what());
            return;
        }
        
        computeTerrainNormals();
        generateExcavationGrid();
    }
    
    void terrainCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        terrain_cloud_.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::fromROSMsg(*msg, *terrain_cloud_);
        
        if (terrain_cloud_->empty()) return;
        
        terrain_kdtree_.reset(new pcl::KdTreeFLANN<pcl::PointXYZRGB>);
        try {
            terrain_kdtree_->setInputCloud(terrain_cloud_);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to build terrain KD-tree: %s", e.what());
        }
    }
    
    void computeTerrainNormals() {
        if (!excavation_area_ || excavation_area_->empty()) return;
        
        try {
            pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGB>());
            kdtree->setInputCloud(excavation_area_);
            
            normal_estimator_.setInputCloud(excavation_area_);
            normal_estimator_.setSearchMethod(kdtree);
            normal_estimator_.setKSearch(0);
            normal_estimator_.setRadiusSearch(opt_params_.surface_normal_search_radius);
            
            terrain_normals_.reset(new pcl::PointCloud<pcl::Normal>);
            normal_estimator_.compute(*terrain_normals_);
            
            for (auto& normal : terrain_normals_->points) {
                if (normal.normal_z < 0) {
                    normal.normal_x = -normal.normal_x;
                    normal.normal_y = -normal.normal_y;
                    normal.normal_z = -normal.normal_z;
                }
            }
            
            RCLCPP_INFO(this->get_logger(), "Computed normals for %zu points with search radius %.2f", 
                       terrain_normals_->size(), opt_params_.surface_normal_search_radius);
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to compute terrain normals: %s", e.what());
            terrain_normals_.reset(new pcl::PointCloud<pcl::Normal>);
        }
    }
    
    void generateExcavationGrid() {
        if (!excavation_area_ || excavation_area_->empty() || !excavation_kdtree_) return;
        
        // 掘削エリアの範囲を計算（Z軸も含む）
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
        
        // グリッドの範囲を拡張
        double margin = opt_params_.grid_resolution;
        grid_min_x_ -= margin; grid_max_x_ += margin;
        grid_min_y_ -= margin; grid_max_y_ += margin;
        
        grid_width_ = static_cast<int>(std::ceil((grid_max_x_ - grid_min_x_) / opt_params_.grid_resolution)) + 1;
        grid_height_ = static_cast<int>(std::ceil((grid_max_y_ - grid_min_y_) / opt_params_.grid_resolution)) + 1;
        
        RCLCPP_INFO(this->get_logger(), 
                   "Excavation area bounds: X[%.2f, %.2f], Y[%.2f, %.2f], Z[%.2f, %.2f]", 
                   grid_min_x_, grid_max_x_, grid_min_y_, grid_max_y_, excavation_min_z_, excavation_max_z_);
        
        excavation_grid_.clear();
        excavation_grid_.resize(grid_height_, std::vector<GridCell>(grid_width_));
        
        int steep_slope_count = 0;
        int total_valid_cells = 0;
        int failed_z_estimation = 0;
        
        for (int i = 0; i < grid_height_; ++i) {
            for (int j = 0; j < grid_width_; ++j) {
                double x = grid_min_x_ + j * opt_params_.grid_resolution;
                double y = grid_min_y_ + i * opt_params_.grid_resolution;
                
                // より柔軟なZ座標推定
                double estimated_z;
                bool z_valid = estimateZCoordinateRobust(x, y, estimated_z);
                
                if (!z_valid) {
                    excavation_grid_[i][j].is_valid = false;
                    failed_z_estimation++;
                    continue;
                }
                
                excavation_grid_[i][j] = GridCell(x, y, estimated_z);
                excavation_grid_[i][j].is_valid = true;
                
                computeCellSlopeInfo(excavation_grid_[i][j], x, y, estimated_z, excavation_kdtree_);
                double cell_weight = calculateCellWeight(x, y, estimated_z, excavation_grid_[i][j]);
                excavation_grid_[i][j].weight = cell_weight;
                
                total_valid_cells++;
                if (excavation_grid_[i][j].is_steep_slope) {
                    steep_slope_count++;
                }
            }
        }
        
        RCLCPP_INFO(this->get_logger(), 
                   "Generated grid: %dx%d, Valid cells: %d, Failed Z estimation: %d, Steep slopes: %d (%.1f%%), Threshold: %.1f degrees", 
                   grid_width_, grid_height_, total_valid_cells, failed_z_estimation, steep_slope_count,
                   total_valid_cells > 0 ? (steep_slope_count * 100.0 / total_valid_cells) : 0.0,
                   opt_params_.steep_slope_threshold * 180.0 / M_PI);
        
        publishGridVisualization();
    }
    
    // 点群のハッシュ値を計算して変化を検出
    std::size_t calculatePointCloudHash(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud) {
        if (!cloud || cloud->empty()) return 0;
        
        std::size_t hash = 0;
        std::hash<float> hasher;
        
        // 点群サイズをハッシュに含める
        hash ^= std::hash<std::size_t>{}(cloud->size()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        
        // 全ての点をハッシュ計算に含めるとコストが高いため、
        // サンプリングして代表的な点のみを使用
        std::size_t step = std::max(1UL, cloud->size() / 100);  // 最大100点をサンプル
        
        for (std::size_t i = 0; i < cloud->size(); i += step) {
            const auto& point = cloud->points[i];
            hash ^= hasher(point.x) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= hasher(point.y) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= hasher(point.z) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        
        return hash;
    }
    
    bool estimateZCoordinateRobust(double x, double y, double& estimated_z) {
        pcl::PointXYZRGB search_point;
        search_point.x = x; search_point.y = y; search_point.z = (excavation_min_z_ + excavation_max_z_) / 2.0;  // 中央値を使用
        
        std::vector<int> point_indices;
        std::vector<float> point_distances;
        
        // 段階的に検索半径を拡大
        std::vector<double> search_radii = {
            opt_params_.grid_resolution * 1.5,
            opt_params_.grid_resolution * 3.0,
            opt_params_.grid_resolution * 5.0,
            std::max(5.0, opt_params_.grid_resolution * 10.0)
        };
        
        for (double search_radius : search_radii) {
            if (excavation_kdtree_->radiusSearch(search_point, search_radius, 
                                               point_indices, point_distances) > 0) {
                
                if (point_distances.size() >= 3) {  // 最低3点は欲しい
                    // 重み付き平均を計算（より近い点により大きな重み）
                    double sum_z = 0.0, sum_weight = 0.0;
                    
                    for (size_t i = 0; i < point_indices.size(); ++i) {
                        const auto& point = excavation_area_->points[point_indices[i]];
                        double distance_2d = sqrt(pow(point.x - x, 2) + pow(point.y - y, 2));
                        
                        // XY平面での距離が検索半径の半分以内の点を優先
                        if (distance_2d <= search_radius * 0.5) {
                            double weight = (distance_2d < 1e-6) ? 1000.0 : 1.0 / (distance_2d * distance_2d + 0.01);
                            sum_z += point.z * weight;
                            sum_weight += weight;
                        }
                    }
                    
                    if (sum_weight > 0) {
                        estimated_z = sum_z / sum_weight;
                        
                        // 推定値が合理的な範囲にあるかチェック
                        if (estimated_z >= excavation_min_z_ - 1.0 && 
                            estimated_z <= excavation_max_z_ + 1.0) {
                            return true;
                        }
                    }
                }
            }
        }
        
        // 最後の手段：最近傍の点を使用
        if (excavation_kdtree_->nearestKSearch(search_point, 1, point_indices, point_distances) > 0) {
            estimated_z = excavation_area_->points[point_indices[0]].z;
            RCLCPP_DEBUG(this->get_logger(), 
                        "Using nearest neighbor fallback for point (%.2f, %.2f) -> Z=%.2f", 
                        x, y, estimated_z);
            return true;
        }
        
        return false;
    }
    
    void computeCellSlopeInfo(GridCell& cell, double x, double y, double z, 
                             pcl::KdTreeFLANN<pcl::PointXYZRGB>::Ptr kdtree) {
        if (!terrain_normals_ || terrain_normals_->empty()) {
            cell.surface_normal.normal_x = 0.0;
            cell.surface_normal.normal_y = 0.0;
            cell.surface_normal.normal_z = 1.0;
            cell.slope_angle = 0.0;
            cell.slope_direction = 0.0;
            cell.surface_roughness = 0.0;
            cell.is_steep_slope = false;
            return;
        }
        
        pcl::PointXYZRGB search_point;
        search_point.x = x; search_point.y = y; search_point.z = z;
        
        std::vector<int> point_indices;
        std::vector<float> point_distances;
        
        if (kdtree->radiusSearch(search_point, opt_params_.surface_normal_search_radius, 
                                point_indices, point_distances) > 0) {
            
            double sum_nx = 0.0, sum_ny = 0.0, sum_nz = 0.0;
            double sum_roughness = 0.0;
            int valid_normals = 0;
            
            for (size_t i = 0; i < point_indices.size() && i < 15; ++i) {
                int idx = point_indices[i];
                if (idx >= 0 && idx < static_cast<int>(terrain_normals_->size())) {
                    const auto& normal = terrain_normals_->points[idx];
                    
                    if (std::isfinite(normal.normal_x) && std::isfinite(normal.normal_y) && 
                        std::isfinite(normal.normal_z) && 
                        (normal.normal_x*normal.normal_x + normal.normal_y*normal.normal_y + 
                         normal.normal_z*normal.normal_z) > 0.5) {
                        
                        sum_nx += normal.normal_x;
                        sum_ny += normal.normal_y;
                        sum_nz += normal.normal_z;
                        
                        double dot = normal.normal_z;
                        sum_roughness += (1.0 - abs(dot));
                        valid_normals++;
                    }
                }
            }
            
            if (valid_normals > 0) {
                double norm = sqrt(sum_nx*sum_nx + sum_ny*sum_ny + sum_nz*sum_nz);
                if (norm > 1e-6) {
                    cell.surface_normal.normal_x = sum_nx / norm;
                    cell.surface_normal.normal_y = sum_ny / norm;
                    cell.surface_normal.normal_z = sum_nz / norm;
                } else {
                    cell.surface_normal.normal_x = 0.0;
                    cell.surface_normal.normal_y = 0.0;
                    cell.surface_normal.normal_z = 1.0;
                }
                cell.surface_roughness = sum_roughness / valid_normals;
            }
        } else {
            cell.surface_normal.normal_x = 0.0;
            cell.surface_normal.normal_y = 0.0;
            cell.surface_normal.normal_z = 1.0;
            cell.surface_roughness = 0.0;
        }
        
        double normal_z_abs = static_cast<double>(abs(cell.surface_normal.normal_z));
        cell.slope_angle = acos(std::max(0.0, std::min(1.0, normal_z_abs)));
        
        if (abs(cell.surface_normal.normal_x) > 1e-6 || abs(cell.surface_normal.normal_y) > 1e-6) {
            cell.slope_direction = atan2(cell.surface_normal.normal_y, cell.surface_normal.normal_x);
        } else {
            cell.slope_direction = 0.0;
        }
        
        cell.is_steep_slope = (cell.slope_angle > opt_params_.steep_slope_threshold);
    }
    
    double calculateCellWeight(double x, double y, double z, const GridCell& cell) {
        double weight = 1.0;
        
        double center_x = (grid_min_x_ + grid_max_x_) / 2.0;
        double center_y = (grid_min_y_ + grid_max_y_) / 2.0;
        double distance_to_center = sqrt(pow(x - center_x, 2) + pow(y - center_y, 2));
        double max_distance = sqrt(pow(grid_max_x_ - grid_min_x_, 2) + pow(grid_max_y_ - grid_min_y_, 2)) / 2.0;
        
        weight *= (1.0 + 2.0 * (1.0 - distance_to_center / max_distance));
        
        double depth_factor = std::max(0.5, 1.0 - z / 10.0);
        weight *= depth_factor;
        
        if (cell.is_steep_slope) weight *= opt_params_.slope_weight_multiplier;
        
        double roughness_factor = 1.0 + cell.surface_roughness;
        weight *= roughness_factor;
        
        return std::max(0.1, weight);
    }
    
    bool getZX120Position() {
        try {
            geometry_msgs::msg::TransformStamped transform_stamped = 
                tf_buffer_->lookupTransform("map", "zx120/base_link", tf2::TimePointZero, 
                                          tf2::durationFromSec(0.1));
            
            zx120_lidar_position_.x = transform_stamped.transform.translation.x + zx120_offset_x_;
            zx120_lidar_position_.y = transform_stamped.transform.translation.y + zx120_offset_y_;
            zx120_lidar_position_.z = transform_stamped.transform.translation.z + zx120_offset_z_;
            zx120_lidar_position_.pitch = zx120_pitch_;
            zx120_lidar_position_.yaw = zx120_yaw_;
            
            return true;
        } catch (tf2::TransformException& ex) {
            return false;
        }
    }
    
    void runOptimization() {
        if (!optimization_enabled_ || excavation_grid_.empty() || 
            !terrain_cloud_ || terrain_cloud_->empty() || !terrain_kdtree_) return;
        
        updateParameters();
        
        if (!getZX120Position()) return;
        
        std::vector<LidarPosition> candidates = generateCandidatePositions();
        
        double best_score = -std::numeric_limits<double>::infinity();
        LidarPosition best_candidate;
        
        candidate_evaluations_.clear();
        
        for (auto& candidate : candidates) {
            DualLidarEvaluation evaluation = evaluateDualLidarSetupGrid(zx120_lidar_position_, candidate);
            candidate.total_score = evaluation.weighted_total_score;
            candidate_evaluations_.push_back(evaluation);
            
            if (candidate.total_score > best_score) {
                best_score = candidate.total_score;
                best_candidate = candidate;
            }
        }
        
        best_mobile_position_ = best_candidate;
        candidate_positions_ = candidates;
        
        publishOptimalPosition();
        publishCandidatePositions();
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
        
        // 碁盤の目状の配置のための間隔を計算
        // num_candidates_から適切なグリッド数を決定
        int grid_candidates_per_side = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(num_candidates_))));
        
        double x_step = (expanded_max_x - expanded_min_x) / (grid_candidates_per_side - 1);
        double y_step = (expanded_max_y - expanded_min_y) / (grid_candidates_per_side - 1);
        
        RCLCPP_INFO(this->get_logger(), 
                   "Generating %dx%d grid of candidates (target: %d) with steps: X=%.2f, Y=%.2f", 
                   grid_candidates_per_side, grid_candidates_per_side, num_candidates_, x_step, y_step);
        
        for (int i = 0; i < grid_candidates_per_side; ++i) {
            for (int j = 0; j < grid_candidates_per_side; ++j) {
                double x = expanded_min_x + i * x_step;
                double y = expanded_min_y + j * y_step;
                
                // ZX120との距離チェック
                double dist_to_zx120 = sqrt(pow(x - zx120_lidar_position_.x, 2) + 
                                          pow(y - zx120_lidar_position_.y, 2));
                if (dist_to_zx120 < 2.0) continue;
                
                // 掘削エリア内の点は除外
                if (x >= grid_min_x_ && x <= grid_max_x_ && y >= grid_min_y_ && y <= grid_max_y_) continue;
                
                double ground_z = getGroundHeight(x, y);
                double z = ground_z + sensor_height_;
                
                // 掘削エリア中心への視線角度を計算
                double dx = center_x - x;
                double dy = center_y - y;
                double dz = center_z - z;
                double horizontal_distance = sqrt(dx*dx + dy*dy);
                
                if (horizontal_distance < 0.1) continue; // 中心に近すぎる場合をスキップ
                
                double elevation_angle = atan2(-dz, horizontal_distance);
                
                if (elevation_angle >= min_elevation_angle_ && elevation_angle <= max_elevation_angle_) {
                    double pitch = -M_PI/2 + elevation_angle;
                    double yaw = atan2(dy, dx);
                    candidates.emplace_back(x, y, z, pitch, yaw);
                }
            }
        }
        
        RCLCPP_INFO(this->get_logger(), "Generated %zu valid grid-based candidate positions", candidates.size());
        
        // 候補数が少なすぎる場合は、より密なグリッドを生成
        if (candidates.size() < static_cast<std::size_t>(num_candidates_ * 0.3)) {
            RCLCPP_WARN(this->get_logger(), 
                       "Grid generated too few candidates (%zu), generating denser grid", candidates.size());
            return generateDenseCandidateGrid(expanded_min_x, expanded_max_x, 
                                            expanded_min_y, expanded_max_y, 
                                            center_x, center_y, center_z);
        }
        
        return candidates;
    }
    
    // より密なグリッドを生成する補助関数
    std::vector<LidarPosition> generateDenseCandidateGrid(double min_x, double max_x, 
                                                         double min_y, double max_y,
                                                         double center_x, double center_y, double center_z) {
        std::vector<LidarPosition> candidates;
        
        // より細かい間隔で生成
        double spacing = std::min(search_radius_ * 0.5, 2.0);  // 0.5m間隔または2m間隔の小さい方
        
        for (double x = min_x; x <= max_x; x += spacing) {
            for (double y = min_y; y <= max_y; y += spacing) {
                double dist_to_zx120 = sqrt(pow(x - zx120_lidar_position_.x, 2) + 
                                          pow(y - zx120_lidar_position_.y, 2));
                if (dist_to_zx120 < 2.0) continue;
                
                if (x >= grid_min_x_ && x <= grid_max_x_ && y >= grid_min_y_ && y <= grid_max_y_) continue;
                
                double ground_z = getGroundHeight(x, y);
                double z = ground_z + sensor_height_;
                
                double dx = center_x - x;
                double dy = center_y - y;
                double dz = center_z - z;
                double horizontal_distance = sqrt(dx*dx + dy*dy);
                
                if (horizontal_distance < 0.1) continue;
                
                double elevation_angle = atan2(-dz, horizontal_distance);
                
                if (elevation_angle >= min_elevation_angle_ && elevation_angle <= max_elevation_angle_) {
                    double pitch = -M_PI/2 + elevation_angle;
                    double yaw = atan2(dy, dx);
                    candidates.emplace_back(x, y, z, pitch, yaw);
                }
            }
        }
        
        RCLCPP_INFO(this->get_logger(), "Generated %zu candidates with dense grid (spacing: %.2f)", 
                   candidates.size(), spacing);
        
        return candidates;
    }
    
    double getGroundHeight(double x, double y) {
        if (!terrain_cloud_ || terrain_cloud_->empty()) return 0.0;
        
        pcl::PointXYZRGB search_point;
        search_point.x = x; search_point.y = y; search_point.z = 0;
        
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
    
    DualLidarEvaluation evaluateDualLidarSetupGrid(const LidarPosition& zx120_pos, 
                                                  const LidarPosition& mobile_pos) {
        DualLidarEvaluation evaluation = {};
        
        double total_weighted_coverage = 0.0;
        double total_weighted_redundancy = 0.0;
        double total_slope_adaptation = 0.0;
        double total_weight = 0.0;
        int covered_cells = 0;
        int redundant_cells = 0;
        int total_valid_cells = 0;
        int steep_slope_cells = 0;
        int well_covered_steep_cells = 0;
        
        for (int i = 0; i < grid_height_; ++i) {
            for (int j = 0; j < grid_width_; ++j) {
                GridCell& cell = excavation_grid_[i][j];
                
                if (!cell.is_valid) continue;
                
                total_valid_cells++;
                if (cell.is_steep_slope) steep_slope_cells++;
                
                double zx120_score = evaluateGridCellFromLidarWithSlope(zx120_pos, cell);
                double mobile_score = evaluateGridCellFromLidarWithSlope(mobile_pos, cell);
                
                bool zx120_visible = (zx120_score > 0);
                bool mobile_visible = (mobile_score > 0);
                
                cell.coverage_score = std::max(zx120_score, mobile_score);
                cell.redundancy_score = (zx120_visible && mobile_visible) ? 
                    std::min(zx120_score, mobile_score) : 0.0;
                
                cell.slope_score = calculateSlopeAdaptationScore(cell, zx120_pos, mobile_pos, 
                                                               zx120_score, mobile_score);
                
                cell.total_score = cell.coverage_score + cell.redundancy_score + cell.slope_score;
                
                if (zx120_visible || mobile_visible) {
                    covered_cells++;
                    total_weighted_coverage += cell.coverage_score * cell.weight;
                    
                    if (cell.is_steep_slope && cell.slope_score > 0.5) {
                        well_covered_steep_cells++;
                    }
                }
                
                if (zx120_visible && mobile_visible) {
                    redundant_cells++;
                    total_weighted_redundancy += cell.redundancy_score * cell.weight;
                }
                
                total_slope_adaptation += cell.slope_score * cell.weight;
                total_weight += cell.weight;
            }
        }
        
        evaluation.covered_cells = covered_cells;
        evaluation.redundant_cells = redundant_cells;
        evaluation.total_cells = total_valid_cells;
        evaluation.steep_slope_cells = steep_slope_cells;
        evaluation.well_covered_steep_cells = well_covered_steep_cells;
        
        if (total_weight > 0) {
            evaluation.coverage_score = total_weighted_coverage / total_weight;
            evaluation.redundancy_score = total_weighted_redundancy / total_weight;
            evaluation.slope_adaptation_score = total_slope_adaptation / total_weight;
            
            double complementary_contribution = 0.0;
            for (int i = 0; i < grid_height_; ++i) {
                for (int j = 0; j < grid_width_; ++j) {
                    const GridCell& cell = excavation_grid_[i][j];
                    if (!cell.is_valid) continue;
                    
                    double zx120_score = evaluateGridCellFromLidarWithSlope(zx120_pos, cell);
                    double mobile_score = evaluateGridCellFromLidarWithSlope(mobile_pos, cell);
                    
                    if (zx120_score <= 0 && mobile_score > 0) {
                        complementary_contribution += mobile_score * cell.weight;
                    }
                }
            }
            evaluation.complementary_score = complementary_contribution / total_weight;
        } else {
            evaluation.coverage_score = 0.0;
            evaluation.redundancy_score = 0.0;
            evaluation.complementary_score = 0.0;
            evaluation.slope_adaptation_score = 0.0;
        }
        
        evaluation.weighted_total_score = 
            opt_params_.coverage_weight * evaluation.coverage_score +
            opt_params_.redundancy_weight * evaluation.redundancy_score +
            opt_params_.complementary_weight * evaluation.complementary_score +
            opt_params_.slope_adaptation_weight * evaluation.slope_adaptation_score;
        
        return evaluation;
    }
    
    double evaluateGridCellFromLidarWithSlope(const LidarPosition& lidar_pos, const GridCell& cell) {
        double base_score = evaluateGridCellFromLidar(lidar_pos, cell);
        if (base_score <= 0) return 0.0;
        
        double dx = cell.x - lidar_pos.x;
        double dy = cell.y - lidar_pos.y;
        double dz = cell.z - lidar_pos.z;
        double distance = sqrt(dx*dx + dy*dy + dz*dz);
        
        double ray_x = dx / distance;
        double ray_y = dy / distance;
        double ray_z = dz / distance;
        
        double dot_product = -(ray_x * cell.surface_normal.normal_x + 
                              ray_y * cell.surface_normal.normal_y + 
                              ray_z * cell.surface_normal.normal_z);
        
        double incidence_angle = acos(std::max(0.0, std::min(1.0, abs(dot_product))));
        
        if (incidence_angle < opt_params_.min_incidence_angle || 
            incidence_angle > opt_params_.max_incidence_angle) {
            return 0.0;
        }
        
        double angle_difference = abs(incidence_angle - opt_params_.optimal_incidence_angle);
        double angle_penalty = exp(-angle_difference / (M_PI/6));
        
        if (cell.is_steep_slope) {
            angle_penalty = exp(-angle_difference / (M_PI/9));
        }
        
        return base_score * angle_penalty;
    }
    
    double calculateSlopeAdaptationScore(const GridCell& cell, 
                                       const LidarPosition& /* zx120_pos */, 
                                       const LidarPosition& /* mobile_pos */,
                                       double zx120_score, double mobile_score) {
        if (!cell.is_steep_slope) return 0.1;
        
        double adaptation_score = 0.0;
        
        if (zx120_score > 0.5 && mobile_score > 0.5) {
            adaptation_score = 1.0;
        } else if (std::max(zx120_score, mobile_score) > 0.7) {
            adaptation_score = 0.7;
        } else if (zx120_score > 0 || mobile_score > 0) {
            adaptation_score = 0.3;
        }
        
        adaptation_score *= (1.0 + cell.surface_roughness * 0.5);
        
        return adaptation_score;
    }
    
    double evaluateGridCellFromLidar(const LidarPosition& lidar_pos, const GridCell& cell) {
        double dx = cell.x - lidar_pos.x;
        double dy = cell.y - lidar_pos.y;
        double dz = cell.z - lidar_pos.z;
        double distance = sqrt(dx*dx + dy*dy + dz*dz);
        
        if (distance < opt_params_.min_distance || distance > opt_params_.max_distance) return 0.0;
        
        double azimuth = atan2(dy, dx);
        double elevation = atan2(dz, sqrt(dx*dx + dy*dy));
        
        double azimuth_diff = fmod(azimuth - lidar_pos.yaw + M_PI, 2*M_PI) - M_PI;
        double elevation_diff = elevation - lidar_pos.pitch;
        
        if (abs(azimuth_diff) > fov_horizontal_/2 || abs(elevation_diff) > fov_vertical_/2) return 0.0;
        
        if (!checkVisibility(lidar_pos, cell)) return 0.0;
        
        double distance_score = 1.0 / (1.0 + distance/10.0);
        double angle_score = cos(abs(elevation_diff)) * cos(abs(azimuth_diff));
        double visibility_score = 1.0;
        
        double total_score = opt_params_.distance_weight * distance_score +
                           opt_params_.angle_weight * angle_score +
                           opt_params_.visibility_weight * visibility_score;
        
        return std::max(0.0, total_score);
    }
    
    bool checkVisibility(const LidarPosition& lidar_pos, const GridCell& cell) {
        double dx = cell.x - lidar_pos.x;
        double dy = cell.y - lidar_pos.y;
        double dz = cell.z - lidar_pos.z;
        double distance = sqrt(dx*dx + dy*dy + dz*dz);
        
        double norm_dx = dx / distance;
        double norm_dy = dy / distance;
        double norm_dz = dz / distance;
        
        double step_distance = ray_step_size_;
        while (step_distance < distance - ray_step_size_) {
            double check_x = lidar_pos.x + norm_dx * step_distance;
            double check_y = lidar_pos.y + norm_dy * step_distance;
            double check_z = lidar_pos.z + norm_dz * step_distance;
            
            pcl::PointXYZRGB search_point;
            search_point.x = check_x;
            search_point.y = check_y;
            search_point.z = check_z;
            
            std::vector<int> point_indices;
            std::vector<float> point_distances;
            
            if (terrain_kdtree_->radiusSearch(search_point, lidar_search_radius_, 
                                            point_indices, point_distances) > 0) {
                return false;
            }
            
            step_distance += ray_step_size_;
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
        zx120_marker.color.g = 0.0;
        zx120_marker.color.b = 1.0;
        zx120_marker.color.a = 1.0;
        
        zx120_marker.lifetime = rclcpp::Duration::from_seconds(10.0);
        marker_array.markers.push_back(zx120_marker);
        
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
            
            double normalized_score = std::max(0.0, std::min(1.0, 
                (candidate_positions_[i].total_score + 50) / 100.0));
            marker.color.r = 1.0 - normalized_score;
            marker.color.g = normalized_score;
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
        best_marker.color.g = 1.0;
        best_marker.color.b = 0.0;
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
        for (int i = 0; i < grid_height_; ++i) {
            for (int j = 0; j < grid_width_; ++j) {
                const GridCell& cell = excavation_grid_[i][j];
                
                if (!cell.is_valid) continue;
                
                visualization_msgs::msg::Marker marker;
                marker.header.stamp = this->now();
                marker.header.frame_id = "map";
                marker.ns = "excavation_grid";
                marker.id = marker_id++;
                marker.type = visualization_msgs::msg::Marker::CUBE;
                marker.action = visualization_msgs::msg::Marker::ADD;
                
                marker.pose.position.x = cell.x;
                marker.pose.position.y = cell.y;
                marker.pose.position.z = cell.z;
                marker.pose.orientation.w = 1.0;
                
                marker.scale.x = opt_params_.grid_resolution * 0.8;
                marker.scale.y = opt_params_.grid_resolution * 0.8;
                marker.scale.z = 0.1;
                
                if (cell.is_steep_slope) {
                    double slope_intensity = std::min(1.0, cell.slope_angle / (M_PI/2));
                    marker.color.r = 0.8 + 0.2 * slope_intensity;
                    marker.color.g = 0.2;
                    marker.color.b = 0.2;
                } else {
                    double normalized_weight = std::min(1.0, cell.weight / 3.0);
                    marker.color.r = 1.0 - normalized_weight;
                    marker.color.g = 0.5;
                    marker.color.b = normalized_weight;
                }
                marker.color.a = 0.6;
                
                marker.lifetime = rclcpp::Duration::from_seconds(15.0);
                marker_array.markers.push_back(marker);
            }
        }
        
        grid_visualization_pub_->publish(marker_array);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GridBasedDualLidarOptimizer>());
    rclcpp::shutdown();
    return 0;
}