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
    double score_zx120;
    double score_mobile;
    double combined_score;
    
    pcl::Normal surface_normal;
    double slope_angle;
    double surface_roughness;
    
    GridCell(double x=0, double y=0, double z=0) 
        : x(x), y(y), z(z), weight(1.0), is_valid(true), 
          score_zx120(0.0), score_mobile(0.0), combined_score(0.0),
          slope_angle(0.0), surface_roughness(0.0) {
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

struct SimplifiedEvaluation {
    double total_score;
    double coverage_ratio;
    double average_score;
    int covered_cells;
    int total_cells;
};

struct OptimizationParams {
    // Simplified evaluation parameters
    double alpha = 1.0;  // Weight for angle term (α*θ)
    double beta = 10.0;  // Weight for distance term (β*(1/L))
    
    // Basic constraints
    double min_distance = 1.0;
    double max_distance = 15.0;
    double grid_resolution = 0.3;
    
    // Surface normal computation
    double surface_normal_search_radius = 1.5;
    
    // FOV and visibility
    double fov_horizontal = 120.0 * M_PI / 180.0;
    double fov_vertical = 90.0 * M_PI / 180.0;
    double ray_step_size = 0.1;
    double visibility_search_radius = 0.5;
};

class SimplifiedDualLidarOptimizer : public rclcpp::Node {
public:
    SimplifiedDualLidarOptimizer() : Node("simplified_dual_lidar_optimizer") {
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        
        declareParameters();
        updateParameters();
        
        excavation_area_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/excavation_area", 10,
            std::bind(&SimplifiedDualLidarOptimizer::excavationAreaCallback, this, std::placeholders::_1));
            
        terrain_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/excavated_terrain", 10,
            std::bind(&SimplifiedDualLidarOptimizer::terrainCallback, this, std::placeholders::_1));
        
        optimal_position_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>(
            "/optimal_mobile_lidar_position", 10);
        candidate_positions_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/mobile_lidar_candidate_positions", 10);
        grid_visualization_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/excavation_grid_visualization", 10);
        
        optimization_timer_ = this->create_wall_timer(
            std::chrono::seconds(3),
            std::bind(&SimplifiedDualLidarOptimizer::runOptimization, this));
            
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
    pcl::KdTreeFLANN<pcl::PointXYZRGB>::Ptr excavation_kdtree_;
    pcl::PointCloud<pcl::Normal>::Ptr terrain_normals_;
    
    std::vector<std::vector<GridCell>> excavation_grid_;
    double grid_min_x_, grid_max_x_, grid_min_y_, grid_max_y_;
    double excavation_min_z_, excavation_max_z_;
    int grid_width_, grid_height_;
    
    int num_candidates_;
    double search_radius_;
    double sensor_height_;
    double min_elevation_angle_;
    double max_elevation_angle_;
    OptimizationParams opt_params_;
    bool optimization_enabled_;
    
    double zx120_offset_x_, zx120_offset_y_, zx120_offset_z_;
    double zx120_pitch_, zx120_yaw_;
    
    std::mt19937 rng_;
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> normal_estimator_;
    
    LidarPosition best_mobile_position_;
    LidarPosition zx120_lidar_position_;
    std::vector<LidarPosition> candidate_positions_;
    
    void declareParameters() {
        // Simplified evaluation parameters
        this->declare_parameter("evaluation.alpha", 1.0);
        this->declare_parameter("evaluation.beta", 10.0);
        
        // Basic optimization parameters
        this->declare_parameter("optimization.num_candidates", 100);
        this->declare_parameter("optimization.search_radius", 5.0);
        this->declare_parameter("optimization.sensor_height", 1.0);
        this->declare_parameter("optimization.min_elevation_angle", -45.0);
        this->declare_parameter("optimization.max_elevation_angle", 30.0);
        this->declare_parameter("optimization.enabled", true);
        this->declare_parameter("optimization.grid_resolution", 0.3);
        
        // Distance constraints
        this->declare_parameter("constraints.min_distance", 1.0);
        this->declare_parameter("constraints.max_distance", 15.0);
        
        // Surface normal computation
        this->declare_parameter("surface.normal_search_radius", 1.5);
        
        // ZX120 LiDAR position
        this->declare_parameter("zx120_lidar.offset_x", 0.4);
        this->declare_parameter("zx120_lidar.offset_y", 0.5);
        this->declare_parameter("zx120_lidar.offset_z", 3.5);
        this->declare_parameter("zx120_lidar.pitch", -M_PI/6);
        this->declare_parameter("zx120_lidar.yaw", 0.0);
        
        // LiDAR specifications
        this->declare_parameter("lidar.fov_horizontal", 120.0);
        this->declare_parameter("lidar.fov_vertical", 90.0);
        this->declare_parameter("lidar.ray_step_size", 0.1);
        this->declare_parameter("lidar.visibility_search_radius", 0.5);
    }
    
    void updateParameters() {
        // Simplified evaluation parameters
        opt_params_.alpha = this->get_parameter("evaluation.alpha").as_double();
        opt_params_.beta = this->get_parameter("evaluation.beta").as_double();
        
        // Basic optimization parameters
        num_candidates_ = this->get_parameter("optimization.num_candidates").as_int();
        search_radius_ = this->get_parameter("optimization.search_radius").as_double();
        sensor_height_ = this->get_parameter("optimization.sensor_height").as_double();
        min_elevation_angle_ = this->get_parameter("optimization.min_elevation_angle").as_double() * M_PI / 180.0;
        max_elevation_angle_ = this->get_parameter("optimization.max_elevation_angle").as_double() * M_PI / 180.0;
        opt_params_.grid_resolution = this->get_parameter("optimization.grid_resolution").as_double();
        optimization_enabled_ = this->get_parameter("optimization.enabled").as_bool();
        
        // Distance constraints
        opt_params_.min_distance = this->get_parameter("constraints.min_distance").as_double();
        opt_params_.max_distance = this->get_parameter("constraints.max_distance").as_double();
        
        // Surface normal computation
        opt_params_.surface_normal_search_radius = this->get_parameter("surface.normal_search_radius").as_double();
        
        // ZX120 LiDAR position
        zx120_offset_x_ = this->get_parameter("zx120_lidar.offset_x").as_double();
        zx120_offset_y_ = this->get_parameter("zx120_lidar.offset_y").as_double();
        zx120_offset_z_ = this->get_parameter("zx120_lidar.offset_z").as_double();
        zx120_pitch_ = this->get_parameter("zx120_lidar.pitch").as_double();
        zx120_yaw_ = this->get_parameter("zx120_lidar.yaw").as_double();
        
        // LiDAR specifications
        opt_params_.fov_horizontal = this->get_parameter("lidar.fov_horizontal").as_double() * M_PI / 180.0;
        opt_params_.fov_vertical = this->get_parameter("lidar.fov_vertical").as_double() * M_PI / 180.0;
        opt_params_.ray_step_size = this->get_parameter("lidar.ray_step_size").as_double();
        opt_params_.visibility_search_radius = this->get_parameter("lidar.visibility_search_radius").as_double();
        
        RCLCPP_INFO(this->get_logger(), "Updated parameters: α=%.2f, β=%.2f", opt_params_.alpha, opt_params_.beta);
    }
    
    void excavationAreaCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        excavation_area_.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::fromROSMsg(*msg, *excavation_area_);
        
        if (excavation_area_->empty()) return;
        
        excavation_kdtree_.reset(new pcl::KdTreeFLANN<pcl::PointXYZRGB>);
        try {
            excavation_kdtree_->setInputCloud(excavation_area_);
            // RCLCPP_INFO(this->get_logger(), "Built excavation KD-tree with %zu points", excavation_area_->size());
        } catch (const std::exception& e) {
            // RCLCPP_ERROR(this->get_logger(), "Failed to build excavation KD-tree: %s", e.what());
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
            
            // Ensure normals point upward
            for (auto& normal : terrain_normals_->points) {
                if (normal.normal_z < 0) {
                    normal.normal_x = -normal.normal_x;
                    normal.normal_y = -normal.normal_y;
                    normal.normal_z = -normal.normal_z;
                }
            }
            
            RCLCPP_INFO(this->get_logger(), "Computed normals for %zu points", terrain_normals_->size());
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to compute terrain normals: %s", e.what());
            terrain_normals_.reset(new pcl::PointCloud<pcl::Normal>);
        }
    }
    
    void generateExcavationGrid() {
        if (!excavation_area_ || excavation_area_->empty() || !excavation_kdtree_) return;
        
        // Calculate excavation area bounds
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
        
        // Add margin
        double margin = opt_params_.grid_resolution;
        grid_min_x_ -= margin; grid_max_x_ += margin;
        grid_min_y_ -= margin; grid_max_y_ += margin;
        
        grid_width_ = static_cast<int>(std::ceil((grid_max_x_ - grid_min_x_) / opt_params_.grid_resolution)) + 1;
        grid_height_ = static_cast<int>(std::ceil((grid_max_y_ - grid_min_y_) / opt_params_.grid_resolution)) + 1;
        
        RCLCPP_INFO(this->get_logger(), 
                   "Grid bounds: X[%.2f, %.2f], Y[%.2f, %.2f], Z[%.2f, %.2f], Size: %dx%d", 
                   grid_min_x_, grid_max_x_, grid_min_y_, grid_max_y_, 
                   excavation_min_z_, excavation_max_z_, grid_width_, grid_height_);
        
        excavation_grid_.clear();
        excavation_grid_.resize(grid_height_, std::vector<GridCell>(grid_width_));
        
        int total_valid_cells = 0;
        
        for (int i = 0; i < grid_height_; ++i) {
            for (int j = 0; j < grid_width_; ++j) {
                double x = grid_min_x_ + j * opt_params_.grid_resolution;
                double y = grid_min_y_ + i * opt_params_.grid_resolution;
                
                double estimated_z;
                bool z_valid = estimateZCoordinate(x, y, estimated_z);
                
                if (!z_valid) {
                    excavation_grid_[i][j].is_valid = false;
                    continue;
                }
                
                excavation_grid_[i][j] = GridCell(x, y, estimated_z);
                excavation_grid_[i][j].is_valid = true;
                computeCellSurfaceNormal(excavation_grid_[i][j]);
                
                total_valid_cells++;
            }
        }
        
        // RCLCPP_INFO(this->get_logger(), "Generated grid with %d valid cells", total_valid_cells);
        publishGridVisualization();
    }
    
    bool estimateZCoordinate(double x, double y, double& estimated_z) {
        pcl::PointXYZRGB search_point;
        search_point.x = x; 
        search_point.y = y; 
        search_point.z = (excavation_min_z_ + excavation_max_z_) / 2.0;
        
        std::vector<int> point_indices;
        std::vector<float> point_distances;
        
        double search_radius = opt_params_.grid_resolution * 2.0;
        if (excavation_kdtree_->radiusSearch(search_point, search_radius, 
                                           point_indices, point_distances) > 0) {
            
            // Weighted average based on 2D distance
            double sum_z = 0.0, sum_weight = 0.0;
            
            for (size_t i = 0; i < point_indices.size(); ++i) {
                const auto& point = excavation_area_->points[point_indices[i]];
                double distance_2d = sqrt(pow(point.x - x, 2) + pow(point.y - y, 2));
                double weight = (distance_2d < 1e-6) ? 100.0 : 1.0 / (distance_2d + 0.01);
                
                sum_z += point.z * weight;
                sum_weight += weight;
            }
            
            if (sum_weight > 0) {
                estimated_z = sum_z / sum_weight;
                return true;
            }
        }
        
        return false;
    }
    
    void computeCellSurfaceNormal(GridCell& cell) {
        if (!terrain_normals_ || terrain_normals_->empty()) {
            cell.surface_normal.normal_x = 0.0;
            cell.surface_normal.normal_y = 0.0;
            cell.surface_normal.normal_z = 1.0;
            return;
        }
        
        pcl::PointXYZRGB search_point;
        search_point.x = cell.x; 
        search_point.y = cell.y; 
        search_point.z = cell.z;
        
        std::vector<int> point_indices;
        std::vector<float> point_distances;
        
        if (excavation_kdtree_->radiusSearch(search_point, opt_params_.surface_normal_search_radius, 
                                           point_indices, point_distances) > 0) {
            
            double sum_nx = 0.0, sum_ny = 0.0, sum_nz = 0.0;
            int valid_normals = 0;
            
            for (size_t i = 0; i < point_indices.size() && i < 10; ++i) {
                int idx = point_indices[i];
                if (idx >= 0 && idx < static_cast<int>(terrain_normals_->size())) {
                    const auto& normal = terrain_normals_->points[idx];
                    
                    if (std::isfinite(normal.normal_x) && std::isfinite(normal.normal_y) && 
                        std::isfinite(normal.normal_z)) {
                        sum_nx += normal.normal_x;
                        sum_ny += normal.normal_y;
                        sum_nz += normal.normal_z;
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
                }
            }
        }
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
        
        for (auto& candidate : candidates) {
            SimplifiedEvaluation evaluation = evaluatePosition(candidate);
            candidate.total_score = evaluation.total_score;
            
            if (candidate.total_score > best_score) {
                best_score = candidate.total_score;
                best_candidate = candidate;
            }
        }
        
        best_mobile_position_ = best_candidate;
        candidate_positions_ = candidates;
        
        RCLCPP_INFO(this->get_logger(), 
                   "ーーーBest position: (%.2f, %.2f, %.2f) with score: %.2fーーー", 
                   best_mobile_position_.x, best_mobile_position_.y, 
                   best_mobile_position_.z, best_score);
        
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
        
        int grid_size = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(num_candidates_))));
        double x_step = (expanded_max_x - expanded_min_x) / (grid_size - 1);//候補位置を格子状に配置する際の間隔
        double y_step = (expanded_max_y - expanded_min_y) / (grid_size - 1);
        
        for (int i = 0; i < grid_size; ++i) {
            for (int j = 0; j < grid_size; ++j) {
                double x = expanded_min_x + i * x_step;
                double y = expanded_min_y + j * y_step;
                
                // Skip positions too close to ZX120
                double dist_to_zx120 = sqrt(pow(x - zx120_lidar_position_.x, 2) + 
                                          pow(y - zx120_lidar_position_.y, 2));
                if (dist_to_zx120 < 2.0) continue;
                
                // Skip positions inside excavation area
                if (x >= grid_min_x_ && x <= grid_max_x_ && y >= grid_min_y_ && y <= grid_max_y_) continue;
                
                double ground_z = getGroundHeight(x, y);
                double z = ground_z + sensor_height_;
                
                // Calculate viewing angle to excavation center
                double dx = center_x - x;
                double dy = center_y - y;
                double dz = center_z - z;
                double horizontal_distance = sqrt(dx*dx + dy*dy);//LiDARから掘削エリア中心までの水平距離
                
                if (horizontal_distance < 0.1) continue;
                
                double elevation_angle = atan2(-dz, horizontal_distance);//LiDARから見て掘削エリアがどの角度にあるか
                
                if (elevation_angle >= min_elevation_angle_ && elevation_angle <= max_elevation_angle_) {
                    double pitch = -M_PI/2 + elevation_angle;
                    double yaw = atan2(dy, dx);
                    candidates.emplace_back(x, y, z, pitch, yaw);
                }
            }
        }
        
        // RCLCPP_INFO(this->get_logger(), "Generated %zu candidate positions", candidates.size());
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
    

    // Mobile LiDARの候補位置がどれだけ優秀な監視位置かを総合評価する
    SimplifiedEvaluation evaluatePosition(const LidarPosition& mobile_pos) {
        SimplifiedEvaluation evaluation = {};
        
        double total_score = 0.0;
        int covered_cells = 0;
        int total_valid_cells = 0;
        
        for (int i = 0; i < grid_height_; ++i) {
            for (int j = 0; j < grid_width_; ++j) {
                GridCell& cell = excavation_grid_[i][j];
                
                if (!cell.is_valid) continue;
                total_valid_cells++;
                
                // Evaluate both LiDARs for this cell
                cell.score_zx120 = evaluateCellScore(zx120_lidar_position_, cell);
                cell.score_mobile = evaluateCellScore(mobile_pos, cell);
                
                // Combined score: take the maximum (best coverage from either LiDAR)
                cell.combined_score = std::max(cell.score_zx120, cell.score_mobile);

                // std::cout << "score_zx120" << cell.score_zx120 << "score_mobile" << cell.score_mobile << std::endl;
                
                if (cell.combined_score > 0) {
                    covered_cells++;
                    total_score += cell.combined_score;
                }
            }
        }
        
        evaluation.total_score = total_score;
        evaluation.covered_cells = covered_cells;
        evaluation.total_cells = total_valid_cells;
        evaluation.coverage_ratio = total_valid_cells > 0 ? static_cast<double>(covered_cells) / total_valid_cells : 0.0;
        evaluation.average_score = covered_cells > 0 ? total_score / covered_cells : 0.0;
        
        return evaluation;
    }
    
    /**
     * Simplified evaluation function: α*θ + β*(1/L)
     * where:
     * - θ is the angle between laser beam and ground surface normal (in radians)
     * - L is the distance from LiDAR to ground point (in meters)
     * - α and β are weighting parameters
     */
    double evaluateCellScore(const LidarPosition& lidar_pos, const GridCell& cell) {
        // Calculate distance L
        double dx = cell.x - lidar_pos.x;
        double dy = cell.y - lidar_pos.y;
        double dz = cell.z - lidar_pos.z;
        double L = sqrt(dx*dx + dy*dy + dz*dz);
        
        // // Check distance constraints
        // if (L < opt_params_.min_distance || L > opt_params_.max_distance) {
        //     return 0.0;
        // }
        
        // // Check if point is within LiDAR's field of view
        // if (!isInFieldOfView(lidar_pos, cell, dx, dy, dz, L)) {
        //     return 0.0;
        // }
        
        // // Check visibility (no obstacles blocking the view)
        // if (!checkVisibility(lidar_pos, cell)) {
        //     return 0.0;
        // }

        std::cout << "evaluateCellScore start" << std::endl;

        
        // Calculate angle θ between laser beam and surface normal
        // Laser beam direction (normalized)
        double beam_x = dx / L;
        double beam_y = dy / L;
        double beam_z = dz / L;
        
        // Dot product between beam direction and surface normal
        double dot_product = beam_x * cell.surface_normal.normal_x + 
                           beam_y * cell.surface_normal.normal_y + 
                           beam_z * cell.surface_normal.normal_z;
        
        // θ is the angle between beam and normal (0 to π/2)
        double theta = acos(std::max(0.0, std::min(1.0, std::abs(dot_product))));
        
        // Apply simplified evaluation function: α*θ + β*(1/L)
        double score = opt_params_.alpha * theta + opt_params_.beta * (1.0 / L);
        std::cout << "theta:" << theta << "L:" << L << std::endl;
        
        return std::max(0.0, score);
    }
    
    bool isInFieldOfView(const LidarPosition& lidar_pos, const GridCell& cell, 
                        double dx, double dy, double dz, double distance) {
        // Calculate azimuth and elevation angles
        double azimuth = atan2(dy, dx);
        double elevation = atan2(dz, sqrt(dx*dx + dy*dy));
        
        // Calculate relative angles with respect to LiDAR orientation
        double azimuth_diff = fmod(azimuth - lidar_pos.yaw + M_PI, 2*M_PI) - M_PI;
        double elevation_diff = elevation - lidar_pos.pitch;
        
        // Check if within field of view
        return (std::abs(azimuth_diff) <= opt_params_.fov_horizontal / 2.0) &&
               (std::abs(elevation_diff) <= opt_params_.fov_vertical / 2.0);
    }
    
    bool checkVisibility(const LidarPosition& lidar_pos, const GridCell& cell) {
        if (!terrain_kdtree_) return true;
        
        double dx = cell.x - lidar_pos.x;
        double dy = cell.y - lidar_pos.y;
        double dz = cell.z - lidar_pos.z;
        double distance = sqrt(dx*dx + dy*dy + dz*dz);
        
        double norm_dx = dx / distance;
        double norm_dy = dy / distance;
        double norm_dz = dz / distance;
        
        // Check for obstacles along the ray
        double step_distance = opt_params_.ray_step_size;
        while (step_distance < distance - opt_params_.ray_step_size) {
            double check_x = lidar_pos.x + norm_dx * step_distance;
            double check_y = lidar_pos.y + norm_dy * step_distance;
            double check_z = lidar_pos.z + norm_dz * step_distance;
            
            pcl::PointXYZRGB search_point;
            search_point.x = check_x;
            search_point.y = check_y;
            search_point.z = check_z;
            
            std::vector<int> point_indices;
            std::vector<float> point_distances;
            
            if (terrain_kdtree_->radiusSearch(search_point, opt_params_.visibility_search_radius, 
                                            point_indices, point_distances) > 0) {
                return false; // Obstacle found
            }
            
            step_distance += opt_params_.ray_step_size;
        }
        
        return true; // No obstacles
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
        
        // Clear existing markers
        visualization_msgs::msg::Marker clear_marker;
        clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        marker_array.markers.push_back(clear_marker);
        
        // ZX120 LiDAR position marker
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
        
        // Candidate positions
        double min_score = std::numeric_limits<double>::max();
        double max_score = std::numeric_limits<double>::lowest();
        
        // Find score range for normalization
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
            
            // Color based on score (red = low, green = high)
            double normalized_score = 0.0;
            if (max_score > min_score) {
                normalized_score = (candidate_positions_[i].total_score - min_score) / (max_score - min_score);
            }
            
            marker.color.r = 1.0 - normalized_score;
            marker.color.g = normalized_score;
            marker.color.b = 0.0;
            marker.color.a = 0.7;
            
            marker.lifetime = rclcpp::Duration::from_seconds(8.0);
            marker_array.markers.push_back(marker);
        }
        
        // Best position marker
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
        
        // Clear existing markers
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
                
                // Color based on combined score
                double score_intensity = std::min(1.0, std::max(0.0, cell.combined_score / 10.0));
                marker.color.r = 1.0 - score_intensity;
                marker.color.g = score_intensity;
                marker.color.b = 0.2;
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
    rclcpp::spin(std::make_shared<SimplifiedDualLidarOptimizer>());
    rclcpp::shutdown();
    return 0;
}