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
    pcl::Normal surface_normal;
    
    GridCell(double x=0, double y=0, double z=0) 
        : x(x), y(y), z(z), is_valid(true), 
          score_zx120(0.0), score_mobile(0.0), combined_score(0.0) {
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
        
        // 簡素化されたパラメータの宣言
        this->declare_parameter("grid_resolution", 0.1);
        this->declare_parameter("sensor_height", 1.1);
        this->declare_parameter("search_radius", 3.0);
        this->declare_parameter("max_distance", 15.0);
        this->declare_parameter("num_candidates", 196);
        
        // パラメータ読み込み
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
            
        rng_.seed(std::chrono::system_clock::now().time_since_epoch().count());//いらないかも
    }
    
private:
    // 簡素化されたパラメータ（固定値）
    static constexpr double ALPHA = 1.0;          // 角度重み
    static constexpr double BETA = 10.0;          // 距離重み
    static constexpr double MIN_DISTANCE = 1.0;   // 最小距離
    static constexpr double ZX120_OFFSET_X = 0.4; // ZX120オフセット
    static constexpr double ZX120_OFFSET_Y = 0.5;
    static constexpr double ZX120_OFFSET_Z = 3.5;
    static constexpr double ZX120_PITCH = -M_PI/6;
    static constexpr double ZX120_YAW = 0.0;
    static constexpr double FOV_HORIZONTAL = 120.0 * M_PI / 180.0;
    static constexpr double FOV_VERTICAL = 90.0 * M_PI / 180.0;
    static constexpr double NORMAL_SEARCH_RADIUS = 1.5;
    static constexpr double RAY_STEP_SIZE = 0.1;
    static constexpr double VISIBILITY_RADIUS = 0.5;
    // static constexpr double MIN_ELEVATION = -60.0 * M_PI / 180.0;
    // static constexpr double MAX_ELEVATION = 45.0 * M_PI / 180.0;
    static constexpr double MIN_ELEVATION = -80.0 * M_PI / 180.0;
    static constexpr double MAX_ELEVATION = 85.0 * M_PI / 180.0;
    
    // 設定可能パラメータ（最小限）
    double grid_resolution_;
    double sensor_height_;
    double search_radius_;
    double max_distance_;
    int num_candidates_;
    
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
    pcl::PointCloud<pcl::Normal>::Ptr terrain_normals_;//法線格納用
    
    std::vector<std::vector<GridCell>> excavation_grid_;
    double grid_min_x_, grid_max_x_, grid_min_y_, grid_max_y_;
    double excavation_min_z_, excavation_max_z_;
    int grid_width_, grid_height_;
    
    std::mt19937 rng_;//いらないかも
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
        
        // RCLCPP_INFO(this->get_logger(), "Parameters: grid_res=%.2f, height=%.1f, search_r=%.1f", 
        //            grid_resolution_, sensor_height_, search_radius_);
    }
    
    void excavationAreaCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        excavation_area_.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::fromROSMsg(*msg, *excavation_area_);//ROS to PCL
        
        if (excavation_area_->empty()) return;
        
        excavation_kdtree_.reset(new pcl::KdTreeFLANN<pcl::PointXYZRGB>);
        try {
            excavation_kdtree_->setInputCloud(excavation_area_);
            computeTerrainNormals();//法線計算
            generateExcavationGrid();//各セルの3D位置と法線の計算
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to process excavation area: %s", e.what());
        }
    }
    
    void terrainCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        terrain_cloud_.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::fromROSMsg(*msg, *terrain_cloud_);//ROS to PCL
        
        if (!terrain_cloud_->empty()) {
            terrain_kdtree_.reset(new pcl::KdTreeFLANN<pcl::PointXYZRGB>);
            try {
                terrain_kdtree_->setInputCloud(terrain_cloud_);//PCLのメソッド
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "Failed to build terrain KD-tree: %s", e.what());
            }
        }
    }
    
    void computeTerrainNormals() {//各地点の表面の向きを計算
        if (!excavation_area_ || excavation_area_->empty()) return;
        
        try {
            pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGB>());
            kdtree->setInputCloud(excavation_area_);
            
            //法線推定器の設定
            normal_estimator_.setInputCloud(excavation_area_);//法線計算のための点群をセット
            normal_estimator_.setSearchMethod(kdtree);//探索方法をセット
            normal_estimator_.setRadiusSearch(NORMAL_SEARCH_RADIUS);//半径探索の範囲をセット
            
            //法線計算の実行
            terrain_normals_.reset(new pcl::PointCloud<pcl::Normal>);//法線を格納する点群を初期化
            normal_estimator_.compute(*terrain_normals_);//法線計算を実行
            
            // 法線を上向きに統一
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
    
    void generateExcavationGrid() {
        if (!excavation_area_ || excavation_area_->empty()) return;
        
        grid_min_x_ = grid_min_y_ = excavation_min_z_ = std::numeric_limits<double>::max();
        grid_max_x_ = grid_max_y_ = excavation_max_z_ = std::numeric_limits<double>::lowest();
        
        for (const auto& point : excavation_area_->points) {//掘削エリアの境界計算
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
        
        grid_width_ = static_cast<int>(std::ceil((grid_max_x_ - grid_min_x_) / grid_resolution_)) + 1;
        grid_height_ = static_cast<int>(std::ceil((grid_max_y_ - grid_min_y_) / grid_resolution_)) + 1;
        
        excavation_grid_.clear();
        excavation_grid_.resize(grid_height_, std::vector<GridCell>(grid_width_));
        
        int valid_cells = 0;
        for (int i = 0; i < grid_height_; ++i) {
            for (int j = 0; j < grid_width_; ++j) {
                double x = grid_min_x_ + j * grid_resolution_;
                double y = grid_min_y_ + i * grid_resolution_;
                
                double estimated_z;
                if (estimateZCoordinate(x, y, estimated_z)) {//Z座標の推定に成功した場合, estimated_zにZ座標が格納される
                    excavation_grid_[i][j] = GridCell(x, y, estimated_z);//std::vector<std::vector<GridCell>> excavation_grid_;
                    computeCellSurfaceNormal(excavation_grid_[i][j]);//cell.surface_normalに法線を格納(GridCell構造体のメンバ)
                    valid_cells++;//有効なセル数をカウント
                } else {
                    excavation_grid_[i][j].is_valid = false;
                }
            }
        }
        
        // RCLCPP_INFO(this->get_logger(), "Generated grid: %dx%d (%d valid cells)", 
        //            grid_width_, grid_height_, valid_cells);
        publishGridVisualization();
    }
    
    bool estimateZCoordinate(double x, double y, double& estimated_z) {
        pcl::PointXYZRGB search_point;
        search_point.x = x; search_point.y = y; search_point.z = excavation_max_z_;
        
        std::vector<int> point_indices;
        std::vector<float> point_distances;
        
        double search_radius = grid_resolution_ * 3.0;
        
        if (excavation_kdtree_->radiusSearch(search_point, search_radius, point_indices, point_distances) > 0) {
            double best_z = excavation_max_z_;
            bool found = false;
            
            for (int idx : point_indices) {
                const auto& point = excavation_area_->points[idx];
                double dist_2d = sqrt(pow(point.x - x, 2) + pow(point.y - y, 2));
                
                if (dist_2d <= grid_resolution_ * 1.5) {
                    if (!found || point.z < best_z) {
                        best_z = point.z;
                        found = true;
                    }
                }
            }
            
            if (found) {
                estimated_z = best_z;
                return true;
            }
        }
        
        return false;
    }
    
    void computeCellSurfaceNormal(GridCell& cell) {//各セルの法線計算
        if (!terrain_normals_ || terrain_normals_->empty()) return;
        
        pcl::PointXYZRGB search_point;//探索したい位置の中心
        search_point.x = cell.x; search_point.y = cell.y; search_point.z = cell.z;
        
        std::vector<int> point_indices;//radiusSearchで取得された近傍点のインデックス配列
        std::vector<float> point_distances;//探索の中心から各近傍店までの距離
        
        if (excavation_kdtree_->radiusSearch(search_point, NORMAL_SEARCH_RADIUS, 
                                           point_indices, point_distances) > 0) {//掘削エリアの点群から法線を取得
            
            double sum_nx = 0.0, sum_ny = 0.0, sum_nz = 0.0;
            int valid_count = 0;
            
            for (int idx : point_indices) {
                if (idx < static_cast<int>(terrain_normals_->size())) {//terrain_normalsは
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
    
    void runOptimization() {
        if (excavation_grid_.empty() || !terrain_cloud_ || !getZX120Position()) return;
        
        updateParameters();
        
        auto candidates = generateCandidatePositions();//候補位置の生成
        
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
        
        RCLCPP_INFO(this->get_logger(), 
                   "Best position: (%.2f, %.2f, %.2f) score: %.2f", 
                   best_mobile_position_.x, best_mobile_position_.y, 
                   best_mobile_position_.z, best_score);
        
        publishOptimalPosition();
        publishCandidatePositions();
        publishGridVisualization();
    }
    
    std::vector<LidarPosition> generateCandidatePositions() {
        std::vector<LidarPosition> candidates;//LidaPosition：x,y,z,pitch,yaw,total_score
        
        double expanded_min_x = grid_min_x_ - search_radius_;//掘削エリアをsearch_radius_分だけ拡大
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
                
                // ZX120に近すぎる位置を除外
                double dist_to_zx120 = sqrt(pow(x - zx120_lidar_position_.x, 2) + 
                                          pow(y - zx120_lidar_position_.y, 2));
                if (dist_to_zx120 < 0.5) continue;
                                
                // より簡潔版：掘削エリアを80%に縮小
                // double margin = 0.2; // 20%のマージン
                // double shrunk_min_x = grid_min_x_ + (grid_max_x_ - grid_min_x_) * margin;
                // double shrunk_max_x = grid_max_x_ - (grid_max_x_ - grid_min_x_) * margin;
                // double shrunk_min_y = grid_min_y_ + (grid_max_y_ - grid_min_y_) * margin;
                // double shrunk_max_y = grid_max_y_ - (grid_max_y_ - grid_min_y_) * margin;

                // if (x >= shrunk_min_x && x <= shrunk_max_x && y >= shrunk_min_y && y <= shrunk_max_y) continue;


                // 掘削エリア内を除外,コメントアウトしてもいいかも
                if (x >= grid_min_x_ && x <= grid_max_x_ && y >= grid_min_y_ && y <= grid_max_y_) continue;
                
                double ground_z = getGroundHeight(x, y);
                double z = ground_z + sensor_height_;
                
                double dx = center_x - x;
                double dy = center_y - y;
                double dz = center_z - z;
                double horizontal_distance = sqrt(dx*dx + dy*dy);//掘削エリアの中心点とモバイルLiDAR候補位置との間のXY平面（水平面）での距離
                
                if (horizontal_distance < 0.1) continue;
                
                double elevation_angle = atan2(-dz, horizontal_distance);
                
                if (elevation_angle >= MIN_ELEVATION && elevation_angle <= MAX_ELEVATION) {//掘削エリアの中心点がモバイルLiDAR候補位置から見てどの角度にあるか
                    double pitch = -M_PI/2 + elevation_angle;
                    double yaw = atan2(dy, dx);
                    candidates.emplace_back(x, y, z, pitch, yaw);//座標系：map??
                }
            }
        }
        
        return candidates;
    }
    
    double getGroundHeight(double x, double y) {
        if (!terrain_cloud_ || terrain_cloud_->empty()) return 0.0;
        
        pcl::PointXYZRGB search_point;
        search_point.x = x; search_point.y = y; search_point.z = 0;
        
        std::vector<int> point_indices;
        std::vector<float> point_distances;
        
        if (terrain_kdtree_->radiusSearch(search_point, 2.0, point_indices, point_distances) > 0) {//探索半径2.0m
            double max_z = std::numeric_limits<double>::lowest();
            for (int idx : point_indices) {//近傍点の中で最も高い点を地面とする
                const auto& point = terrain_cloud_->points[idx];
                double dx = point.x - x;
                double dy = point.y - y;
                if (sqrt(dx*dx + dy*dy) < 1.0) {//1m以内の点のみ考慮
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
        int total_valid_cells = 0;
        
        for (int i = 0; i < grid_height_; ++i) {
            for (int j = 0; j < grid_width_; ++j) {
                GridCell& cell = excavation_grid_[i][j];
                
                if (!cell.is_valid) continue;
                total_valid_cells++;
                
                cell.score_zx120 = evaluateCellScore(zx120_lidar_position_, cell);
                cell.score_mobile = evaluateCellScore(mobile_pos, cell);
                cell.combined_score = std::max(cell.score_zx120, cell.score_mobile);
                std::cout << "Cell (" << i << "," << j << ") - ZX120 Score: " 
                          << cell.score_zx120 << ", Mobile Score: " 
                          << cell.score_mobile << ", Combined: " 
                          << cell.combined_score << std::endl;
                
                if (cell.combined_score > 0) {
                    covered_cells++;
                    total_score += cell.combined_score;
                }
            }
        }
        
        evaluation.total_score = total_score;
        evaluation.covered_cells = covered_cells;
        evaluation.total_cells = total_valid_cells;
        evaluation.coverage_ratio = total_valid_cells > 0 ? 
            static_cast<double>(covered_cells) / total_valid_cells : 0.0;
        // std::cout << "total_score: " << total_score << std::endl;
        return evaluation;
    }
    
    double evaluateCellScore(const LidarPosition& lidar_pos, const GridCell& cell) {
        double dx = cell.x - lidar_pos.x;
        double dy = cell.y - lidar_pos.y;
        double dz = cell.z - lidar_pos.z;
        double L = sqrt(dx*dx + dy*dy + dz*dz);
        
        // // 距離制約
        // if (L < MIN_DISTANCE || L > max_distance_) return 0.0;
        
        // // FOV確認
        // if (!isInFieldOfView(lidar_pos, cell, dx, dy, dz, L)) return 0.0;
        
        // // 視認性確認
        // if (!checkVisibility(lidar_pos, cell)) return 0.0;
        
        // 角度計算
        double beam_x = dx / L;
        double beam_y = dy / L;
        double beam_z = dz / L;
        
        //dot_product = beam_vector · normal_vector = |beam|(単位ベクトル) × |normal|(単位ベクトル) × cos(α)なので、|beam|=1, |normal|=1より、dot_product = cos(α)
        double dot_product = beam_x * cell.surface_normal.normal_x + 
                           beam_y * cell.surface_normal.normal_y + 
                           beam_z * cell.surface_normal.normal_z;
        
        double theta = acos(std::max(0.0, std::min(1.0, std::abs(dot_product))));//逆余弦関数で角度を求める, 0~π/2の範囲に制限,垂直のときは0度,水平のときは90度
        
        // 評価関数: α*θ + β*(1/L)
        // double score = ALPHA * theta + BETA * (1.0 / L);
        // 修正版：小さい角度（垂直入射）を高く評価
        double score = ALPHA * (M_PI/2 - theta) + BETA * (1.0 / L);
        // std::cout << "Cell at (" << cell.x << "," << cell.y << "," << cell.z 
        //           << ") - Distance: " << L << ", Angle: " << theta 
        //           << ", Score: " << score << std::endl;
        
        return std::max(0.0, score);
    }
    
    bool isInFieldOfView(const LidarPosition& lidar_pos, const GridCell& cell, 
                        double dx, double dy, double dz, double distance) {
        double azimuth = atan2(dy, dx);
        double elevation = atan2(dz, sqrt(dx*dx + dy*dy));
        
        double azimuth_diff = fmod(azimuth - lidar_pos.yaw + M_PI, 2*M_PI) - M_PI;
        double elevation_diff = elevation - lidar_pos.pitch;
        
        return (std::abs(azimuth_diff) <= FOV_HORIZONTAL / 2.0) &&
               (std::abs(elevation_diff) <= FOV_VERTICAL / 2.0);
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
        
        double step_distance = RAY_STEP_SIZE;
        while (step_distance < distance - RAY_STEP_SIZE) {
            double check_x = lidar_pos.x + norm_dx * step_distance;
            double check_y = lidar_pos.y + norm_dy * step_distance;
            double check_z = lidar_pos.z + norm_dz * step_distance;
            
            pcl::PointXYZRGB search_point;
            search_point.x = check_x; search_point.y = check_y; search_point.z = check_z;
            
            std::vector<int> point_indices;
            std::vector<float> point_distances;
            
            if (terrain_kdtree_->radiusSearch(search_point, VISIBILITY_RADIUS, 
                                            point_indices, point_distances) > 0) {
                return false;
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
        
        // 既存マーカークリア
        visualization_msgs::msg::Marker clear_marker;
        clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        marker_array.markers.push_back(clear_marker);
        
        // ZX120位置マーカー
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
        
        // 候補位置のスコア範囲を計算
        double min_score = std::numeric_limits<double>::max();
        double max_score = std::numeric_limits<double>::lowest();
        
        for (const auto& candidate : candidate_positions_) {
            min_score = std::min(min_score, candidate.total_score);
            max_score = std::max(max_score, candidate.total_score);
        }
        
        // 候補位置マーカー
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
            
            // スコアに基づく色付け（赤=低、緑=高）
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
        
        // 最適位置マーカー
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
        
        // 既存マーカークリア
        visualization_msgs::msg::Marker clear_marker;
        clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        marker_array.markers.push_back(clear_marker);
        
        // スコア範囲とZ範囲を計算
        double min_score = std::numeric_limits<double>::max();
        double max_score = std::numeric_limits<double>::lowest();
        double min_z = std::numeric_limits<double>::max();
        double max_z = std::numeric_limits<double>::lowest();
        
        for (int i = 0; i < grid_height_; ++i) {
            for (int j = 0; j < grid_width_; ++j) {
                const GridCell& cell = excavation_grid_[i][j];
                if (!cell.is_valid) continue;
                
                min_score = std::min(min_score, cell.combined_score);
                max_score = std::max(max_score, cell.combined_score);
                min_z = std::min(min_z, cell.z);
                max_z = std::max(max_z, cell.z);
            }
        }
        
        double z_range = max_z - min_z;
        double marker_height = std::max(0.05, z_range * 0.05);
        
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
                
                marker.scale.x = grid_resolution_;
                marker.scale.y = grid_resolution_;
                marker.scale.z = marker_height;
                
                // 観測可能かどうかで単純な色分け
                if (cell.combined_score > 0) {
                    // 観測可能：緑色
                    marker.color.r = 0.0;
                    marker.color.g = 1.0;
                    marker.color.b = 0.0;
                    marker.color.a = 0.7;
                } else {
                    // 観測不可能：赤色
                    marker.color.r = 1.0;
                    marker.color.g = 0.0;
                    marker.color.b = 0.0;
                    marker.color.a = 0.5;
                }
                
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