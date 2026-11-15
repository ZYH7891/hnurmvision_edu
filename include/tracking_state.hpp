#pragma once
#include "types.hpp"
#include <Eigen/Geometry>
#include <deque>
#include <opencv2/core.hpp>

namespace training {
// 一次 250ms 预测的记录：在 t_pred 时刻预测目标在 t_pred+250ms 会到达 position。
struct PredictionRecord {
  TimePoint t_pred{};
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
};

// 第四讲使用的跨帧跟踪状态。
// 记录上一帧在「云台固定系」下的目标位置与速度，用于帧间速度估计与平滑；
// 并维护预测历史用于「预测值 vs 未来真实观测」的误差评估。
struct TrackingState {
  bool last_valid = false;                                  // 上一帧是否有有效观测
  Eigen::Vector3d last_position = Eigen::Vector3d::Zero();   // 上一帧目标位置（云台系，单位 m）
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();        // 平滑后的目标速度（云台系，单位 m/s）
  TimePoint last_time{};                                    // 上一帧时间戳

  // —— 选板时间一致性：优先延续上一帧锁定的板，避免两板面积接近时左右横跳 ——
  bool last_center_valid = false;
  cv::Point2f last_center{};                                 // 上一帧锁定板的图像中心

  // —— 相机系下的速度估计（用于把预测位置反投影回图像画预测框）——
  bool last_cam_valid = false;
  Eigen::Vector3d last_camera_position = Eigen::Vector3d::Zero();
  Eigen::Vector3d camera_velocity = Eigen::Vector3d::Zero();

  // —— 250ms 预测误差评估：预测值 vs 未来观测（真实值）——
  std::deque<PredictionRecord> predictions;
  double last_prediction_error = 0.0;   // 最近一次评估误差（m），-1 表示尚未有可用评估
  bool has_prediction_error = false;

  // —— 静止靶 PnP 距离参考真值（前 N 帧观测均值收敛到真值）——
  double dist_sum = 0.0;
  int dist_count = 0;
  double distance_reference = 0.0;      // 0 = 尚未标定
  bool calibrated = false;
};
}  // namespace training
