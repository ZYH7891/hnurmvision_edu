#pragma once
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <array>
#include <chrono>

namespace training {
using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
constexpr double kPi = 3.14159265358979323846;

// IO 提供的算法输入；错误、连接状态和设备模式留在 IO 层。
struct CameraCalibration {
  cv::Mat matrix; // 3x3 CV_64F，相机光学坐标系
  cv::Mat distortion;
  Eigen::Matrix3d camera_to_gimbal = Eigen::Matrix3d::Identity();
  Eigen::Vector3d camera_to_gimbal_translation = Eigen::Vector3d::Zero();
  unsigned width=0, height=0;
  bool valid() const { return !matrix.empty(); }
};

struct SensorData {
  cv::Mat image;  // CV_8UC3，BGR，独立拥有像素内存
  TimePoint timestamp{};
  Eigen::Quaterniond imu_q = Eigen::Quaterniond::Identity();
  double bullet_speed = 0;  // m/s
};

// 第三讲输出：一块装甲板的图像观测。
struct Armor {
  std::array<cv::Point2f, 4> corners{};  // 左上、右上、右下、左下
  bool large_armor = false;
  int class_id = -1;
  double confidence = 0;
};

// 第四讲输出：当前观测时刻的单目标状态。无目标用 std::nullopt 表示。
struct Target {
  Eigen::Vector3d position = Eigen::Vector3d::Zero();  // m，IMU稳定系中相对云台原点的目标向量
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();  // m/s，同一坐标系

  // —— 250ms 匀速预测（任务 2）——
  bool has_prediction = false;                          // 是否已产生有效预测框
  std::array<cv::Point2f, 4> predicted_corners{};       // 预测框在图像上的四角（另一颜色绘制）
  double prediction_error = 0.0;                        // 最近一次「预测 vs 250ms后真实观测」误差（m）
  bool has_prediction_error = false;

  // —— 静止靶 PnP 位姿 vs 参考真值（任务 1）——
  double distance = 0.0;                                // 当前 PnP 距离（m）
  double distance_reference = 0.0;                      // 静止参考真值距离（m），0=未标定
  bool calibrated = false;
};

// 第五讲输出：绝对角使用 rad，IO 边界统一转换单位和方向。
struct Command {
  bool control = false;
  bool shoot = false;
  double yaw = 0;
  double pitch = 0;
};
}  // namespace training
