#include "lessons.hpp"
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Geometry>
#include <cmath>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <optional>

namespace training {
#ifndef M_PI
constexpr double M_PI = 3.14159265358979323846;
#endif

// 装甲板真实尺寸（米）。与检测端的大小板判定保持一致。
constexpr double kSmallArmorW = 0.135;
constexpr double kLargeArmorW = 0.230;
constexpr double kArmorH = 0.055;

// 第四讲：位姿、坐标与基础跟踪
// 流程：取最显著装甲板 -> solvePnP 求相机系位姿 -> 相机系转云台系 ->
//       转 IMU 稳定系（相对云台原点）并做帧间速度估计。
std::optional<Target> lesson04_tracking(const std::vector<Armor>& armors, TimePoint timestamp,
    const Eigen::Quaterniond& imu_q, const CameraCalibration& calibration, TrackingState& state) {
  if (armors.empty() || !calibration.valid()) return std::nullopt;

  // 选取面积最大的装甲板作为本帧目标（最近/最显著）。
  const Armor* best = nullptr;
  double best_area = -1.0;
  for (const auto& a : armors) {
    const double w = cv::norm(a.corners[1] - a.corners[0]);
    const double h = cv::norm(a.corners[2] - a.corners[1]);
    const double area = w * h;
    if (area > best_area) {
      best_area = area;
      best = &a;
    }
  }
  if (!best) return std::nullopt;

  const double ah = kArmorH;
  const std::vector<cv::Point2f> image = {
      best->corners[0], best->corners[1], best->corners[2], best->corners[3]};

  // 板型判定：用「小板」「大板」两组尺寸各解一次 PnP，取重投影误差更小的那组。
  // 不能只用包围框宽高比：斜视时框会被压窄，大板可能被判成小板，而板宽直接
  // 决定 solvePnP 的深度，判错就是 230/135 ≈ 1.7 倍的距离误差。
  // 两种板高度相同(55mm)、宽度不同，因此重投影误差能区分它们（不是纯缩放）。
  double aw = kSmallArmorW;
  bool large = false;
  cv::Mat rvec, tvec;
  {
    double best_err = 1e18;
    double err_small = -1.0, err_large = -1.0;
    for (const auto& cand : {std::pair<double, bool>{kSmallArmorW, false},
                             std::pair<double, bool>{kLargeArmorW, true}}) {
      const double w = cand.first;
      const std::vector<cv::Point3f> object = {
          cv::Point3f(-w / 2, ah / 2, 0), cv::Point3f(w / 2, ah / 2, 0),
          cv::Point3f(w / 2, -ah / 2, 0), cv::Point3f(-w / 2, -ah / 2, 0)};
      cv::Mat rv, tv;
      if (!cv::solvePnP(object, image, calibration.matrix, calibration.distortion, rv, tv,
                        false, cv::SOLVEPNP_IPPE))
        continue;
      std::vector<cv::Point2f> projected;
      cv::projectPoints(object, rv, tv, calibration.matrix, calibration.distortion, projected);
      double err = 0.0;
      for (std::size_t i = 0; i < image.size(); ++i) err += cv::norm(projected[i] - image[i]);
      err /= static_cast<double>(image.size());
      if (cand.second)
        err_large = err;
      else
        err_small = err;
      if (err < best_err) {
        best_err = err;
        aw = w;
        large = cand.second;
        rvec = rv;
        tvec = tv;
      }
    }
    if (best_err > 1e17) return std::nullopt;  // 两组都失败
    if (std::getenv("PRINT_DIST")) {
      static int dbg = 0;
      if (++dbg % 20 == 0)
        std::fprintf(stderr, "[pnp] plate=%s  err_S=%.2f  err_L=%.2f px\n",
                     large ? "L(230mm)" : "S(135mm)", err_small, err_large);
    }
  }
  (void)aw;
  (void)large;

  cv::Mat tcam;
  tvec.convertTo(tcam, CV_64F);
  const Eigen::Vector3d p_camera(tcam.at<double>(0), tcam.at<double>(1), tcam.at<double>(2));

  // 相机系 -> 云台系（固定外参）。
  const Eigen::Vector3d p_gimbal =
      calibration.camera_to_gimbal * p_camera + calibration.camera_to_gimbal_translation;

  // 帧间速度估计与指数平滑。
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  if (state.last_valid) {
    const double dt = std::chrono::duration<double>(timestamp - state.last_time).count();
    if (dt > 1e-3) {
      velocity = (p_gimbal - state.last_position) / dt;
      state.velocity = 0.6 * state.velocity + 0.4 * velocity;
    }
  }
  state.last_position = p_gimbal;
  state.last_time = timestamp;
  state.last_valid = true;

  // 转 IMU 稳定系（相对云台原点）。训练场景下底盘静止，imu_q 近似单位阵，
  // 该变换与下游命令映射自洽；远动目标下仍提供一致的世界系向量。
  Target target;
  target.position = imu_q * p_gimbal;
  target.velocity = imu_q * state.velocity;
  return target;
}
}  // namespace training
