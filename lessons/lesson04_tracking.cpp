#include "lessons.hpp"
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Geometry>
#include <chrono>
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

// 第四讲预测提前量：n = 250 ms（任务要求，可改）。
constexpr double kPredictHorizon = 0.250;

// 静止靶真值标定：前 N 帧观测距离均值收敛到真值（静止靶真值恒定）。
constexpr int kCalibFrames = 30;

// 相机系 3D 点直接投影到像素（点已在相机系，rvec=tvec=0）。
static cv::Point2f project_cam(const Eigen::Vector3d& p_cam, const CameraCalibration& cal) {
  std::vector<cv::Point3f> pts = {cv::Point3f(p_cam.x(), p_cam.y(), p_cam.z())};
  cv::Mat r0 = cv::Mat::zeros(3, 1, CV_64F);
  cv::Mat t0 = cv::Mat::zeros(3, 1, CV_64F);
  std::vector<cv::Point2f> out;
  cv::projectPoints(pts, r0, t0, cal.matrix, cal.distortion, out);
  return out[0];
}

// 第四讲：位姿、坐标与基础跟踪
// 流程：取最显著装甲板 -> solvePnP 求相机系位姿 -> 相机系转云台系 ->
//       转 IMU 稳定系（相对云台原点）并做帧间速度估计 -> 匀速模型预测 250ms。
std::optional<Target> lesson04_tracking(const std::vector<Armor>& armors, TimePoint timestamp,
    const Eigen::Quaterniond& imu_q, const CameraCalibration& calibration, TrackingState& state) {
  if (armors.empty() || !calibration.valid()) return std::nullopt;

  // 选板（时间一致性）：优先延续上一帧锁定的板（最近邻关联），避免两块板
  // 面积接近时目标来回横跳（讲义「状态机」一节指出的问题的最小实现）。
  // 关联不上的帧退回「面积最大」策略并重置关联。
  constexpr float kAssocRadiusPx = 120.0f;
  const Armor* best = nullptr;
  double best_score = -1.0;
  cv::Point2f best_center(0, 0);
  for (const auto& a : armors) {
    cv::Point2f c(0, 0);
    for (const auto& p : a.corners) c += p;
    c *= 0.25f;
    const double w = cv::norm(a.corners[1] - a.corners[0]);
    const double h = cv::norm(a.corners[2] - a.corners[1]);
    double score = w * h;
    if (state.last_center_valid && cv::norm(c - state.last_center) < kAssocRadiusPx) {
      score += 1e6;  // 强偏好延续上一帧目标
    }
    if (score > best_score) {
      best_score = score;
      best = &a;
      best_center = c;
    }
  }
  if (!best) return std::nullopt;
  // 目标更换（关联失败）时，位置突跳不代表目标真实运动：重置速度估计与预测历史，
  // 否则一次板间切换会注入巨大的虚假速度，让预测框持续偏向一边。
  const bool switched =
      state.last_center_valid && cv::norm(best_center - state.last_center) >= kAssocRadiusPx;
  state.last_center = best_center;
  state.last_center_valid = true;
  if (switched) {
    state.velocity = Eigen::Vector3d::Zero();
    state.camera_velocity = Eigen::Vector3d::Zero();
    state.last_valid = false;
    state.last_cam_valid = false;
    state.predictions.clear();
  }

  const double ah = kArmorH;
  const std::vector<cv::Point2f> image = {
      best->corners[0], best->corners[1], best->corners[2], best->corners[3]};

  // 板型判定：用「小板」「大板」两组尺寸各解一次 PnP，取重投影误差更小的那组。
  // 不能只用包围框宽高比：斜视时框会被压窄，大板可能被判成小板，而板宽直接
  // 决定 solvePnP 的深度，判错就是 230/135 ≈ 1.7 倍的距离误差。
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

  const double dt = state.last_valid
                        ? std::chrono::duration<double>(timestamp - state.last_time).count()
                        : 0.0;

  // —— 云台系速度估计与指数平滑（供瞄准用）——
  Eigen::Vector3d velocity = state.velocity;
  if (state.last_valid && dt > 1e-3) {
    const Eigen::Vector3d v_inst = (p_gimbal - state.last_position) / dt;
    velocity = 0.6 * state.velocity + 0.4 * v_inst;
  }
  state.velocity = velocity;
  state.last_position = p_gimbal;
  state.last_time = timestamp;
  state.last_valid = true;

  // —— 相机系速度估计（用于把 250ms 预测位置反投影回图像）——
  Eigen::Vector3d v_cam = state.camera_velocity;
  if (state.last_cam_valid && dt > 1e-3) {
    const Eigen::Vector3d v_inst = (p_camera - state.last_camera_position) / dt;
    v_cam = 0.6 * state.camera_velocity + 0.4 * v_inst;
  }
  state.camera_velocity = v_cam;
  state.last_camera_position = p_camera;
  state.last_cam_valid = true;

  // —— 匀速模型预测：p_future = p_now + v * n ——
  const Eigen::Vector3d p_pred_cam = p_camera + v_cam * kPredictHorizon;
  const Eigen::Vector3d p_pred_gimbal = p_gimbal + velocity * kPredictHorizon;

  // 预测框：把预测中心投影回图像，将当前框平移到该中心。
  Target target;
  target.has_prediction = false;
  if (dt > 1e-3) {
    const cv::Point2f obs_center = project_cam(p_camera, calibration);
    const cv::Point2f pred_center = project_cam(p_pred_cam, calibration);
    const cv::Point2f shift = pred_center - obs_center;
    for (std::size_t i = 0; i < 4; ++i) {
      target.predicted_corners[i] = image[i] + shift;
    }
    target.has_prediction = true;
  }

  // —— 预测误差评估：预测值 vs 250ms 后的真实观测 ——
  state.predictions.push_back({timestamp, p_pred_gimbal});
  // 清理过早的记录（保留最近 ~0.5s）
  while (!state.predictions.empty() &&
         std::chrono::duration<double>(timestamp - state.predictions.front().t_pred).count() >
             2.0 * kPredictHorizon) {
    state.predictions.pop_front();
  }
  // 弹出直到「第二旧的也 ≥ 250ms」，使 front 成为「距今刚好 ≥250ms 的最近预测」。
  while (state.predictions.size() >= 2 &&
         std::chrono::duration<double>(timestamp - state.predictions[1].t_pred).count() >=
             kPredictHorizon) {
    state.predictions.pop_front();
  }
  target.has_prediction_error = false;
  if (!state.predictions.empty() &&
      std::chrono::duration<double>(timestamp - state.predictions.front().t_pred).count() >=
          kPredictHorizon) {
    target.prediction_error = (p_gimbal - state.predictions.front().position).norm();
    target.has_prediction_error = true;
    state.last_prediction_error = target.prediction_error;
    state.has_prediction_error = true;
  }

  // —— 静止靶 PnP 距离参考真值：前 N 帧观测均值 ——
  const double dist = p_gimbal.norm();
  if (!state.calibrated) {
    state.dist_sum += dist;
    if (++state.dist_count >= kCalibFrames) {
      state.distance_reference = state.dist_sum / static_cast<double>(state.dist_count);
      state.calibrated = true;
    }
  }

  // 转 IMU 稳定系（相对云台原点）。训练场景下底盘静止，imu_q 近似单位阵。
  target.position = imu_q * p_gimbal;
  target.velocity = imu_q * velocity;
  target.distance = dist;
  target.distance_reference = state.distance_reference;
  target.calibrated = state.calibrated;
  return target;
}
}  // namespace training
