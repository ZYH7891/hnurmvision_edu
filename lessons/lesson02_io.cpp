#include "io_context.hpp"

#include <Eigen/Geometry>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <utility>

namespace training::io {

namespace {

constexpr double kDegToRad = kPi / 180.0;
constexpr double kRadToDeg = 180.0 / kPi;
constexpr double kPitchLimitRad = 80.0 * kDegToRad;  // 俯仰机械限位（仿真环境放宽）

// sensor_msgs::Image -> cv::Mat(BGR, CV_8UC3, 独立拥有像素内存)。
// 不能只引用 msg->data：回调返回后消息可能被释放。
cv::Mat to_bgr(const sensor_msgs::msg::Image& msg) {
  const int w = static_cast<int>(msg.width);
  const int h = static_cast<int>(msg.height);
  if (w <= 0 || h <= 0 || msg.data.empty()) return {};

  const std::string& enc = msg.encoding;
  auto* data = const_cast<unsigned char*>(msg.data.data());
  if (enc == "rgb8") {
    const cv::Mat rgb(h, w, CV_8UC3, data, msg.step);
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    return bgr;
  }
  if (enc == "bgr8") {
    return cv::Mat(h, w, CV_8UC3, data, msg.step).clone();
  }
  if (enc == "rgba8") {
    const cv::Mat rgba(h, w, CV_8UC4, data, msg.step);
    cv::Mat bgr;
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
    return bgr;
  }
  if (enc == "bgra8") {
    const cv::Mat bgra(h, w, CV_8UC4, data, msg.step);
    cv::Mat bgr;
    cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
    return bgr;
  }
  if (enc == "mono8" || enc == "8UC1") {
    const cv::Mat gray(h, w, CV_8UC1, data, msg.step);
    cv::Mat bgr;
    cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
  }
  return {};
}

// ROS 时间戳 -> 程序的 Clock(steady_clock)。
// 两个时钟没有共同原点：用"此刻 steady - 此刻 ROS"做偏移量，再把它加到采样时刻上。
TimePoint to_steady(rclcpp::Node& node, const builtin_interfaces::msg::Time& stamp) {
  const auto steady_now = Clock::now();
  const auto ros_now_ns = node.get_clock()->now().nanoseconds();
  const auto offset = steady_now - TimePoint(std::chrono::nanoseconds(ros_now_ns));
  return TimePoint(std::chrono::nanoseconds(rclcpp::Time(stamp).nanoseconds())) + offset;
}

// 下位机反馈的三个欧拉角（度）-> 四元数，轴序按消息定义与讲义第三节。
Eigen::Quaterniond to_quaternion(const hnurm_interfaces::msg::VisionRecvData& f) {
  const Eigen::Quaterniond qy(Eigen::AngleAxisd(f.yaw * kDegToRad, Eigen::Vector3d::UnitY()));
  const Eigen::Quaterniond qx(Eigen::AngleAxisd(f.pitch * kDegToRad, Eigen::Vector3d::UnitX()));
  const Eigen::Quaterniond qz(Eigen::AngleAxisd(f.roll * kDegToRad, Eigen::Vector3d::UnitZ()));
  return qy * qx * qz;
}

}  // namespace

// 析构兜底：即使 main 忘了调 close()，也不会因为 std::thread 还 joinable 而 abort。
IoContext::~IoContext() { close(); }

// ---------------------------------------------------------------- Open
// 建立通信：订阅图像与云台反馈，创建指令发布器，并让回调能被调度执行。
void IoContext::open() {
  if (opened_) return;  // 幂等：重复调用只生效一次

  // 1) 若 ROS2 还没初始化就先初始化。shutdown_on_signal=false：
  //    Ctrl+C 的退出流程由 main 自己控制（只有 main 的信号处理器会置位 interrupted）
  if (!rclcpp::ok()) {
    rclcpp::InitOptions opts;
    opts.shutdown_on_signal = false;
    // SignalHandlerOptions::None：连信号处理器都不装，Ctrl+C 完全交给 main 处理
    rclcpp::init(0, nullptr, opts, rclcpp::SignalHandlerOptions::None);
  }

  // 2) 节点：ROS2 里的通信参与者，名字会出现在 ros2 node list 里
  node_ = std::make_shared<rclcpp::Node>("training_io");

  // 3) 两个订阅。QoS 必须与仿真器发送端兼容，否则「Topic 在但收不到数据」
  //    /image_raw            仿真器用默认(reliable)发布 -> 订阅也用默认
  //    /vision_receive_data  仿真器用 sensor_data(best_effort)发布 -> SensorDataQoS
  image_sub_ = node_->create_subscription<sensor_msgs::msg::Image>(
      "/image_raw", rclcpp::QoS(10),
      [this](sensor_msgs::msg::Image::SharedPtr msg) { on_image(std::move(msg)); });
  feedback_sub_ = node_->create_subscription<hnurm_interfaces::msg::VisionRecvData>(
      "/vision_receive_data", rclcpp::SensorDataQoS(),
      [this](hnurm_interfaces::msg::VisionRecvData::SharedPtr msg) { on_feedback(std::move(msg)); });

  // 4) 一个发布器。仿真器用 sensor_data(best_effort)订阅，所以发布也必须用同一类 QoS
  command_pub_ = node_->create_publisher<hnurm_interfaces::msg::VisionSendData>(
      "/vision_send_data", rclcpp::SensorDataQoS());

  // 5) 回调不会自己运行：必须由 Executor 调度。主线程要跑 read/write 循环，
  //    所以 spin 放到后台线程，两个线程通过 mutex_ 交换数据。
  executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
  executor_->add_node(node_);
  spin_thread_ = std::thread([this]() { executor_->spin(); });

  opened_ = true;
  RCLCPP_INFO(node_->get_logger(), "IO opened: /image_raw, /vision_receive_data -> /vision_send_data");
}

// 图像回调：六类编码统一转成 BGR，并记下采集时刻与帧号。
void IoContext::on_image(const sensor_msgs::msg::Image::SharedPtr msg) {
  cv::Mat bgr = to_bgr(*msg);
  if (bgr.empty()) return;
  const TimePoint stamp = to_steady(*node_, msg->header.stamp);

  std::lock_guard<std::mutex> lock(mutex_);
  latest_image_ = std::move(bgr);
  image_time_ = stamp;
  ++image_seq_;
}

// 云台反馈回调：只做转存，云台姿态与弹速都来自这里。
void IoContext::on_feedback(const hnurm_interfaces::msg::VisionRecvData::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(mutex_);
  feedback_ = *msg;
  has_feedback_ = true;
}

// ---------------------------------------------------------------- Read
// 取一帧对齐好的输入。没有新数据返回 nullopt，绝不阻塞主循环。
std::optional<SensorData> IoContext::read() {
  if (!opened_) return std::nullopt;

  SensorData data;
  bool has_new_image = false;
  bool print_status = false;
  double fb_yaw = 0.0, fb_pitch = 0.0, fb_speed = 0.0;
  {
    std::lock_guard<std::mutex> lock(mutex_);

    // 帧号相等 -> 这一帧已经交付过，不再重复算
    has_new_image = image_seq_ != delivered_seq_;
    if (has_new_image) {
      // 直接把缓冲的所有权移交给算法，不再整图 clone（1440×1080 拷贝约 2~4ms，
      // 而且会拉长持锁时间、和回调线程抢锁）。回调下次写入时是 move 赋值，
      // 这块内存从此归 SensorData 所有，不会被覆写。
      data.image = std::move(latest_image_);
      data.timestamp = image_time_;        // 图像采集时刻，不是处理时刻
      delivered_seq_ = image_seq_;
      if (has_feedback_) {
        data.imu_q = to_quaternion(feedback_);
        data.bullet_speed = static_cast<double>(feedback_.bullet_speed.data);
      } else {
        data.imu_q = Eigen::Quaterniond::Identity();  // 还没有反馈，先给单位四元数
        data.bullet_speed = 0.0;
      }
    }

    // 每秒一行云台反馈，录屏演示 Read 时用
    const auto now = Clock::now();
    if (has_feedback_ && now - last_status_time_ >= std::chrono::seconds(1)) {
      last_status_time_ = now;
      print_status = true;
      fb_yaw = feedback_.yaw;
      fb_pitch = feedback_.pitch;
      fb_speed = static_cast<double>(feedback_.bullet_speed.data);
    }
  }

  if (print_status) {
    std::printf("yaw=%.2f deg  pitch=%.2f deg  bullet_speed=%.0f m/s\n", fb_yaw, fb_pitch, fb_speed);
    std::fflush(stdout);
  }

  if (!has_new_image) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));  // 让出 CPU，且不阻塞退出
    return std::nullopt;
  }
  return data;
}

// ---------------------------------------------------------------- Write
// 把 Command 转换成控制消息发出。这里是最后一道安全关口：任何一项不满足就发停止。
void IoContext::write(const Command& command) {
  if (!opened_ || !command_pub_ || !node_) return;

  bool control = command.control;
  bool feedback_ready = false;
  bool auto_aim = false;
  float fb_yaw = 0.0f, fb_pitch = 0.0f;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    feedback_ready = has_feedback_;
    auto_aim = has_feedback_ &&
               feedback_.work_mode.data == hnurm_interfaces::msg::WorkMode::AUTO_AIM;
    fb_yaw = feedback_.yaw;
    fb_pitch = feedback_.pitch;
  }

  if (!feedback_ready) control = false;                       // 不知道云台朝哪，不能乱转
  if (!auto_aim) control = false;                             // 非自瞄模式
  if (!std::isfinite(command.yaw) || !std::isfinite(command.pitch)) control = false;  // NaN/Inf
  if (std::abs(command.pitch) > kPitchLimitRad) control = false;                      // 超出限位

  hnurm_interfaces::msg::VisionSendData msg;
  msg.header.stamp = node_->get_clock()->now();  // ROS 时间
  msg.header.frame_id = "gimbal_link";
  msg.control_id = 0.0f;

  if (!control) {
    msg.target_state.data = hnurm_interfaces::msg::TargetState::LOST_TARGET;
    msg.target_distance = -1.0f;  // 负值：告诉仿真器本帧数据不可用
    // 回填当前角度：仿真器会采用消息里的绝对角，若一律发 0 会把云台慢慢拽向 0°，
    // 表现为「不控制时云台自己乱转」。这里发实测角 = 保持原位。
    msg.yaw = fb_yaw;
    msg.pitch = fb_pitch;
    command_pub_->publish(msg);
    return;
  }

  const bool fire = command.shoot && config.allow_fire;  // 配置禁止发射时永远不开火
  msg.target_state.data = fire ? hnurm_interfaces::msg::TargetState::FIRE
                               : hnurm_interfaces::msg::TargetState::CONVERGING;
  // Command 是绝对角(rad)，消息是角度制(deg)，方向按协议转换
  msg.yaw = static_cast<float>(command.yaw * kRadToDeg);
  msg.pitch = static_cast<float>(-command.pitch * kRadToDeg);
  msg.target_distance = 1.0f;  // 暂未测距，给一个占位正数
  command_pub_->publish(msg);
}

void IoContext::publish_stop() {
  if (!command_pub_ || !node_) return;
  hnurm_interfaces::msg::VisionSendData msg;
  msg.header.stamp = node_->get_clock()->now();
  msg.header.frame_id = "gimbal_link";
  msg.target_state.data = hnurm_interfaces::msg::TargetState::LOST_TARGET;
  msg.target_distance = -1.0f;
  msg.control_id = 0.0f;
  command_pub_->publish(msg);
}

// ---------------------------------------------------------------- Close
// 先让云台停下，再结束调度、释放 ROS2 资源。
void IoContext::close() {
  if (!opened_) return;  // 幂等：析构里还会再调一次

  publish_stop();
  std::this_thread::sleep_for(std::chrono::milliseconds(50));  // 给 DDS 一点时间把停止指令发出去

  opened_ = false;
  if (executor_) executor_->cancel();           // 让 spin() 返回，否则 join 永远等不到
  if (spin_thread_.joinable()) spin_thread_.join();
  if (executor_ && node_) executor_->remove_node(node_);

  image_sub_.reset();
  feedback_sub_.reset();
  command_pub_.reset();
  node_.reset();
  executor_.reset();

  if (rclcpp::ok()) rclcpp::shutdown();
}

}  // namespace training::io
