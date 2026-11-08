#pragma once
#include "config.hpp"
#include "types.hpp"

#include <hnurm_interfaces/msg/vision_recv_data.hpp>
#include <hnurm_interfaces/msg/vision_send_data.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

namespace training::io {

// IO 层对主程序只暴露四个动作：Open / Read / Write / Close。
// 通信状态都是本对象的成员，生命周期与 IoContext 一致。
struct IoContext {
  explicit IoContext(Config config_in = Config{}) : config(std::move(config_in)) {}
  ~IoContext();

  IoContext(const IoContext&) = delete;
  IoContext& operator=(const IoContext&) = delete;

  Config config;

  void open();
  std::optional<SensorData> read();
  void write(const Command& command);
  void close();

 private:
  // ① 通信对象：销毁即断开，所以要活到 close()
  rclcpp::Node::SharedPtr node_;
  rclcpp::executors::SingleThreadedExecutor::SharedPtr executor_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<hnurm_interfaces::msg::VisionRecvData>::SharedPtr feedback_sub_;
  rclcpp::Publisher<hnurm_interfaces::msg::VisionSendData>::SharedPtr command_pub_;
  std::thread spin_thread_;

  // ② 跨线程共享的数据（回调在后台线程写，read/write 在主线程读）
  std::mutex mutex_;
  cv::Mat latest_image_;                     // 最近一帧，BGR、独立拥有像素内存
  TimePoint image_time_{};                   // 该帧的采集时刻（已换算到 Clock）
  std::uint64_t image_seq_ = 0;              // 已收到的帧序号
  std::uint64_t delivered_seq_ = 0;          // 已交给算法的帧序号
  hnurm_interfaces::msg::VisionRecvData feedback_{};
  bool has_feedback_ = false;
  bool opened_ = false;
  TimePoint last_status_time_{};             // 状态打印节流（每秒一行）

  // ③ 回调：声明放这里，定义放 .cpp
  void on_image(const sensor_msgs::msg::Image::SharedPtr msg);
  void on_feedback(const hnurm_interfaces::msg::VisionRecvData::SharedPtr msg);
  void publish_stop();
};

}  // namespace training::io
