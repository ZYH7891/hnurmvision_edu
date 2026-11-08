#include "lessons.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace training {

// 第三讲：装甲板检测（神经网络 / YOLO11-pose + ONNX Runtime）
//
// 模型输出形状 (1, 13, N)：4 个框参数 + 1 个类别置信度 + 4 个关键点(x,y)。
// 关键点顺序按数据集定义是 左上、左下、右下、右上，需要重排成 Armor 约定的
// 左上、右上、右下、左下。
//
// 重要：模型只在进程内加载一次（第二讲 2.2 节「每帧重新加载模型」是典型性能浪费），
// 因此这里用函数内静态对象做惰性初始化。

namespace {

constexpr int kInputSize = 640;
constexpr float kConfThreshold = 0.35f;   // 置信度阈值
constexpr float kIouThreshold = 0.45f;    // NMS IoU 阈值
constexpr int kNumChannels = 13;          // 4 + 1 + 4*2

struct Letterbox {
  float scale = 1.0f;
  float pad_x = 0.0f;
  float pad_y = 0.0f;
};

// 保持长宽比缩放到 640x640，空白处填 114（与训练时一致）
cv::Mat letterbox(const cv::Mat& src, Letterbox& info, int size) {
  const int w = src.cols;
  const int h = src.rows;
  info.scale = std::min(static_cast<float>(size) / w, static_cast<float>(size) / h);
  const int new_w = static_cast<int>(std::round(w * info.scale));
  const int new_h = static_cast<int>(std::round(h * info.scale));

  cv::Mat canvas(size, size, CV_8UC3, cv::Scalar(114, 114, 114));
  info.pad_x = (size - new_w) * 0.5f;
  info.pad_y = (size - new_h) * 0.5f;
  cv::Mat resized;
  cv::resize(src, resized, cv::Size(new_w, new_h));
  resized.copyTo(canvas(cv::Rect(static_cast<int>(info.pad_x), static_cast<int>(info.pad_y), new_w, new_h)));
  return canvas;
}

// 惰性加载的检测器（全局唯一）
class PoseDetector {
 public:
  static PoseDetector& instance() {
    static PoseDetector detector;
    return detector;
  }

  bool ok() const { return session_ != nullptr; }

  std::vector<Armor> detect(const cv::Mat& bgr) {
    std::vector<Armor> armors;
    if (session_ == nullptr || bgr.empty()) return armors;

    Letterbox info;
    cv::Mat canvas = letterbox(bgr, info, input_size_);

    // BGR -> RGB, HWC -> CHW, 归一化到 [0,1]
    cv::Mat rgb;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    std::vector<float> input_data(3 * input_size_ * input_size_);
    std::vector<cv::Mat> planes(3);
    for (int c = 0; c < 3; ++c) planes[c] = cv::Mat(input_size_, input_size_, CV_32F, input_data.data() + c * input_size_ * input_size_);
    cv::split(rgb, planes);

    const std::array<int64_t, 4> input_shape{1, 3, input_size_, input_size_};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    const float* data = outputs[0].GetTensorData<float>();

    // 解析 (1, 13, num_anchors_)
    std::vector<cv::Rect2d> boxes;
    std::vector<float> scores;
    std::vector<std::array<cv::Point2f, 4>> keypoints;

    for (int i = 0; i < num_anchors_; ++i) {
      const float score = data[4 * num_anchors_ + i];
      if (score < kConfThreshold) continue;

      const float cx = data[0 * num_anchors_ + i];
      const float cy = data[1 * num_anchors_ + i];
      const float bw = data[2 * num_anchors_ + i];
      const float bh = data[3 * num_anchors_ + i];

      std::array<cv::Point2f, 4> kps;
      for (int k = 0; k < 4; ++k) {
        kps[k] = {data[(5 + 2 * k) * num_anchors_ + i], data[(6 + 2 * k) * num_anchors_ + i]};
      }
      boxes.emplace_back(cx - bw * 0.5, cy - bh * 0.5, bw, bh);
      scores.push_back(score);
      keypoints.push_back(kps);
    }
    if (boxes.empty()) return armors;

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, kConfThreshold, kIouThreshold, keep);

    for (int idx : keep) {
      Armor armor;
      const auto& kps = keypoints[idx];
      auto to_image = [&](const cv::Point2f& p) {
        return cv::Point2f((p.x - info.pad_x) / info.scale, (p.y - info.pad_y) / info.scale);
      };
      // 模型顺序 LT, LB, RB, RT -> Armor 顺序 LT, RT, RB, LB
      armor.corners[0] = to_image(kps[0]);
      armor.corners[1] = to_image(kps[3]);
      armor.corners[2] = to_image(kps[2]);
      armor.corners[3] = to_image(kps[1]);

      const cv::Rect2d& box = boxes[idx];
      const double ratio = box.width / std::max(box.height, 1e-6);
      armor.large_armor = ratio > 3.2;
      armor.class_id = 0;
      armor.confidence = scores[idx];
      armors.push_back(armor);
    }
    return armors;
  }

 private:
  PoseDetector() : env_(ORT_LOGGING_LEVEL_WARNING, "armor_pose") {
    const char* env_path = std::getenv("ARMOR_ONNX");
    const std::string model_path = env_path ? env_path : "models/armor_pose_sim.onnx";

    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(4);  // CPU 回退时：1 线程 ~100ms、4 线程 ~40ms
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // 优先用 GPU（实测 640×640 推理 46ms → 5.4ms）。失败则自动回退 CPU。
    bool gpu_ok = false;
    try {
      options.AppendExecutionProvider("CUDA", {{"device_id", "0"}});
      gpu_ok = true;
    } catch (const std::exception& e) {
      std::fprintf(stderr, "[lesson03] CUDA 追加失败: %s\n", e.what());
      gpu_ok = false;
    }
    auto open_session = [&](const Ort::SessionOptions& opts) {
      session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), opts);
      Ort::AllocatorWithDefaultOptions allocator;
      input_name_ = session_->GetInputNameAllocated(0, allocator).get();
      output_name_ = session_->GetOutputNameAllocated(0, allocator).get();
      // 输入尺寸与输出锚点数从模型读取，换 512/640 都不用改代码
      const auto in_shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
      if (in_shape.size() == 4 && in_shape[2] > 0) input_size_ = static_cast<int>(in_shape[2]);
      const auto out_shape = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
      if (out_shape.size() == 3) {
        num_channels_ = static_cast<int>(out_shape[1]);
        num_anchors_ = static_cast<int>(out_shape[2]);
      }
      std::fprintf(stderr, "[lesson03] model %dx%d  channels=%d anchors=%d\n", input_size_,
                   input_size_, num_channels_, num_anchors_);
    };
    try {
      open_session(options);
      if (gpu_ok) std::fprintf(stderr, "[lesson03] 推理后端: CUDA\n");
    } catch (const std::exception& e) {
      // GPU 不可用时回退到纯 CPU，保证程序仍能跑
      std::fprintf(stderr, "[lesson03] CUDA 不可用(%s)，回退 CPU\n", e.what());
      try {
        Ort::SessionOptions cpu_opts;
        cpu_opts.SetIntraOpNumThreads(4);
        cpu_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        open_session(cpu_opts);
        std::fprintf(stderr, "[lesson03] 推理后端: CPU\n");
      } catch (const std::exception& e2) {
        std::fprintf(stderr, "[lesson03] 加载模型失败 (%s): %s\n", model_path.c_str(), e2.what());
        session_.reset();
      }
    }
  }

  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;
  std::string input_name_;
  std::string output_name_;
  int input_size_ = 640;      // 从模型输入形状读取
  int num_anchors_ = 8400;    // 从模型输出形状读取
  int num_channels_ = 13;
};

}  // namespace

std::vector<Armor> lesson03_detection(const cv::Mat& image) {
  auto& detector = PoseDetector::instance();
  if (!detector.ok()) return {};
  return detector.detect(image);
}

}  // namespace training
