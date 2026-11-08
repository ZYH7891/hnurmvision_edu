#include "lessons.hpp"
#include "io_context.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
volatile std::sig_atomic_t interrupted = 0;
void on_signal(int) { interrupted = 1; }

using Clock = training::Clock;

// 各阶段累计耗时（毫秒），用于定位瓶颈：先测量，再优化。
struct StageTimer {
  double read = 0, detect = 0, track = 0, aim = 0, write = 0, draw = 0, total = 0;
  int n = 0;

  static double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  }

  void report() const {
    const double k = n > 0 ? 1.0 / n : 0.0;
    const double loop_ms = total * k;
    std::fprintf(stderr,
                 "[stats] n=%d  read=%.2f detect=%.2f track=%.2f aim=%.2f write=%.2f draw=%.2f"
                 "  |  loop=%.2f ms  ~%.1f fps\n",
                 n, read * k, detect * k, track * k, aim * k, write * k, draw * k, loop_ms,
                 loop_ms > 0 ? 1000.0 / loop_ms : 0.0);
  }
};

// 把第三讲的识别结果画到相机画面上：装甲板四角连成的框 + 四个角点。
void draw_armors(cv::Mat& image, const std::vector<training::Armor>& armors) {
  for (const auto& armor : armors) {
    std::vector<cv::Point> pts;
    pts.reserve(armor.corners.size());
    for (const auto& corner : armor.corners) pts.emplace_back(cv::Point2i(corner));

    // 大装甲板用橙色，小装甲板用绿色，方便一眼区分
    const cv::Scalar color = armor.large_armor ? cv::Scalar(0, 165, 255) : cv::Scalar(0, 255, 0);
    cv::polylines(image, pts, /*isClosed=*/true, color, 2);
    for (const auto& p : pts) cv::circle(image, p, 3, cv::Scalar(0, 0, 255), cv::FILLED);
  }
}
}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  const auto config = training::load_config(argc > 1 ? argv[1] : "configs/simulation.yaml");
  const auto calibration = training::load_calibration(config);
  training::io::IoContext devices{config};
  training::TrackingState tracking;
  devices.open();

  const bool print_stats = std::getenv("PRINT_STATS") != nullptr;
  const bool print_dist = std::getenv("PRINT_DIST") != nullptr;
  const bool demo_window = std::getenv("DEMO_WINDOW") != nullptr;
  StageTimer timer;
  int frame = 0;
  int win_loops = 0, win_inputs = 0;

  long loops = 0;
  while (!interrupted) {
    ++loops;
    const auto t0 = Clock::now();
    const auto input = devices.read();
    const auto t1 = Clock::now();

    training::Command command{};
    std::vector<training::Armor> armors;
    if (input) {
      armors = training::lesson03_detection(input->image);
      const auto t2 = Clock::now();

      const auto target = training::lesson04_tracking(
          armors, input->timestamp, input->imu_q, calibration, tracking);
      const auto t3 = Clock::now();

      if (target) {
        command = training::lesson05_aiming(*target, input->timestamp, input->bullet_speed);
        if (print_dist) {
          static int dbg = 0;
          if (++dbg % 20 == 0)
            std::fprintf(stderr, "[dist] %.3f m  n_armors=%zu\n", target->position.norm(),
                         armors.size());
        }
      }
      const auto t4 = Clock::now();

      timer.detect += StageTimer::ms(t1, t2);
      timer.track += StageTimer::ms(t2, t3);
      timer.aim += StageTimer::ms(t3, t4);
    }
    if (interrupted) break;
    const auto t5 = Clock::now();
    devices.write(command);
    const auto t6 = Clock::now();
    timer.write += StageTimer::ms(t5, t6);

    if (input && devices.config.display) {
      cv::Mat canvas = input->image.clone();  // 画框不污染算法输入
      draw_armors(canvas, armors);
      cv::imshow("RoboMaster training", canvas);
      if (demo_window) {
        static int ui_frames = 0;
        if (++ui_frames < 60) {
          cv::moveWindow("RoboMaster training", 0, 0);
          cv::resizeWindow("RoboMaster training", 1280, 960);
          cv::setWindowProperty("RoboMaster training", cv::WND_PROP_TOPMOST, 1);
        }
      }
      const int key = cv::waitKey(1);
      if (key == 27 || key == 'q') break;
    }
    const auto t7 = Clock::now();
    if (print_stats) {
      timer.read += StageTimer::ms(t0, t1);
      timer.write += StageTimer::ms(t5, t6);
      timer.draw += StageTimer::ms(t6, t7);
      timer.total += StageTimer::ms(t0, t7);
      ++win_loops;
      if (input) ++win_inputs;

      static Clock::time_point last_report = Clock::now();
      if (t7 - last_report >= std::chrono::seconds(1)) {
        const double sec = std::chrono::duration<double>(t7 - last_report).count();
        const double div = win_inputs > 0 ? win_inputs : 1;
        std::fprintf(stderr,
                     "[stats] 1s: loops=%d(%.0f/s) inputs=%d(%.0f fps) | "
                     "per-input read=%.2f detect=%.2f track=%.2f aim=%.2f write=%.2f draw=%.2f "
                     "loop=%.2f ms\n",
                     win_loops, win_loops / sec, win_inputs, win_inputs / sec, timer.read / div,
                     timer.detect / div, timer.track / div, timer.aim / div, timer.write / div,
                     timer.draw / div, timer.total / div);
        last_report = t7;
        timer = StageTimer{};
        win_loops = 0;
        win_inputs = 0;
      }
    }
    (void)frame;
  }

  devices.close();
  if (print_stats) { std::fprintf(stderr, "[stats] loops=%ld  inputs=%d\n", loops, timer.n); }
  if (print_stats && timer.n > 0) timer.report();
}
