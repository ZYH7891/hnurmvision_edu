# 跃鹿战队算法组二阶段培训与考核

本仓库是面向学员的自瞄教学骨架。

## 个人信息

- 学号：202693020517
- 姓名：周瑀涵
- 学校：湖南大学
- 专业：机器人工程

## 目录结构

```text
hnurmvision-training/
├── CMakeLists.txt              # 培训主程序构建配置
├── README.md
├── configs/
│   └── simulation.yaml         # 显示、发射许可、相机内参和坐标变换
├── include/
│   ├── lessons.hpp             # 第二至第五讲函数声明
│   ├── tracking_state.hpp      # 第四讲跨帧状态
│   └── types.hpp               # 各讲之间传递的数据类型
├── io/
│   ├── config.cpp/.hpp         # YAML 读取和校验
│   └── io_context.hpp          # 第二讲 IO 状态与四个操作声明
├── lessons/
│   ├── lesson02_io.cpp         # 第二讲：仿真 IO
│   ├── lesson03_detection.cpp  # 第三讲：装甲板检测
│   ├── lesson04_tracking.cpp   # 第四讲：位姿与跟踪
│   └── lesson05_aiming.cpp     # 第五讲：弹道与瞄准
├── src/
│   └── main.cpp                # 单线程顺序主循环
├── scripts/                    # 构建和启动入口
├── docs/                       # 课程讲义和仿真环境简介
└── simulation/
    ├── simulator/              # Bevy 仿真器源码、资产和 config.toml
    ├── ros2/                   # 仿真通信使用的 hnurm_interfaces
    └── tools/                  # 场地生成辅助工具
```

## 编译

### 环境要求

- Ubuntu 22.04
- ROS2 Humble、colcon、ament_cmake
- CMake、支持 C++17 的编译器
- OpenCV、Eigen3、yaml-cpp
- Rustup 和 Rust 1.95.0（只在编译仿真器时需要）

所有命令都在仓库根目录执行。首次使用时先编译主程序及 ROS2 消息，再编译仿真器。

### 1. 编译培训主程序

```bash
./scripts/build_training.sh
```

该脚本依次完成：

1. 编译 `simulation/ros2/hnurm_interfaces`；
2. 配置根目录 CMake 工程；
3. 生成 `build/training`。

### 2. 编译仿真器

```bash
./scripts/build_simulator.sh
```

该脚本会生成教学场地、以 ROS2 特性编译 Bevy 仿真器，并生成：

```text
simulation/simulator/bin/daedalus
```

首次编译时间较长。修改仿真器 Rust 源码、资产生成逻辑或 ROS2 接口后需要重新执行；只修改
`lessons/`、`io/` 或 `src/` 时，重新执行 `build_training.sh`。

## 运行指南

运行时需要两个终端，并且两个终端都位于仓库根目录。

### 终端 1：启动仿真器

```bash
./scripts/run_simulator.sh
```

看到仿真窗口和训练场地后，再启动培训程序。脚本会自动配置 ROS2 环境、
`ROS_DOMAIN_ID=89`、动态库路径和仿真资产路径。

### 终端 2：启动培训程序

```bash
./scripts/run_training.sh
```

该脚本使用 `configs/simulation.yaml` 启动 `build/training`。初始骨架等待输入，按 Ctrl+C 退出；此时没有图像窗口和 ROS2 通信。


### 退出

- 完成第二讲并显示图像后，在培训图像窗口按 `q` 或 `Esc`；
- 或在对应终端按 `Ctrl+C`；
- 建议先退出培训程序，再关闭仿真器。

### 常用配置

运行配置位于 `configs/simulation.yaml`；学员还需要完成 `lessons/` 和 IO 骨架：

- `display`：是否显示培训图像窗口；
- `allow_fire`：是否允许最终发射，默认 `false`；
- 相机内参、畸变参数和相机到云台的坐标变换。

仿真场景和靶车由 `simulation/simulator/config.toml` 配置。`[training] enabled=true` 时生成
教学观察车和训练靶，设为 `false` 时回到手动场景。

## 第三讲：装甲板识别

### A. 程序实现（神经网络）

`lessons/lesson03_detection.cpp` 用 YOLO11-pose 模型 + ONNX Runtime 推理：

1. letterbox 缩放到 640×640（保持长宽比，空白填 114）；
2. ONNX Runtime 推理，输出 (1,13,8400)：框 + 置信度 + 4 关键点；
3. 置信度阈值 0.35 + NMS（IoU 0.45）；
4. 关键点顺序 LT/LB/RB/RT 重排为 `Armor` 约定的 LT/RT/RB/LB；
5. `src/main.cpp` 把四角连框画在相机画面上（大板橙色、小板绿色）。

模型在进程内只加载一次（`models/armor_pose_sim.onnx`，可用环境变量 `ARMOR_ONNX` 覆盖）。

### B. 神经网络训练产物

数据集：14 类装甲板（B1-B5/BO/BS、R1-R5/RO/RS），YOLO pose 格式（框 + 4 关键点）。

训练配置：`yolo11n-pose.pt` 预训练，子集 1200 训练 / 400 验证，30 轮，imgsz 640，batch 8。

验证集指标：Box mAP50 = 0.875，mAP50-95 = 0.585；Pose mAP50 = 0.737。

训练产物（`docs/lesson03/`）：

| 文件 | 说明 |
|------|------|
| `weights/best.pt` | 最佳权重 |
| `weights/last.pt` | 最终权重 |
| `curve_results.png` | loss / mAP 训练曲线 |
| `curve_box_pr.png`、`curve_pose_pr.png` | PR 曲线 |
| `confusion_matrix.png` | 混淆矩阵 |
| `val_pred.jpg` | 验证集预测可视化 |
| `inference_conf.png` | 推理可视化（带类别与置信度） |

复现训练：

```bash
yolo pose train model=yolo11n-pose.pt data=armor_subset.yaml epochs=30 imgsz=640 batch=8 device=0
```

> 两套模型的分工：实拍数据集（7046 张）训练的模型对仿真渲染风格泛化不足，仅作
> 训练练习与产物提交；程序内实际使用的是在仿真画面上自动标注的 1016 帧上微调的
> 模型（`docs/lesson03/sim/`），因此仿真里能稳定检出。
>
> 仿真模型指标：Box mAP50 = 0.995，Pose mAP50 = 0.995（验证集 204 张）。

### D. 识别效果优化（v1 → v2）

评测方法：固定测试集 `test_hard`（100 张含装甲板 + 100 张纯背景），
脚本 `tools/eval_detector.py`，IoU=0.5、conf=0.35，与 C++ 部署路径一致（ONNX Runtime）。

| 版本 | 训练数据 | Precision | Recall | F1 | 单帧耗时 |
|------|---------|-----------|--------|----|---------|
| v1 | 1016 正样本 | 0.925 | 0.971 | 0.947 | 41.3 ms |
| v2 | 1651 张（含 600 负样本）+ 光照/几何增强，60 轮 | 0.934 | 0.971 | 0.952 | 36.4 ms |
| **v3** | 同上，几何增强加强（`degrees=15`、`shear=5`、`perspective=0.001`、`scale=0.7`） | **0.943** | 0.971 | **0.957** | **31.6 ms** |

#### 板型判定修正（距离解算正确性）

原实现用包围框宽高比猜大小板，再据此选 135/230 mm 作 PnP 的板宽。斜视时框会被压窄，
大板可能被判成小板，而板宽直接决定 `solvePnP` 的深度——判错就是 230/135 ≈ 1.7 倍距离误差。

改为「两种尺寸各解一次 PnP，取重投影误差更小者」。两种板高度相同(55 mm)、宽度不同，
不是纯缩放关系，因此残差可区分。实测：

| 场景 | err_S | err_L | 判定 |
|------|-------|-------|------|
| 正视小板 | 0.07~0.12 px | 0.21~0.43 px | S（正确） |
| 另一块板 | 0.35 px | **0.13 px** | L（正确切换） |

距离精度：仿真靶真实距离 √(3²+2²)=3.606 m，估计 3.617 m，**误差 0.3%**。

| **v4** | 同 v3，输入 512×512 重训 40 轮 | **0.952** | 0.971 | **0.961** | **29.1 ms** |

> v4 是当前部署版本（`models/armor_pose_sim.onnx`，512×512 输入）。输入尺寸与锚点数
> 在加载时从 ONNX 形状自动读取，换 512/640 不用改代码。相比 v2：误检 7→6，
> F1 +0.005，单帧耗时 36.4→31.6 ms。侧视/斜视鲁棒性主要靠 `shear` 与
> `perspective` 增强获得。

优化手段：

- **降低误检**：加入 600 张纯背景负样本，并用 100 张纯背景做专项测试（v1 误检 8 → v2 误检 7）；
- **提升光照/角度鲁棒性**：`hsv_h/s/v`、`degrees=8`、`translate=0.15`、`scale=0.6`、`perspective`、`mixup=0.1`；
- **提升稳定性**：60 轮 + `cos_lr`，`close_mosaic=15`（后期关闭马赛克，收敛更稳）；
- **实时性**：ORT 线程 1→4（100ms→40ms）、8 线程 35ms；int8 动态量化因 `ConvInteger`
  在 CPU EP 不支持而放弃；
- **自适应尺寸**：训练用 `scale=0.6` 缩放增强 + 关键点监督，远近距离共用同一模型。

### E. 性能剖析与优化

`PRINT_STATS=1 ./scripts/run_training.sh` 每秒输出各阶段耗时。实测（仿真器在线）：

| 阶段 | 每帧耗时 | 说明 |
|------|---------|------|
| `detect` | **26 ms**（512 输入） | 占整圈 ~75%，仍是主要开销 |
| `read` | 0.8~6 ms（偶发 20~30 ms） | 已改为缓冲所有权移交，去掉整图拷贝 |
| `draw` | 2.4 ms | `clone` + `imshow` |
| `track` / `aim` / `write` | < 0.2 ms | 可忽略 |
| 整圈 | **33~36 ms** | **28~30 fps**（相机 24 fps，已有余量） |

尝试过的提速方案：

| 方案 | 结果 |
|------|------|
| int8 动态量化 | ❌ `ConvInteger` 在 CPU EP 不支持 |
| QDQ 静态量化 | ⚠️ 22.5 ms（快 2.2×）但精度归零，YOLO 解码头对量化敏感 |
| ONNX Runtime GPU（Python 验证） | ✅ 8.7 ms，精度不变（F1 0.957） |
| C++ 端 GPU | ⏸ 需要官方 GPU 版 `libonnxruntime.so`，pip 包内的是核心库不注册 CUDA；代码已写好「优先 CUDA、失败回退 CPU」 |

### C. 演示视频

`docs/lesson03/demo.mp4`：19 秒 / 10 fps / 1120×540 / mp4v。
相机画面实时绘制装甲板识别框，云台缓慢扫动，画面内两台靶车的装甲板同时被框住
（190 帧中 180 帧检出）。
