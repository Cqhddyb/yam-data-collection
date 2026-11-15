# YAM 机械臂数据采集工程

面向 YAM 主从遥操作、机器人状态、多路 RealSense 彩色图像和 MCAP/MP4 Episode
的可审计本地代码整理包。保留实际项目边界和本地修改版 i2rt，附技术文档、真实数据统计与离线测试。

[English](README.md) · [工程审计报告](docs/00_engineering_audit_report.md) ·
[完整复现指南](docs/11_yam_reproduction_guide.md) · [真实数据审计](docs/08_yam_data_audit.md)

## 1. 项目概述

完整配置为两个Leader、两个Follower和三路相机；新增单臂入口只记录选中侧的一对主从臂，
配腕部相机或腕部+顶部。2026-11-15实际读取3,400个MCAP及3,402个关联H.264视频容器，
共2,115,579帧、35,308.61秒机器人记录跨度；时长含文件中保留的ARMING样本。
另完整解码12个视频核验PTS。原始数据不包含在此包内。

## 2. 核心功能

显式左右选择、七维关节/夹爪输入和反馈、独立相机线程、有界队列、异步编码、ROS2 CDR MCAP、
source视频时间线、帧号诊断、验证后发布、数据集排他锁、腕部10FPS预览，以及显式启用的定点移动和主从配对就位。

## 3. 硬件平台

实机采集必须连接YAM和兼容Leader，单臂至少一主一从、两路CAN和腕部相机，顶部相机可选。
完整双臂布局需四条臂、四CAN、三相机。当前USB-CAN是gs_usb兼容设备，VID/PID为1d50:606f，
商业型号待确认；普通串口CAN不能直接替换。离线文件审计和合成测试无需机械臂、CAN或相机。

## 4. YAM 结构

采用YAM v1配置，每条臂六个旋转关节。配置中前3关节为DM4340、后3为DM4310；
Follower线性夹爪linear_4310也用DM4310。Leader是yam_teaching_handle被动手柄。
这些是源码/配置证据，不替代实物铭牌核验；SDK内其他模型不表示项目使用了其他型号。

## 5. Leader–Follower 遥操作

按钮0切换同步；左右独立一对一，不做镜像，没有双侧硬件同步控制屏障。
Leader最后一维为经过端点映射的手柄输入，Follower最后一维是归一化反馈位置。
Leader速度/力矩数组为空。本机bilateral Kp为0，不能宣称启用了非零双向力反馈。

## 6. CAN 通信

SocketCAN 1Mbps、restart-ms0、txqueuelen2000，启动必须为ERROR-ACTIVE。
角色名为can_leader_l/can_follower_l/can_leader_r/can_follower_r。
SDK的250Hz常量用于估算和参考，当前循环没有严格250Hz deadline，不能当作实测控制发送频率。

## 7. 机器人状态采集

目标180Hz。Follower position/velocity/effort各7个float64，Leader只有position[7]及空导数。
前六维单位为rad、rad/s、电机反馈Nm；夹爪position/velocity为归一化值及每秒变化，
effort末维仍是夹爪电机Nm。普通Episode不记录末端位姿、原始encoder count、电机温度或CAN目标命令包。

## 8. 多路相机采集

每设备独立RealSense color pipeline、native queue及读取线程，实机需要2.56.x RealSense SDK；
已审计环境为pyrealsense2 2.56.5.9235。仅记录彩色流，无Depth；更换相机厂家需要新增backend与验收。

## 9. 相机配置

| 名称 | 型号/角色 | 60Hz | 30Hz | 输入 |
|---|---|---|---|---|
| cam_high | D435/顶部 | 960×540@60 | 1280×720@30 | YUYV |
| cam_left | D405/左腕 | 848×480@60 | 1280×720@30 | YUYV |
| cam_right | D405/右腕 | 848×480@60 | 1280×720@30 | YUYV |

唯一双臂样本高/左/右实测约59.530/59.895/59.853FPS；大量单腕数据另按组统计。
设备序列号已替换为占位符，历史内参数值仅用于展示配置结构，必须在目标机核对或重测。

## 10. MCAP 记录

Schema是JointState、CameraInfo、String，消息CDR编码，不要求运行ROS节点。
双臂九个Topic，单臂单腕五个Topic；相机Topic虽然包含image_raw/compressed字样，
实际只有CameraInfo，像素保存在MP4。另有/task_info与/diagnostics/frame_timing。

## 11. 视频编码

当前libx264、CRF23、ultrafast、每路1线程、avc1、yuv420p；采集时持续编码，结束时通常remux重建source PTS。
历史HEVC配置是停止后转码的另一条路径。没有显式配置GOP长度，不能编造固定GOP参数。

## 12. 时间戳管理

机器人Header/publish time是worker导出缓存snapshot的主机时间；log time是采集器本批采样时间，
排队后仍保留。相机优先RealSense global time，另保留host receive。它们不是每个电机的硬件采样时间，也不证明同时曝光。

## 13. 数据同步

CameraInfo与视频按同路接受帧顺序对应，source_timeline定义Unix origin和微秒PTS。
12个完整解码样本存在−972至−10µs恒定mux起始偏移，逐帧相对时间线一致；使用方需读取实际PTS并计入偏移。
yam_60hz允许59.0–60.5FPS、69ms间隔、17ms时间匹配误差；strict是单独标准。
现存文件有1,414个设备帧号缺口和重复机器人源时间戳，因此“通过验收”不等于源完全无丢帧。

## 14. Episode 管理

空格后约2秒ARMING，等Recording episode再遥操作；Enter保存，复核提示Enter保留；q丢弃/删除，
空闲Ctrl+C退出采集器。原程序可自动删除正式质量失败数据。Ctrl+C不是硬件急停，无单条中途暂停/续录保证。

## 15. 数据结构

```text
dataset/
├── record_config.yaml
├── 0/
│   ├── YYYY-MM-DD_HH-MM.mcap
│   └── videos/
│       ├── cam_left.mp4
│       ├── cam_high.mp4             # 仅启用顶部时
│       └── source_timeline.json
└── runtime_diagnostics/             # 启用时
```

配置快照在实际部署后可能带设备身份，原始数据与host配置应独立于公开仓库管理。

## 16. 代码目录

src/i2rt保留SDK与MIT许可证/模型；src/yam_data_collection是基础及queued采集、编码、验证和导出；
src/yam_single_arm_collection是选中侧采集、预览和运动扩展；src/deploy及single_arm_transfer含CAN模板、CPU适配。
scripts含安装、离线测试、只读数据审计；configs是脱敏模板；docs是完整技术交接。
[核心清单](docs/12_core_code_inventory.md)和SOURCE_MANIFEST.json记录来源及SHA-256。

## 17. 环境要求

Linux x86_64、推荐Python3.11、FFmpeg/ffprobe含libx264；实机另需iproute2、can-utils、gs_usb、USB权限。
预览才需要Tk/display。两个uv环境保留各自锁文件。录制库无需ROS运行时，原wrapper仍保留部署主机ROS Humble library path。

## 18. 安装

```bash
# release根目录；先自行安装uv
sudo apt-get install ffmpeg can-utils iproute2 usbutils
bash scripts/setup_env.sh
bash scripts/test_offline.sh
```

setup重建Python环境，不配置CAN或启动电机。ruckig0.15.3可能需C++编译环境。
仅离线采集库可在src/yam_data_collection中执行uv sync --frozen --python 3.11 --extra dev --extra timeline。
requirements.txt不代替SDK的pyproject与锁文件。

## 19. 硬件部署

按[复现指南](docs/11_yam_reproduction_guide.md)一致替换启动器/udev/安装器的CAN serial占位符，
填写相机serial、任务、内参和手柄端点，适配CPU。历史数值不代表新机器已通过标定或资源验收。

## 20. 机器人初始化

完成现场检查和CAN配置后，在独立终端依次：

```bash
bash yam_single_arm.sh follower --side left --confirm-motion
bash yam_single_arm.sh leader --side left --confirm-motion
```

启动可能使能电机、校准夹爪。配对就位需用leader-motion替代普通leader；默认各自实时起点，
计划关节限速1rad/s，动作需MOVE/PREPOSITION精确确认，无碰撞规划。

## 21. 相机初始化

```bash
src/yam_data_collection/.venv/bin/yam-data-collect discover-cameras
bash yam_single_arm.sh camera-test --side left --cameras wrist --duration 180 --no-video
bash yam_single_arm.sh camera-test --side left --cameras wrist --duration 60
```

测试与采集不同时抢占相机，每种组合/模式单独验收。早期30Hz右腕测试失败尚无文档闭环。

## 22. 数据采集

先修改真实任务信息，再开第三终端：

```bash
bash yam_single_arm.sh check --side left --cameras wrist
bash yam_single_arm.sh capture --side left --cameras wrist --preview
```

另一侧用right，加顶部用wrist-top。双臂需先初始化四角色，再bash yam.sh capture --preview。
先做20–30秒端到端短段验收，完整流程及可选配对就位见11号文档。

## 23. 数据验证

```bash
bash yam_single_arm.sh validate /path/to/episode.mcap \
  --side left --cameras wrist --mode 60hz \
  --config /path/to/record_config.yaml --json
bash summarize_dataset_duration.sh /path/to/dataset
```

双臂验证器与14维LeRobot导出器要求四状态/三相机，不直接支持单臂。
scripts/audit_dataset.py可按原始data根目录布局重算统计，输出放在输入数据目录之外。
Dream验证是独立可选SDK集成，本次未测试。

## 24. 故障排查

UP不等于CAN健康；区分source FPS、host交付和编码吞吐。阅读设备帧号、global/host时间、队列和pipe耗时。
相机首帧错误保留角色上下文。[历史复盘](docs/09_yam_debugging.md)区分已证实根因与未闭环问题。
没有通用掉电损坏MCAP无损恢复承诺。

## 25. 当前限制

可访问工作区无Git历史，不能提供commit级追溯。大多数单臂文件仍为smoke任务ID，需要人工复核实际任务。
没有硬件曝光同步、CAN命令记录、独立执行时刻或掉电原子提交保证。早期D435掉帧和右CAN通信异常的最终根因未完整确认。
异路径脱敏副本通过687项离线测试：461项采集/单臂/部署、43项模拟控制、183项模型装配/仿真；依赖复用既有安装库，
尚未验证目标机全新联网安装或实机硬件。软件回归不能代替现场验收。

## 26. 依赖与致谢

I2RT Robotics SDK、RealSense、python-can、Portal、MCAP ROS2支持、NumPy、PyYAML、MuJoCo、PyAV及FFmpeg。
见[第三方说明](THIRD_PARTY_NOTICES.md)和src/i2rt/LICENSE。采集项目原目录没有单独LICENSE，
不能默认SDK的MIT覆盖全部项目，发布前由所有者选择。此交付没有GitHub上传，不含原始数据和凭据。
