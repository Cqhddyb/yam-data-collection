# 完整采集复现指南

本文所有动作命令供现场操作者手动执行。本次审计只执行离线读取和软件测试。公开包的设备身份、任务和右手柄端点已模板化，未填写时不能用于实机采集。

## 1. 环境与安装

目标为Linux x86_64、Python3.11；采集包声明>=3.10,<3.13。需要SocketCAN/gs_usb、iproute2、can-utils、ffmpeg/ffprobe含libx264，source视频时间线需要PyAV16，RealSense需要2.56.x的pyrealsense2及可用系统驱动/USB权限。录制与离线验证不要求ROS运行时；本机wrapper保留了`/opt/ros/humble/lib`的library path，迁移需确认系统实际库来源。

从`release/`根目录：

```bash
sudo apt-get install ffmpeg can-utils iproute2 usbutils
bash scripts/setup_env.sh
bash scripts/test_offline.sh
```

setup脚本分别执行两个项目的`uv sync --frozen --python 3.11`，采集环境加camera/timeline/dev extras；不复用打包虚拟环境。`requirements.txt`仅为采集/审计便携安装的依赖说明，硬件控制依赖以本地修改版i2rt的pyproject和uv.lock为准。ruckig0.15.3可能需本地编译，SDK配置已经约束scikit-build-core<0.10；准备C++工具链后按实际错误安装依赖。

可选双臂LeRobot导出依赖`lerobot` extra中的pyarrow>=16；棋盘标定依赖`calibration` extra中的opencv-python>=4.8。需要时从采集项目目录运行下面命令，同时保留采集所需extras：

```bash
cd src/yam_data_collection
uv sync --frozen --python 3.11 --extra camera --extra timeline --extra dev --extra lerobot --extra calibration
cd ../..
```

不具备硬件时可只安装采集包dev/timeline，在合成图像和MCAP测试中运行。CAN仿真测试仍需python-can等控制库依赖，但不需要插USB-CAN。

## 2. 填写目标机配置

```bash
cp configs/host.template.yaml src/yam_data_collection/config.host.yaml
cp configs/single_arm.template.yaml src/yam_single_arm_collection/config.host.yaml
```

按真实设备替换camera serial、任务信息和目标输出路径；重新获得各分辨率内参及真实畸变模型。template保留历史数值便于理解结构，不能当成新设备标定。不要将allow_uncalibrated_intrinsics设置true用于正式采集。

CAN四角色身份需同时替换`src/yam.sh`、`src/yam_single_arm_collection/leader_motion.sh`、`src/deploy/install_can.sh`和`src/deploy/90-yam-can.rules`中的同名占位符。右手柄`REPLACE_RIGHT_GRIPPER_CLOSED/OPEN`也要由现场被动测量结果替换。USB serial身份不是物理左右接线的替代证明。

CPU编号应适配目标机，先预览再应用到**release副本**：

```bash
src/yam_data_collection/.venv/bin/python src/single_arm_transfer/configure_cpu.py
src/yam_data_collection/.venv/bin/python src/single_arm_transfer/configure_cpu.py --apply
```

算法要求至少6个可用逻辑CPU，自动分组只是建议；资源分区仍需实机验收。原机0-11/12-15/16-19不是最低硬件规格。

## 3. 硬件连接和CAN初始化

供电、线缆、终端电阻、机械支撑、夹爪端点和急停由现场人员检查。默认完整布局需四条YAM、四个gs_usb兼容CAN接口和三相机；单臂只需选中侧一Leader、一Follower、两CAN以及腕部D405，可选顶部D435。只做离线验证不需要这些硬件。

维护窗口内，控制器停止后，操作者执行：

```bash
# 完整四角色安装；只用单侧时选择下面的单侧命令
sudo bash src/deploy/install_can.sh
# 单侧示例：
bash src/single_arm_transfer/install_can_single.sh left --preview
sudo bash src/single_arm_transfer/install_can_single.sh left
ip -details -statistics link show can_follower_l
ip -details -statistics link show can_leader_l
```

四角色与单侧安装二选一。要求UP、ERROR-ACTIVE、1Mbps、restart-ms0、txqueuelen2000；安装操作会改系统接口，不能与采集并行。ERROR-PASSIVE需先排查通信，不能只看UP便启动电机。

## 4. 初始化机器人

两个独立终端，先Follower后Leader。启动可使能电机、校准夹爪：

```bash
bash yam_single_arm.sh follower --side left --confirm-motion
bash yam_single_arm.sh leader --side left --confirm-motion
```

若要采集前配对就位，Leader终端使用下列入口替代普通leader，不要同时启动两者：

```bash
bash yam_single_arm.sh leader-motion --side left --confirm-motion
```

第三个终端仅检查配置和状态：

```bash
bash yam_single_arm.sh show-config --side left --cameras wrist
bash yam_single_arm.sh check --side left --cameras wrist
bash yam_single_arm.sh read-poses --side left
```

默认普通采集不要求预编程移动；leader-motion接口默认关闭。右侧替换side，腕部+顶部替换为wrist-top。不能向同一硬件启动重复控制器。

## 5. 相机检查

```bash
src/yam_data_collection/.venv/bin/yam-data-collect discover-cameras
bash yam_single_arm.sh camera-test --side left --cameras wrist --duration 180 --no-video
bash yam_single_arm.sh camera-test --side left --cameras wrist --duration 60
```

测试与采集不能同时占用相机。先检查report.json的source时序，再检查视频codec/尺寸/帧数。60Hz一次通过不代表其他组合或30Hz通过。30Hz使用`--mode 30hz`并单独验收。

## 6. 可选目标移动与配对就位

复制`src/yam_single_arm_collection/plans/left_target.template.json`为新计划，将目标占位符替换为实测六个rad及一个[0,1]手柄/夹爪值；设置实际安全的duration_s。无效字符串模板不能执行。

```bash
# 只读预览；默认每次读取实时起点
bash yam_single_arm.sh move --side left --plan measured_target.json
bash yam_single_arm.sh preview-preposition --side left --plan measured_target.json

# 操作者确定工作区安全后，显式执行单Follower移动
bash yam_single_arm.sh move --side left --plan measured_target.json --execute --confirm MOVE

# 或执行配对就位后进入采集
bash yam_single_arm.sh capture --side left --cameras wrist --preposition measured_target.json --confirm PREPOSITION
```

Leader手柄夹爪不能被程序驱动，需先人工靠近计划末维；主从应保持未同步，放开按钮0。当前关节上限1rad/s、夹爪0.25/s、到位0.12rad/0.12，且无碰撞规划；提高限速不会自动缩短计划duration。

## 7. 正式单臂录制

```bash
bash yam_single_arm.sh capture --side left --cameras wrist --preview
```

空格进入约2秒ARMING，等`Recording episode`后操作。Enter保存，校验结束后的复核提示Enter保留；`q`丢弃/删除当前episode，不是退出键。空闲Ctrl+C退出采集器；机器人服务另按现场停机规程关闭。先完成20–30秒端到端试采，再扩大采集。

默认单臂输出在`src/yam_single_arm_collection/data/<side>/<setup>/<mode>/`。填写task_info；现存数据说明默认smoke标签会造成任务统计不准确。

## 8. 双臂流程

在四个终端先启动`bash yam.sh follower-l --confirm-motion`、`follower-r`，再启动`leader-l`、`leader-r`。第三方SDK入口与本机wrapper的参数不要混用。完成后：

```bash
bash yam.sh check
bash yam.sh camera-test --duration 180 --no-video
bash yam.sh camera-test --duration 60
bash yam.sh capture --preview
```

wrapper使用queued状态写入和source时间线；基础`yam-data-collect capture`行为不同。切30Hz用`YAM_MODE=30hz`，必须独立验收。

## 9. 保存后离线验证

```bash
# 单臂：路径替换为真实成品和其record_config.yaml
bash yam_single_arm.sh validate /path/to/episode.mcap --side left --cameras wrist --mode 60hz --config /path/to/record_config.yaml --json

# 双臂：
src/yam_data_collection/.venv/bin/yam-data-collect validate /path/to/episode.mcap --config /path/to/record_config.yaml --mode 60hz

# 递归时长；每episode一次
bash summarize_dataset_duration.sh /path/to/dataset
```

除正式pass/fail外阅读frame_timing、设备缺号、队列和逐臂图像同步；保留验证报告。公开包的`scripts/audit_dataset.py`可对与原项目相同的data根结构重算本次全量统计；输出必须放在输入数据目录之外。

安装lerobot extra后，双臂source文件可用`src/yam_data_collection/.venv/bin/python src/yam_data_collection/scripts/convert_episode_to_lerobot_v21.py --help`查看本地导出参数；单臂不支持该四流14维导出器。不要对原文件运行update_task_info或repair作为只读验收步骤。

## 10. 故障定位

相机超时检查角色、占用、USB链路和profile；CAN错误检查真实state、供电/接线，参见09；Portal连接失败先确认对应服务/端口和Python环境。视频队列上升时查看runtime_diagnostics的pipe耗时、编码吞吐与系统I/O，不能通过增大无界队列掩盖问题。未正常关闭的文件先复制到隔离位置，再评估是否可恢复，软件没有通用无损修复承诺。
