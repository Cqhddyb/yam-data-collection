# 机器人状态、动作语义与记录字段

## 六种容易混淆的数据

| 数据 | 实际来源和含义 | 维度/类型/单位 | 去向 |
|---|---|---|---|
| Leader 测量状态 | DM 反馈经方向/offset处理的六关节缓存；原始 encoder count 不直接保存 | 6×float64，rad | `get_info()` 六维部分 |
| Follower 测量状态 | DM 反馈→`JointMapper`→robot observations | 6关节+夹爪；double | Follower telemetry/MCAP |
| Leader 操作输入 | 六关节位置+经端点映射的被动手柄开合 | 7×float64；前6 rad，末维无量纲 | Leader telemetry、目标命令来源 |
| Follower 目标命令 | 同步接通时 Leader 数组，或显式预编程插值 | 7维；rad+[0,1] | RPC命令队列→SDK；常规采集未单独保存此流 |
| 实际执行后的状态 | Follower 反馈缓存，包含跟踪误差、延迟与夹爪限制效果 | position/velocity/effort 各7维 | Follower JointState |
| 导出数据集 Action | 双臂 LeRobot 导出脚本按目标时间最近邻选 Leader position，再拼接右→左 | 14×float32 | Parquet `action`；不是 Follower 实测状态或 CAN 包 |

Leader 第一至六维是观测到的关节位置，可用作示教目标来源；最后一维已是操作输入映射。称整条 Leader Topic 为“纯原始测量”不准确。常规 MCAP 未保留 `synchronized`，所以不能仅凭 Leader 位置断言该时刻已发送同值控制命令。

## 字段表

| 字段 | 数据源 | 维度 | CDR类型 | 单位 | 记录目标频率 | 保存位置 |
|---|---|---:|---|---|---:|---|
| Leader `position` | `YAMLeaderRobot.get_info` | 7 | float64[] | 6×rad + 归一化手柄输入 | 180 Hz | leader JointState |
| Leader `velocity/effort` | writer 明确置空 | 0 / 0 | float64[] | 无 | 随 JointState | leader JointState；不是七维零测量 |
| Follower `position` | `_robot_observations_to_telemetry` | 7 | float64[] | 6×rad + 归一化夹爪开合 | 180 Hz | follower JointState |
| Follower `velocity` | `JointMapper.to_command_joint_vel_space` | 7 | float64[] | 6×rad/s + 归一化开合/s | 180 Hz | follower JointState |
| Follower `effort` | 电机反馈 `eff` | 7 | float64[] | 电机协议力矩 Nm；末维也是夹爪电机力矩 | 180 Hz | follower JointState |
| `header.stamp` | telemetry 导出端 `time.time_ns()` | sec+nanosec | int32+uint32 | Unix秒/纳秒 | 随消息 | JointState Header |
| `name` | writer 固定顺序 | 7 | string[] | 无 | 随消息 | joint1..joint6,gripper |
| sequence | 遥测端共享计数 | 标量 | Python int | 无 | worker更新 | 采集期间用于新鲜度检查；未作为原始source sequence写入JointState |
| MCAP sequence | writer 每Topic从0递增 | 标量 | MCAP uint32 | 无 | 每写一次 | MCAP message envelope |

Follower `effort` 没有做夹爪力→N转换，也不是外置六维力传感器测量。协议力矩标定准确度、传动侧等效力矩精度未验证；不能将最后一维与归一化 position 作同单位运算。

## 实际 Topic

| Role | Topic | Schema | 语义 |
|---|---|---|---|
| left_leader | `/robot/arm_left_leader/joint_states` | sensor_msgs/msg/JointState | 左Leader位置/手柄输入；导数空 |
| right_leader | `/robot/arm_right_leader/joint_states` | 同上 | 右Leader位置/手柄输入；导数空 |
| left_follower | `/robot/arm_left_follow/joint_states` | 同上 | 左Follower反馈，三数组均7维 |
| right_follower | `/robot/arm_right_follow/joint_states` | 同上 | 右Follower反馈，三数组均7维 |

四路存在于真实双臂样本。单臂样本只有选中侧 Leader、Follower 两路，没有补零的另一侧状态；实际数量和统计见 [数据审计](08_yam_data_audit.md)。Topic 的语义已经沿服务端 `_publish_telemetry`、映射器、writer 和真实解码结果交叉核验。

## 没有进入当前正式 Episode 的字段

没有独立原始 encoder count、末端位姿、数字按钮、motor status/error、MOS/rotor温度、所需补偿力矩或每条 CAN 目标命令 Topic。SDK 的 observations 与独立 `i2rt/utils/recording.py` 能记录比采集 Episode 更多的信息，不能据此宣称当前数据集也保存了这些字段。独立 SDK recorder 有 `required_torque`、温度等字段，是另一条记录路径。

夹爪位置、速度和电机力矩拼在 Follower JointState 末维，没有独立 Gripper Topic。`Gripper State` 的开/闭离散标签、接触真值不在当前文件中。

## 配置频率、程序循环与数据频率

capture mode 解析器固定 `state_save_freq=180`。采样线程使用 monotonic deadline，超时后重置 deadline，不补写理想格点。读取多个服务使用线程池，完成后生成本批共用 `capture_time_ns`；source Header 各自保留。queued writer 保留该时间，实际写盘时刻并未另记。

遥测 worker 的2 ms sleep和CAN带宽估算250 Hz不是MCAP真实采集频率。重复导出序号、源时间倒退、过老和采样间隙会受运行期检查，但源时间是**读取缓存后发布的主机时间**，不是每个关节反馈帧的硬件采样时间。即使 Header 无重复，也不能证明底层每一组物理测量都更新了。

审计分别计算 log/header 相邻差、`(N-1)/(last-first)`、正间隔中位数的倒数频率、P95、标准差、重复和倒退。无消息被写入的时间段不跨 episode 拼接；source sequence 未保存，所以精确未记录源样本数和真正控制发送率为待确认。字段逐维范围、非有限值计数和维度分布保存在 `audit/data/public_summary.json` 与 `field_ranges.csv`；原始逐Episode解码明细仅在本地`audit/data/episodes.jsonl`保留。

## 导出范围限制

`scripts/convert_episode_to_lerobot_v21.py` 要求三相机、四状态流和 `source_timeline.json`，视频重采样到30 FPS，状态最近邻对齐；observation 为右Follower[7]+左Follower[7]，action为右Leader[7]+左Leader[7]。它不直接支持当前单臂五Topic文件。公开包保留此脚本并明确边界，不补造另一侧数据。
