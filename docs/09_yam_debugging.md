# 历史故障复盘与证据强度

本章区分日志/快照已证实的历史问题、参考对话记载的问题、仅从代码防御分支发现的潜在问题。没有原始堆栈或结果时明确留空，不把测试覆盖当作现场事故。

## B01：CAN 服务拒绝自动 Bus-Off restart（历史部署记录确认）

1. 现象：接口成功重命名后，CAN服务初始化失败。
2. 触发：内核`6.8.0-138-generic`的gs_usb设置非零`restart-ms`。
3. 初判：名称或总线状态异常需核对；记录明确否定“已经Bus-Off”的推断。
4. 排查：看systemd状态和journal，定位发生在参数设置阶段。
5. 命令：`systemctl status yam-can@INTERFACE.service`、`journalctl -b -u yam-can@INTERFACE.service`、`ip -details link show INTERFACE`。
6. 关键日志：`Error: Device doesn't support restart from Bus Off.`（`DEPLOYMENT_HOST.md`保留，原始journal未随包保存）。
7. 根因：该驱动不支持所请求的自动恢复参数。
8. 修复：`deploy/yam-can@.service`使用`restart-ms 0`，`yam.sh`校验同步修改。
9. 验证：历史部署记录四接口UP/ERROR-ACTIVE，四路7维遥测ready；本次复用纯字典测试校验该门禁，不重新初始化硬件。
10. 经验：命名成功、link UP和CAN健康是三个不同层次。
11. 预防：固定driver兼容参数，检查CAN state，不将自动恢复当作必需功能。

## B02：顶部 D435 平均帧率略低、偶发帧号缺口（日志与报告确认，最终根因未确认）

1. 现象：早期60Hz测试顶部约59.45–59.49Hz，低于当时59.5Hz边界；编码帧数仍与接受帧数一致。
2. 触发：三相机30/180秒与顶部单独30秒均能复现。
3. 初判：编码竞争、USB带宽、曝光或设备源帧率均曾考虑。
4. 排查：分别不写视频、只开顶部、读device frame number/actual_fps、临时约10ms固定曝光对照。无视频和隔离测试仍出现，不能归结为“仅编码过慢”。
5. 命令：`bash yam.sh camera-test --duration 180 --no-video`、独立顶部camera-test配置；具体SDK诊断脚本未完整留存。
6. 日志：`deploy/camera_test_180s_timing.log`：顶部59.45Hz、最大33.6ms；`camera_high_diagnostic.json`记录1488帧、1个缺号；曝光对照1189帧、2个缺号、59.426Hz。
7. 根因：**待确认**。global_time实际元数据约59.52Hz，仅能说明源时序接近设备实际节奏。
8. 处理：恢复曝光设置；保留native queue、global-time、帧号和编码队列诊断。不能证明这些代码都是本次事件引入，缺少Git链。
9. 验证：11月7日20:28的180秒时序和20:31的60秒视频三路通过；顶部约59.529Hz。后续通过不构成根因闭环。
10. 经验：source FPS、host交付抖动、容器FPS、编码帧数需要分开测。
11. 预防：换Hub/线缆/负载后复验；不能以调整质量下界来证明物理丢帧已修复。

## B03：30Hz 右腕约100ms间隔（报告确认，未见后续闭环）

现象/触发：11月7日19:18三相机30Hz 60秒带视频测试；右腕29.678Hz、最大100.005ms。初判与排查：视频验证通过但camera timing失败，说明“编码帧数一致”不等于“源无缺口”。命令为`YAM_MODE=30hz bash yam.sh camera-test --duration 60`。关键证据为`deploy/camera_test_60s_30hz_video.log`及对应report.json。根因、已实施的硬件修复及重新通过结果均待确认；不能把代码中30Hz补帧派生工具当作此事件的真实源帧恢复。预防为独立验收30Hz模式、检查设备帧号并保留原始证据。

## B04：更换左腕相机后沿用旧内参（部署记录确认）

现象/触发：本机相机身份与原YAML所对应的左腕设备不同。初判为配置迁移不完整。排查对比SDK枚举身份、stream profile和标定文件来源（`deploy/hardware_inventory.json`）；完整当时枚举命令未保留。根因是把旧设备标定带到替换设备。修复为host配置使用实际左D405的模式匹配出厂内参，并保留`inverse_brown_conrady`。验证在本次真实MCAP CameraInfo解码中再次确认此模型；未重新执行棋盘标定。经验和预防：标定必须绑定设备与stream profile，替换设备后重做配置核验，禁止只改畸变模型名称。

## B05：相机首帧错误被空 TimeoutError 掩盖（当前/11月12日快照差分确认）

现象：首帧失败时上层可能只看到空超时，缺少camera role。触发为camera.read自己的1秒超时与Future相同超时竞争。初判/排查证据来自代码差分与`test_collector_safety.py`，没有足够现场堆栈复原某一次事故。根因是两个同长超时层掩盖后端异常，executor退出仍会等待read。修复为`_read_camera_startup_frames()`依赖后端有界read，用`result()`取原异常，附加role及异常类型后串联raise；三次启动重试仍在。验证为打包后的离线回归。预防：保留上下文和cause；如果替代backend违反有界read契约，该helper仍可能等待，不能泛称所有backend绝不会卡住。

## B06：右 Follower ERROR-PASSIVE 与 USB Hub 重连（参考对话记载）

参考“新增机械臂定点移动与采集前主从臂同步就位”对话记载：接口UP但state为ERROR-PASSIVE，启动器拒绝动作，错误计数增加；kernel有Hub断开重连。排查命令包括`ip -details -statistics link show can_follower_r`与journal。根因只能定位到通信异常，供电/接线/终端电阻/Hub是待排查项，不是已证实的单一硬件根因。对话建议先停止相关控制和采集、检查连接后重启选中CAN服务；未见持久原始kernel日志及最终修复验收，故不写“已修复”。启动器现有ERROR-ACTIVE门禁是保护措施。本次未接触这些服务。

## B07：read-poses 的 NumPy 按钮数组序列化（参考对话与源码/测试支持）

对话记录只读姿态查询修复后通过32项检查；现有`test_read_pair_poses_serializes_numpy_button_states`与`read_pair_poses()`显式转list对应。触发是Portal返回的button_state可能为ndarray，不能直接JSON编码。完整原始TypeError堆栈未保存；不能虚构行号和报错原文。当前实现统一转换后输出JSON，本次离线测试验证；无需控制动作。预防为在RPC→JSON边界规范化容器类型。

## B08：空闲会话中的腕部取帧超时（本次读取原始JSONL确认，根因未闭环）

1. 现象：后台相机worker进入failed，采集器保留background_error。
2. 触发条件：两份会话分别观察到左腕、右腕`Frame did not arrive in time!`；第一条错误快照均为`state=idle, episode_id=null`。日志没有证明当时正在录制某个Episode。
3. 初判：相机没有在read的有界等待内交付帧；USB/SDK/硬件原因仍需现场证据。
4. 排查过程：本次只读扫描56份runtime JSONL，共1,010,563行；递归提取background_error，按每文件状态变化统计，核对首次出错时的state和episode_id。
5. 使用的命令：`src/yam_data_collection/.venv/bin/python scripts/audit_runtime_logs.py --root src --out /tmp/yam-runtime-review.json`（扫描本地保留日志，不启动采集）；随后读取输出`docs/audit/runtime_log_statistics.json`定位首例。
6. 关键日志：`camera cam_left failed: camera cam_left did not produce a frame: Frame did not arrive in time!`；右腕同结构、role为cam_right。左腕首例位于双臂会话`capture-1794055339307125856.jsonl:1145`，右腕首例位于单臂会话`capture-1794579702086878856.jsonl:65882`。真实目录映射仅在本地审计结果保留。
7. 根本原因：待确认；没有足够的同一时间USB/kernel/供电记录，不能把其他对话的Hub重连自动关联为本事件根因。
8. 修复代码或配置：未见可关联的现场修复提交或明确配置闭环。当前源码会传播相机错误并阻止继续产生有效Episode；这属于保护行为。
9. 验证过程：读取日志确认错误存在；本次未热插拔、重启相机或运行真实采集，因此不声明硬件修复通过。
10. 经验：87条左腕、118,953条右腕快照保存了同一类持续错误，按状态进入仅各1次观察；不能当作119,040次独立事故或失败Episode。
11. 预防：现场复现时同时保存USB拓扑、kernel事件、相机角色、占用进程和异常前后source/host时序；在恢复后重新完成该组合的camera-test和端到端试采。

## 已观察到的编码停顿与删除账本（不作为已定位根因的Bug）

运行日志记录`max_pipe_write_ms=451.593819`、`max_queue_depth=10`，软件`dropped_frames`峰值为0。它证明有编码pipe阻塞和队列缓冲，不能单独证明阻塞来自磁盘、CPU或ffmpeg，也不否定MCAP中1,414个设备帧号缺口。未保留逐帧落盘时间，无法精确重建端到端写入延迟。

7份`deleted_episode_durations.txt`含166条历史删除：37条validation_failed、129条user_deleted。账本不保存所有失败检查详情，且功能引入前未知；不能把37条统称相机故障或计算全历史失败率。当前数据审计只统计现存成品，未删除任何输入文件。

## 正常门禁与潜在风险

计划起点偏差、被动Leader手柄不能自动到0.5、leader遥测未启动、确认词`MOV`少了`E`均在参考对话出现；这些是输入/运行条件被正确拦截，不是已证实的算法Bug。当前默认实时起点和1rad/s限速是功能演进，不能写成“校准Bug修复”。

Portal Future timeout后重建client、latest-wins队列、后台健康检查、队列满拒绝、发布rollback等防御有代码和合成测试；没有完整事件日志时只说明机制。磁盘满、MCAP schema mismatch、断电恢复失败等属于待测试故障场景，不列为已经发生的历史Bug。
