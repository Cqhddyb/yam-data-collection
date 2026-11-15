# YAM 数据采集工程文档

审计日期：2026-11-15。以当前源文件、配置、真实数据、部署日志和可用历史快照为依据。硬件未启动；原始业务文件未修改。缺少Git对象的限制、证据强度和未确认事项均在报告中明确标注。

| 文档 | 内容 |
|---|---|
| [00 工程审计总报告](00_engineering_audit_report.md) | 交付、架构、实测结论、打包、验证和未确认事项 |
| [01 系统架构](01_yam_data_collection_architecture.md) | 硬件连接、进程/线程、版本边界和调用图 |
| [02 机器人与主从控制](02_yam_robot_and_leader_follower.md) | YAM、CAN、左右映射、夹爪、预编程配对就位 |
| [03 机器人状态](03_yam_robot_state_collection.md) | 状态/输入/命令/Action语义、字段与单位 |
| [04 相机采集](04_yam_camera_collection.md) | D435/D405、profile、缓存、线程和内参 |
| [05 MCAP与视频](05_yam_mcap_and_video_recording.md) | Topic/Schema、CDR、H.264/旧HEVC、发布契约 |
| [06 时间戳与同步](06_yam_timestamp_and_synchronization.md) | 时钟、frame_timing、质量门限、统计口径 |
| [07 Episode与数据组织](07_yam_episode_and_dataset_structure.md) | 生命周期、真实目录、元数据和失败恢复边界 |
| [08 真实数据审计](08_yam_data_audit.md) | 全量3,400 MCAP与3,402关联视频统计、CSV附录 |
| [09 故障复盘](09_yam_debugging.md) | 已证实故障、排查/修复证据及尚未闭环项 |
| [10 技术决策](10_yam_technical_decisions.md) | 真实设计依据、工程推断、权衡和局限 |
| [11 完整复现指南](11_yam_reproduction_guide.md) | 安装、配置、CAN、控制器、相机、采集与离线验收 |
| [12 核心代码清单](12_core_code_inventory.md) | 输入/输出、依赖、调用者、数据契约和模块图 |
| [历史审计](audit/git_history.md) | 无可用Git历史、14个快照差异及证据边界 |

原工作区`docs/audit/`还保留真实路径的详细报告，用于本地追溯。独立`release/`只包含脱敏源码、模板、文档和匿名统计，不包含MCAP、视频、采集者目录映射、虚拟环境或运行日志。公开包的第一方代码尚无新增许可证，第三方SDK保留MIT许可，见其THIRD_PARTY_NOTICES。
