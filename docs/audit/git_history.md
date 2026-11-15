# Git只读审计与快照演化

2026-11-15在工作区执行`git status --short`和`git log -30 --date=iso-strict`，均返回“不是git仓库”。根目录`.git/`为空；`i2rt`和`yam_data_collection`没有自己的.git。只读检查两个tar.gz的成员，11月7日包557项、11月12日包388项，均无.git对象。

因此无法给出commit作者、提交时间、分支、修改记录、git blame或历史Bug引入提交。不能将文件mtime、参考对话的时间或SDK版本号当成commit证据。本次没有git add/commit/push、没有修改任何Git对象。

替代证据是`handoff/yam_single_arm_transfer_20261112/`与当前文件逐项SHA-256比较：14个匹配路径内容有差异，详见`snapshot_differences.json`。差异包含：

- 单臂CLI从“显式--from-current”变成默认读取实时起点，并新增--from-recorded。
- 单臂客户端和Leader服务端预编程关节限速从0.25改为1rad/s。
- 相机startup首帧错误处理改为保留camera role和底层异常，避免同长Future timeout掩盖错误。
- 时长报告、文档、验证元数据及相应测试更新。

这些差异只能证明两个可访问快照不同，不能证明内部修改的先后提交关系。新增单臂入口、腕部10Hz预览和配对就位的设计历史还参考了已存在README、verification.json与用户指定对话；对话中的现场观察属于二级证据，未保存原始日志处另行标明。

更早部署说明表明存在HEVC最终转码、D405640×480候选、严格60Hz质量档位、原始i2rt独立MCAP recorder。当前host选择H.264-final、D405848×480和source时间线；仅凭共存文件不能断言旧路径已经从所有用户流程中废弃。
