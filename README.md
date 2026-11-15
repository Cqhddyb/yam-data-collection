# YAM Data Collection Engineering

An audited local source release for YAM leader–follower teleoperation, robot
telemetry, RealSense color acquisition, and synchronized MCAP/MP4 episodes.
It includes the locally modified I2RT SDK required by the recorder, reproducible
offline checks, and a detailed engineering handover.

[中文](README_zh.md) · [Architecture](docs/01_yam_data_collection_architecture.md) ·
[Reproduction guide](docs/11_yam_reproduction_guide.md) ·
[Measured data audit](docs/08_yam_data_audit.md) ·
[Engineering audit](docs/00_engineering_audit_report.md)

## 1. Project Overview

The full configuration uses two YAM leaders, two YAM followers, and three color
cameras. A separate single-arm entry point records one selected leader/follower
pair with its wrist camera, optionally adding the overhead camera. The release
preserves the original project boundaries rather than rewriting the control stack.

The 2026-11-15 audit read **3,400 MCAP episodes** and headers from **3,402 associated
H.264 videos**, totaling **2,115,579 camera frames** and **35,308.61 seconds** of
robot-recorded time. This duration includes retained arming samples. Twelve videos
were fully decoded for timestamp validation. Raw recordings are excluded.

## 2. Key Features

- Explicit left/right selection and separate single-arm validation.
- Seven-value joint/gripper telemetry; follower position, velocity and effort.
- Independent camera workers, bounded queues and asynchronous encoding.
- ROS2 CDR MCAP schemas without a running ROS graph.
- Source timestamp video timelines and frame-number diagnostics.
- Verified episode publication, dataset locks and optional wrist preview.
- Explicitly enabled preprogrammed moves and paired prepositioning.

## 3. Hardware Platform

Live capture requires real YAM hardware and a compatible leader device. The audited
setup uses YAM v1 configurations, linear_4310 follower grippers, and YAM teaching
handles. It uses gs_usb-compatible SocketCAN adapters (USB VID/PID `1d50:606f`);
an exact commercial adapter model was not established. The hardware paths cannot
be replaced by arbitrary serial-CAN adapters without additional integration.

Offline file analysis and synthetic tests require no robot, CAN adapter or camera.
MuJoCo-based and controller-mock tests still require the SDK's Python dependencies.

## 4. YAM Robot Architecture

Each arm has six rotary joints. The follower adds one actuated gripper; the leader
adds a passive teaching-handle input to its recorded seven-value vector. Stock
YAM configuration assigns DM4340 motors to joints 1–3 and DM4310 to joints 4–6;
the linear follower gripper also uses DM4310. See the [hardware/control analysis](docs/02_yam_robot_and_leader_follower.md).

## 5. Leader–Follower Teleoperation

The teaching handle's button 0 toggles synchronization. Each side maps its own
leader positions to its follower; no left/right mirroring or simultaneous bimanual
hardware trigger is implemented. The host launcher sets bilateral Kp to zero.
Leader `position[6]` is a mapped gripper input, while follower `position[6]` is
normalized measured gripper position. Leader derivatives are intentionally empty.

## 6. CAN Communication

The host configuration is **1 Mbps**, `restart-ms 0`, `txqueuelen 2000`, and
`ERROR-ACTIVE` before motor startup. Persistent roles are `can_leader_l`,
`can_follower_l`, `can_leader_r`, and `can_follower_r`. The SDK uses transaction-based
standard CAN frames and cached feedback. Its 250 Hz constant is a bandwidth/control
reference, not evidence of a strictly scheduled or measured 250 Hz bus loop.

## 7. Robot State Collection

The configured recorder rate is **180 Hz**. Each follower JointState carries three
float64 arrays of length seven; each leader carries position[7] and empty
velocity/effort. Units are six arm positions in rad, six velocities in rad/s,
normalized gripper position and velocity, and motor-reported torque in Nm.
End-effector pose, raw encoder counts, motor temperatures and executed command
packets are not recorded in the normal episode contract.

## 8. Multi-Camera Acquisition

Each enabled RealSense device has its own color pipeline, native frame queue and
reader thread. RealSense SDK **2.56.x** is required for live camera access. The
audited devices used pyrealsense2 **2.56.5.9235**. No depth stream is enabled.
Other camera vendors require a backend and qualification work.

## 9. Camera Configuration

| Camera | Model / role | 60 Hz profile | 30 Hz profile | Input |
|---|---|---|---|---|
| cam_high | D435 / overhead | 960×540 @ 60 | 1280×720 @ 30 | YUYV |
| cam_left | D405 / left wrist | 848×480 @ 60 | 1280×720 @ 30 | YUYV |
| cam_right | D405 / right wrist | 848×480 @ 60 | 1280×720 @ 30 | YUYV |

The single available bimanual episode measured approximately **59.530 / 59.895 /
59.853 FPS** for overhead / left / right. Larger wrist-only groups are reported
separately in the data audit. Device identities are placeholders. Historical
intrinsics remain as format references and require verification or replacement
for the target camera and profile.

## 10. MCAP Recording

Topics use `sensor_msgs/msg/JointState`, `sensor_msgs/msg/CameraInfo`, and
`std_msgs/msg/String` serialized as CDR. A bimanual episode has four joint topics,
three CameraInfo topics, `/task_info`, and `/diagnostics/frame_timing`.
A single-arm wrist episode has five topics. Despite their image-like names,
the camera topics contain **CameraInfo, not image pixels**; pixels are in MP4.

## 11. Video Encoding

The current host path uses CPU **libx264, CRF 23, ultrafast, one thread per encoder**,
`avc1`, and `yuv420p`. Frames enter bounded queues and are encoded during capture.
Source PTS are applied at close, normally by packet remux. Legacy configurations
can transcode to HEVC after capture; those are a distinct workflow. No explicit
GOP size is configured.

## 12. Timestamp Management

Robot Header/publish time is the host time when a cached telemetry snapshot is
exported. Robot MCAP log time is the recorder's sampling time, retained across
writer queueing. Camera timestamps prefer RealSense global time and retain host
delivery time separately. These are not per-motor hardware timestamps or proof of
simultaneous exposure.

## 13. Data Synchronization

CameraInfo and video share the accepted-frame order. `source_timeline.json`
defines a shared Unix origin and microsecond video PTS mapping. The sampled
videos had constant mux start offsets of −972 to −10 µs and matching relative
source timelines. Consumers should read actual PTS and account for that offset.

Current `yam_60hz` accepts 59.0–60.5 average FPS, ≤69 ms camera gaps and ≤17 ms
temporal matching errors. `yam_60hz_strict` is separate. Passing these gates does
not establish zero source-frame loss: the audited files contain **1,414 missing
device frame numbers** and repeated robot source timestamps.

## 14. Episode Management

Space starts a roughly two-second arming check; wait for `Recording episode`
before teleoperating. Enter saves, then Enter keeps the reviewed episode. `q`
discards/deletes an episode; Ctrl+C exits from idle. Failed terminal-capture
episodes may be automatically deleted by the original implementation. Ctrl+C is
not a hardware emergency stop. There is no mid-episode pause/resume guarantee.

## 15. Dataset Structure

```text
dataset/
├── record_config.yaml
├── 0/
│   ├── YYYY-MM-DD_HH-MM.mcap
│   └── videos/
│       ├── cam_left.mp4             # selected wrist in this example
│       ├── cam_high.mp4             # only if enabled
│       └── source_timeline.json
└── runtime_diagnostics/             # session diagnostics when enabled
```

Configuration snapshots may contain real device identities after local setup;
keep recordings and host-specific configuration private when publishing.

## 16. Repository Structure

```text
release/
├── README.md / README_zh.md
├── requirements.txt / requirements-dev.txt
├── configs/                         # sanitized configuration templates
├── src/
│   ├── i2rt/                        # local SDK snapshot + MIT license/models
│   ├── yam_data_collection/          # base/queued recorder, QA, export, tests
│   ├── yam_single_arm_collection/    # selected-side capture and motion
│   ├── deploy/                      # CAN installation templates
│   └── single_arm_transfer/          # CPU adaptation / single-side CAN setup
├── scripts/                         # setup, offline tests, read-only audit
├── tests/                           # test entry-point notes
└── docs/                            # engineering documentation and aggregates
```

[Core file catalog](docs/12_core_code_inventory.md) · `SOURCE_MANIFEST.json` records
source/release hashes and every sanitization. Source layout is preserved so
relative configuration, model and launcher paths work together.

## 17. Environment Requirements

Linux x86_64, Python **3.11** recommended; FFmpeg/ffprobe with libx264; iproute2,
can-utils and USB permissions for live use; Tk/display only for optional preview.
Two independent uv environments retain their original lockfiles. A running ROS
installation is not required by the recorder, although the original launcher
retains a ROS Humble library-path entry from its deployment host.

## 18. Installation

```bash
# From the release directory; install uv separately if unavailable.
sudo apt-get install ffmpeg can-utils iproute2 usbutils
bash scripts/setup_env.sh
bash scripts/test_offline.sh
```

Setup rebuilds dependencies and does not configure CAN or start motors. The SDK
may compile ruckig 0.15.3 and require a C++ toolchain. For offline recorder work:

```bash
cd src/yam_data_collection
uv sync --frozen --python 3.11 --extra dev --extra timeline
```

## 19. Hardware Setup

Follow the [reproduction guide](docs/11_yam_reproduction_guide.md) to replace serial
placeholders consistently in the launchers and udev/installer templates, remeasure
camera intrinsics and teaching-handle endpoints, and adapt CPU affinity. Do not
reuse the audited host's CPU IDs or calibration as target-host validation.

## 20. Robot Initialization

After physical checks and CAN commissioning, use separate terminals:

```bash
bash yam_single_arm.sh follower --side left --confirm-motion
bash yam_single_arm.sh leader --side left --confirm-motion
```

These commands can enable motors and calibrate the gripper. Optional paired motion
requires `leader-motion` instead of `leader`; it defaults to a fresh current start,
with a 1 rad/s arm-plan limit. Plans and `MOVE`/`PREPOSITION` confirmations are
explicit. No collision planner is provided.

## 21. Camera Initialization

```bash
src/yam_data_collection/.venv/bin/yam-data-collect discover-cameras
bash yam_single_arm.sh camera-test --side left --cameras wrist --duration 180 --no-video
bash yam_single_arm.sh camera-test --side left --cameras wrist --duration 60
```

Run camera tests separately from capture. Qualify each selected combination and
mode; the earlier 30 Hz right-wrist test failed and has no documented closure.

## 22. Data Collection

Replace task metadata first, then use another terminal:

```bash
bash yam_single_arm.sh check --side left --cameras wrist
bash yam_single_arm.sh capture --side left --cameras wrist --preview
```

Use `right` for the other pair or `wrist-top` to include overhead. For a full
bimanual run, commission all four roles and use `bash yam.sh capture --preview`.
The full guide documents role startup order and current/recorded-start differences.

## 23. Data Validation

```bash
bash yam_single_arm.sh validate /path/to/episode.mcap \
  --side left --cameras wrist --mode 60hz \
  --config /path/to/record_config.yaml --json
bash summarize_dataset_duration.sh /path/to/dataset
```

The bimanual validator and 14-value LeRobot exporter require the full four-state,
three-camera contract. They are not direct single-arm exporters. `scripts/audit_dataset.py`
recomputes the census for the original data-root layout and writes reports outside
input data directories. Optional Dream validation needs its separately installed
SDK and was not tested here.

## 24. Troubleshooting

Check CAN state in addition to link UP; distinguish source FPS from encoder
throughput; inspect device frame numbers, host/capture timing and encoder queue
diagnostics. Camera startup errors retain device-role context. See the
[incident review](docs/09_yam_debugging.md) for evidence, fixes and unresolved causes.
No general recovery of power-loss-damaged MCAP files is promised.

## 25. Limitations

The workspace contains no usable Git history, so commit-level provenance is
unavailable. All audited single-arm files carry a smoke-test task ID; actual task
labels require review. There is no hardware exposure synchronization, recorded
CAN-command stream, independent command-execution timestamp, or guaranteed
power-loss atomic commit. Early D435 drops and a reported right-CAN fault have
incomplete root-cause closure. Offline regression is not target-hardware acceptance.

The isolated release passed **687 offline tests**: 461 collector/single-arm/deployment,
43 mock controller, and 183 model assembly/simulation tests, using existing third-party dependencies. Clean installation
on a new host and live hardware qualification remain target-host work.

## 26. Dependencies and Acknowledgements

Built on I2RT Robotics' SDK, RealSense, python-can, Portal, MCAP ROS2 support,
NumPy/PyYAML, MuJoCo, PyAV and FFmpeg. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
and the included SDK MIT license. The recorder's original workspace provided no
separate project license; the owner must choose one before publication.
This delivery performs no GitHub upload and contains no raw dataset or credentials.
