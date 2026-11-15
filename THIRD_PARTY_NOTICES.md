# Dependencies and licensing

`src/i2rt/` is a selected local snapshot of I2RT Robotics' SDK (metadata version
1.1.2), including local telemetry, control-health and preprogrammed-motion changes.
Its included MIT license is at [src/i2rt/LICENSE](src/i2rt/LICENSE). Robot model
assets are retained with this SDK; model variants present in the SDK are not
claims about hardware used by this project. Flow Base code and firmware binaries
are excluded. No Git commit identity is available in this workspace.

Other runtime dependencies are installed separately. They are not vendored:
python-can 4.5.0, Portal 3.7.3, MuJoCo, NumPy, PyYAML, MCAP and mcap-ros2-support,
PyAV 16, pyrealsense2 2.56.x and FFmpeg with libx264 (and libx265 only for legacy
HEVC workflows). Package versions and transitive source distributions are in the
two retained `uv.lock` files. FFmpeg build features and applicable binary licenses
depend on the installation. ROS2 message definitions are embedded by the original
recorder; a running ROS installation is not required for file recording.

The local recorder and single-arm extension had no separate license file in the
audited workspace. This delivery does not assign a new license or claim that the
SDK MIT license covers those project files. The project owner should choose a
license before publishing. Dream dataset validation is an optional script requiring
a separately installed Dream SDK; no Dream SDK code, credentials or datasets are
included, and that optional integration was not exercised in this audit.
