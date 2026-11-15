# Offline test entry point

Run `bash scripts/test_offline.sh`. Tests remain beside their original project to preserve fixture paths. They use mocks, synthetic MCAP/images, MuJoCo model assembly/simulation and temporary directories. Portal integration tests bind ephemeral loopback ports; they do not connect to robot services. The audited selection passed 461 + 43 + 183 = 687 tests with existing third-party dependencies.
