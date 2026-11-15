# Packaging exclusions and transformations

The source selection is recorded file by file in SOURCE_MANIFEST.json. EXCLUDED_FILES.csv records the remaining scanned inputs by file or anonymous data/session pattern. Raw episode paths are grouped to avoid publishing operator names and local dataset locations. Counts refer to the original audit inventory, not to virtual environments or hidden credential directories.

Selected source: 246 files / 78,479,153 bytes. Excluded scanned inputs: 10,888 files / 24,508,595,655 bytes.

| Category | Treatment |
|---|---|
| Raw MCAP, MP4, dataset images and complete recordings | Excluded, including 3,400 MCAP and 3,416 MP4 files; anonymous aggregate tables only |
| Runtime JSONL, deleted-episode ledgers and camera-test outputs | Raw files excluded; evidence is summarized without operator paths |
| Historical handoff copies and deployment archives | Excluded to preserve one current source snapshot; hash differences retained |
| Camera/CAN device identities | Replaced consistently with role placeholders in launchers, udev rules, installers and configs |
| Local task metadata and measured right-handle endpoints | Replaced with REPLACE_* placeholders; matching fixture assertion updated |
| Device-bound calibration files | Excluded; historical numeric intrinsics in templates require verification/recalibration |
| Firmware and Flow Base code | Excluded from the selected SDK snapshot |
| Environments, caches, .git, .aws, .codex, .agents, .claude | Not copied or traversed for contents; no credentials or Git objects are packaged |
| Model weights / training data / training logs | Not selected; SDK geometry meshes are runtime assets and retained under SDK provenance |
| Third-party dependencies | Installed separately from lockfiles; only the included MIT SDK snapshot is vendored |

19 source files have sanitization transformations. Business control/recording logic is preserved. The source/release hashes identify every changed copy; no original business files were edited. First-party project licensing remains an owner decision; see THIRD_PARTY_NOTICES.md.
