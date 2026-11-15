#!/usr/bin/env python3
"""Read-only MCAP census and video-header audit; never imports robot control.

Run with the collector's Python 3.11 environment (mcap-ros2-support, numpy, av).
Output paths must be outside source data directories. Videos are decoded only
for the deterministic first episode in each dataset group, unless --decode-all.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import av
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


def describe(values):
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return {"count": 0}
    return {"count": len(a), "min": float(a.min()), "mean": float(a.mean()),
            "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "max": float(a.max()), "std": float(a.std())}


def timing(t):
    a = np.asarray(t, dtype=np.int64)
    d = np.diff(a)
    positive = d[d > 0]
    return {"messages": len(a), "span_s": int(a[-1] - a[0]) / 1e9 if len(a) else 0,
            "frequency_hz": (len(a) - 1) * 1e9 / int(a[-1] - a[0])
            if len(a) > 1 and a[-1] > a[0] else None,
            "median_frequency_hz": 1e9 / float(np.median(positive)) if len(positive) else None,
            "interval_ms": describe(d / 1e6),
            "duplicate_adjacent": int(np.count_nonzero(d == 0)),
            "duplicate_total": len(a) - len(np.unique(a)),
            "backwards": int(np.count_nonzero(d < 0))}


def nearest(a, b):
    """Signed nearest time difference a-b; sorted search, no interpolation."""
    a = np.asarray(a, dtype=np.int64)
    b = np.sort(np.asarray(b, dtype=np.int64))
    if not len(a) or not len(b):
        return np.array([], dtype=np.int64)
    i = np.searchsorted(b, a)
    lo = a - b[np.maximum(0, i - 1)]
    hi = a - b[np.minimum(len(b) - 1, i)]
    return np.where(np.abs(lo) <= np.abs(hi), lo, hi)


def audit_mcap(path):
    logs, headers, publish, seqs = (defaultdict(list) for _ in range(4))
    fields = defaultdict(lambda: defaultdict(list))
    dimensions = defaultdict(lambda: defaultdict(Counter))
    schemas, diag, task = {}, {}, {}
    header_publish_mismatch = Counter()
    with path.open("rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()], validate_crcs=True)
        summary = reader.get_summary()
        for schema, ch, msg, decoded in reader.iter_decoded_messages(log_time_order=False):
            topic = ch.topic
            schemas[topic] = {"schema": schema.name, "schema_encoding": schema.encoding,
                              "message_encoding": ch.message_encoding}
            logs[topic].append(msg.log_time)
            publish[topic].append(msg.publish_time)
            seqs[topic].append(msg.sequence)
            if hasattr(decoded, "header"):
                stamp = decoded.header.stamp
                ts = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
                headers[topic].append(ts)
                header_publish_mismatch[topic] += ts != msg.publish_time
            if topic.endswith("joint_states"):
                for key in ("position", "velocity", "effort"):
                    v = list(getattr(decoded, key))
                    dimensions[topic][key][len(v)] += 1
                    fields[topic][key].append(v)
            elif topic == "/diagnostics/frame_timing":
                payload = json.loads(decoded.data)
                diag[payload["role"]] = payload
            elif topic == "/task_info":
                payload = json.loads(decoded.data)
                task = {k: payload.get(k) for k in ("task_id", "task_name")}
                task["layout"] = payload.get("collection_layout", {})
            elif "camera_info" in topic:
                schemas[topic].update(width=int(decoded.width), height=int(decoded.height),
                                      distortion_model=decoded.distortion_model)
    report = {"topics": {}, "diagnostics": {}, "sync": {}, "task": task,
              "summary_message_count": summary.statistics.message_count if summary and summary.statistics else None}
    for topic, stamps in logs.items():
        record = {**schemas[topic], "log_time": timing(stamps),
                  "publish_time": timing(publish[topic]),
                  "log_minus_publish_ms": describe((np.asarray(stamps, dtype=np.int64) -
                                                     np.asarray(publish[topic], dtype=np.int64)) / 1e6),
                  "sequence_skips": sum(max(0, b - a - 1) for a, b in zip(seqs[topic], seqs[topic][1:])),
                  "header_publish_mismatch": header_publish_mismatch[topic]}
        if headers[topic]:
            record["header_time"] = timing(headers[topic])
        if topic in fields:
            record["fields"] = {}
            for key, rows in fields[topic].items():
                dims = dimensions[topic][key]
                entry = {"dimensions": dict(dims), "schema_dtype": "float64", "nonfinite": 0}
                # Preserve unusual variable dimensions instead of coercing malformed rows.
                by_dim = defaultdict(list)
                for row in rows:
                    if row:
                        by_dim[len(row)].append(row)
                entry["ranges_by_dimension"] = {}
                for dim, rs in by_dim.items():
                    a = np.array(rs, dtype=np.float64)
                    entry["nonfinite"] += int(np.count_nonzero(~np.isfinite(a)))
                    entry["ranges_by_dimension"][str(dim)] = {
                        "min": np.nanmin(a, axis=0).tolist(), "max": np.nanmax(a, axis=0).tolist()}
                record["fields"][key] = entry
        report["topics"][topic] = record
    for role, d in diag.items():
        capture = np.array(d["capture_timestamp_ns"], dtype=np.int64)
        host = np.array(d["host_receive_timestamp_ns"], dtype=np.int64)
        nums = d.get("device_frame_number", [])
        usable = [int(x) for x in nums if x is not None]
        drops = sum(max(0, b - a - 1) for a, b in zip(usable, usable[1:]))
        topic = f"/robot/camera/{role}/image_raw/compressed/camera_info"
        report["diagnostics"][role] = {
            "frames": len(capture), "host_minus_capture_ms": describe((host - capture) / 1e6),
            "device_frame_number_available": len(usable), "device_frame_gaps": drops,
            "device_frame_duplicates": sum(a == b for a, b in zip(usable, usable[1:])),
            "device_frame_backwards": sum(b < a for a, b in zip(usable, usable[1:])),
            "capture_timestamp_sources": d.get("capture_timestamp_sources"),
            "device_timestamp_domains": d.get("device_timestamp_domains"),
            "camera_info_timestamps_equal": capture.tolist() == headers.get(topic, [])}
    joints = [t for t in headers if t.endswith("joint_states")]
    cameras = [t for t in headers if "camera_info" in t]
    sync_arrays = {}
    for a, b in itertools.combinations(joints, 2):
        key = f"nearest:{a}|{b}"
        sync_arrays[key] = nearest(headers[a], headers[b])
    for a in cameras:
        for b in joints:
            key = f"nearest:{a}|{b}"
            sync_arrays[key] = nearest(headers[a], headers[b])
    for a, b in itertools.combinations(cameras, 2):
        key = f"nearest:{a}|{b}"
        sync_arrays[key] = nearest(headers[a], headers[b])
    for k, a in sync_arrays.items():
        report["sync"][k] = {"signed_ms": describe(a / 1e6), "absolute_ms": describe(np.abs(a) / 1e6)}
    robot_logs = [t for t in logs if t.endswith("joint_states")]
    first = min(logs[t][0] for t in robot_logs) if robot_logs else min(map(min, logs.values()))
    last = max(logs[t][-1] for t in robot_logs) if robot_logs else max(map(max, logs.values()))
    report["duration_s"] = (last - first) / 1e9
    return report, logs, headers, publish, sync_arrays


def audit_video(path, decode=False, expected_stamps=None, origin=None):
    result = {"bytes": path.stat().st_size, "decoded": False}
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        result.update(codec=s.codec_context.name, width=s.width, height=s.height,
                      frames_header=s.frames, pixel_format=s.codec_context.format.name,
                      average_rate=str(s.average_rate), time_base=str(s.time_base),
                      duration_s=float(s.duration * s.time_base) if s.duration is not None else None)
        if decode:
            times = []
            for frame in c.decode(s):
                times.append(round(float(frame.pts * frame.time_base) * 1e6))
            result.update(decoded=True, frames_decoded=len(times), pts_us=timing(np.array(times, dtype=np.int64)*1000))
            if expected_stamps is not None and origin is not None:
                expected = (np.array(expected_stamps, dtype=np.int64) - int(origin)) // 1000
                result["pts_equal_mcap_source"] = times == expected.tolist()
                if len(times) == len(expected):
                    result["pts_minus_expected_us"] = describe(np.array(times) - expected)
                    offsets = np.array(times, dtype=np.int64) - expected
                    result["relative_pts_match_mcap_source"] = bool(len(offsets) and np.all(offsets == offsets[0]))
                    result["constant_start_offset_us"] = int(offsets[0]) if result["relative_pts_match_mcap_source"] else None
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--decode-all", action="store_true")
    args = ap.parse_args()
    root = args.root.resolve()
    out = args.out.resolve()
    data_roots = [root / "yam_data_collection/data", root / "yam_single_arm_collection/data"]
    if any(out.is_relative_to(p) for p in data_roots):
        raise SystemExit("Output must be outside all input data directories")
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for base in data_roots if base.exists() for p in base.rglob("*.mcap"))
    group_paths = sorted({p.parent.parent for p in paths})
    group_ids = {p: f"S{i+1:02d}" for i, p in enumerate(group_paths)}
    (out / "local_group_mapping.json").write_text(json.dumps({v: str(k.relative_to(root)) for k,v in group_ids.items()}, ensure_ascii=False, indent=2))
    aggregate = defaultdict(lambda: {"episodes": 0, "durations": [], "topics": defaultdict(list), "video_count": 0, "video_frames_header": 0, "bytes": 0, "task_ids": Counter()})
    topic_intervals = defaultdict(list)
    source_intervals = defaultdict(list)
    age_arrays = defaultdict(list)
    sync_all = defaultdict(list)
    errors, video_errors, episodes, videos, diag_total = [], [], [], [], defaultdict(Counter)
    started = time.monotonic()
    with (out / "episodes.jsonl").open("w") as details:
        for index, path in enumerate(paths):
            group = group_ids[path.parent.parent]
            rel = str(path.relative_to(root))
            try:
                r, logs, headers, pub, sync_arrays = audit_mcap(path)
                r.update(path=rel, group=group, mcap_bytes=path.stat().st_size)
                g = aggregate[group]
                decode = args.decode_all or g["episodes"] == 0
                g["episodes"] += 1
                g["durations"].append(r["duration_s"])
                g["bytes"] += path.stat().st_size
                g["task_ids"][str(r["task"].get("task_id"))] += 1
                for topic, tr in r["topics"].items():
                    g["topics"][topic].append(tr)
                    topic_intervals[(group, topic)].append(np.diff(np.array(logs[topic], dtype=np.int64)))
                    if headers[topic]:
                        source_intervals[(group, topic)].append(np.diff(np.array(headers[topic], dtype=np.int64)))
                    age_arrays[(group, topic)].append(np.array(logs[topic], dtype=np.int64)-np.array(pub[topic], dtype=np.int64))
                for k,a in sync_arrays.items():
                    sync_all[(group,k)].append(a)
                for role,d in r["diagnostics"].items():
                    for k in ["frames","device_frame_number_available","device_frame_gaps","device_frame_duplicates","device_frame_backwards"]:
                        diag_total[(group,role)][k] += d[k]
                    diag_total[(group,role)]["camera_info_mismatch_episodes"] += not d["camera_info_timestamps_equal"]
                timeline_path = path.parent / "videos/source_timeline.json"
                timeline = json.loads(timeline_path.read_text()) if timeline_path.exists() else {}
                r["timeline_mode"] = timeline.get("mode")
                r["videos"] = {}
                for video in sorted((path.parent / "videos").glob("*.mp4")):
                    try:
                        topic = f"/robot/camera/{video.stem}/image_raw/compressed/camera_info"
                        vr = audit_video(video, decode, headers.get(topic), timeline.get("origin_timestamp_ns"))
                        vr["mcap_camera_messages"] = len(headers.get(topic, []))
                        vr["header_count_matches_mcap"] = vr["frames_header"] == vr["mcap_camera_messages"]
                        vr.update(group=group, path=str(video.relative_to(root)), role=video.stem)
                        videos.append(vr)
                        r["videos"][video.stem] = vr
                        g["video_count"] += 1
                        g["video_frames_header"] += vr["frames_header"]
                        g["bytes"] += vr["bytes"]
                    except Exception as e:
                        video_errors.append({"path":str(video.relative_to(root)),"error":str(e)})
                details.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
                episodes.append({"group": group, "path":rel, "duration_s":r["duration_s"],
                                 "mcap_bytes":r["mcap_bytes"], "task_id":r["task"].get("task_id"),
                                 "videos":len(r["videos"]), "messages":sum(len(v) for v in logs.values())})
            except Exception as e:
                errors.append({"path":rel,"error":repr(e)})
            if (index+1) % 100 == 0 or index+1 == len(paths):
                print(f"Audited {index+1}/{len(paths)} MCAP, {len(videos)} videos, {time.monotonic()-started:.1f}s",flush=True)
    summary = {"method": "full MCAP decode in file order with CRC validation; video container headers for all episode videos; full video decode for first lexicographic episode per group", "episodes":len(episodes), "mcap_candidates":len(paths), "videos":len(videos), "errors":errors, "video_errors":video_errors, "groups":{}, "diagnostics":{}, "sync":{}, "duration_s":describe([e["duration_s"] for e in episodes]), "total_duration_s":sum(e["duration_s"] for e in episodes)}
    for group,g in aggregate.items():
        result = {k:v for k,v in g.items() if k not in ["durations","topics"]}
        result["duration_s"] = describe(g["durations"])
        result["total_duration_s"] = sum(g["durations"])
        result["topics"] = {}
        for topic,rows in g["topics"].items():
            dt = np.concatenate(topic_intervals[(group,topic)])
            src = np.concatenate(source_intervals[(group,topic)]) if source_intervals[(group,topic)] else np.array([])
            age = np.concatenate(age_arrays[(group,topic)])
            rec = {"episodes":len(rows), "schema":rows[0]["schema"], "messages":sum(x["log_time"]["messages"] for x in rows),
                   "interval_ms":describe(dt/1e6), "header_interval_ms":describe(src/1e6),
                   "frequency_hz_weighted":len(dt)*1e9/int(dt.sum()) if len(dt) and dt.sum()>0 else None,
                   "episode_frequency_hz":describe([x["log_time"]["frequency_hz"] for x in rows if x["log_time"]["frequency_hz"] is not None]),
                   "duplicate_log":sum(x["log_time"]["duplicate_total"] for x in rows),
                   "backwards_log":sum(x["log_time"]["backwards"] for x in rows),
                   "duplicate_header":sum(x.get("header_time",{}).get("duplicate_total",0) for x in rows),
                   "backwards_header":sum(x.get("header_time",{}).get("backwards",0) for x in rows),
                   "sequence_skips":sum(x["sequence_skips"] for x in rows),
                   "log_minus_publish_ms":describe(age/1e6), "fields":{}}
            for field in ["position","velocity","effort"]:
                dims=Counter(); ranges=defaultdict(list);nonfinite=0
                for row in rows:
                    entry=row.get("fields",{}).get(field)
                    if entry:
                        dims.update(entry["dimensions"]);nonfinite+=entry["nonfinite"]
                        for dim,v in entry["ranges_by_dimension"].items():ranges[dim].append(v)
                if dims:
                    rec["fields"][field]={"dimensions":dict(dims), "nonfinite":nonfinite, "ranges_by_dimension":{dim:{"min":np.min([v["min"] for v in vs],axis=0).tolist(),"max":np.max([v["max"] for v in vs],axis=0).tolist()} for dim,vs in ranges.items()}}
            result["topics"][topic]=rec
        summary["groups"][group]=result
    for (group,role),v in diag_total.items():summary["diagnostics"][f"{group}:{role}"]=dict(v)
    for (group,key),vs in sync_all.items():
        a=np.concatenate(vs)/1e6
        summary["sync"][f"{group}:{key}"]={"signed_ms":describe(a),"absolute_ms":describe(np.abs(a))}
    summary["video_summary"]={"frames_header":sum(v["frames_header"] for v in videos),"decoded_videos":sum(v["decoded"] for v in videos),"frames_decoded":sum(v.get("frames_decoded",0) for v in videos),"header_count_mismatches":sum(not v["header_count_matches_mcap"] for v in videos),"pts_source_mismatches":sum(v.get("pts_equal_mcap_source") is False for v in videos),"codecs":dict(Counter(v["codec"] for v in videos)),"profiles":dict(Counter(f'{v["role"]}:{v["width"]}x{v["height"]}' for v in videos))}
    summary["video_summary"]["relative_pts_mismatches"] = sum(v.get("relative_pts_match_mcap_source") is False for v in videos)
    summary["video_summary"]["constant_start_offsets_us"] = [v["constant_start_offset_us"] for v in videos if v.get("relative_pts_match_mcap_source")]
    for filename, rows in [("episodes.csv",episodes),("videos.csv",videos)]:
        if rows:
            keys=sorted(set().union(*(set(r) for r in rows)))
            with (out/filename).open("w",newline="") as f:
                w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps({k:summary[k] for k in ["episodes","videos","total_duration_s","video_summary","errors","video_errors"]},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
