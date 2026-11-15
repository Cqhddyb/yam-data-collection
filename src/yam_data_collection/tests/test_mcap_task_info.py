from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from mcap.reader import make_reader
from mcap.writer import Writer
from mcap_ros2._dynamic import serialize_dynamic

from scripts.mcap_task_info import read_task_info, update_main


class TaskInfoUpdateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_episode(self, name: str, *task_ids: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        schema_name = "std_msgs/msg/String"
        definition = "string data\n"
        encode = serialize_dynamic(schema_name, definition)[schema_name]
        with path.open("wb") as stream:
            writer = Writer(stream)
            writer.start(profile="ros2")
            schema = writer.register_schema(schema_name, "ros2msg", definition.encode())
            task_channel = writer.register_channel("/task_info", "cdr", schema)
            other_channel = writer.register_channel("/other", "json", 0)
            for index, task_id in enumerate(task_ids):
                value = {"task_name": "test", "task_description": "old ", "task_id": task_id, "extra": 42}
                writer.add_message(task_channel, index, encode({"data": json.dumps(value)}), index)
            writer.add_message(other_channel, 100, b'{"keep":true}', 99, 7)
            writer.finish()
        return path

    def run_update(self, *args: str) -> tuple[int, str]:
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            result = update_main([str(self.root), *args])
        return result, output.getvalue()

    def test_filter_preview_apply_and_repeat(self) -> None:
        target = self.write_episode("1/episode.mcap", "T3", "T3")
        other = self.write_episode("2/episode.mcap", "T4")
        pending = self.write_episode(".pending/episode.mcap", "T3")
        partial = self.write_episode("3/episode.partial.mcap", "T3")
        before = {p: p.read_bytes() for p in (target, other, pending, partial)}
        args = ("--match-task-id", "T3", "--task-description", "修正后的描述。")

        result, output = self.run_update(*args)
        self.assertEqual(result, 0)
        self.assertIn("would update 1", output)
        self.assertEqual(target.read_bytes(), before[target])
        self.assertFalse(list(self.root.rglob("*.bak*")))

        result, output = self.run_update(*args, "--apply")
        self.assertEqual(result, 0, output)
        self.assertEqual(
            read_task_info(target),
            [{"task_name": "test", "task_description": "修正后的描述。", "task_id": "T3", "extra": 42}] * 2,
        )
        backup = target.with_suffix(".mcap.bak")
        self.assertEqual(backup.read_bytes(), before[target])
        for p in (other, pending, partial):
            self.assertEqual(p.read_bytes(), before[p])
        with target.open("rb") as stream:
            messages = list(make_reader(stream).iter_messages(topics=["/other"]))
        self.assertEqual(len(messages), 1)
        message = messages[0][2]
        self.assertEqual((message.log_time, message.publish_time, message.sequence, message.data),
                         (100, 99, 7, b'{"keep":true}'))

        updated_bytes = target.read_bytes()
        result, output = self.run_update(*args, "--apply")
        self.assertEqual(result, 0, output)
        self.assertIn("unchanged 1", output)
        self.assertEqual(target.read_bytes(), updated_bytes)
        self.assertEqual(list(self.root.rglob("*.bak*")), [backup])

    def test_mixed_ids_rejected_before_any_replacement(self) -> None:
        valid = self.write_episode("1.mcap", "T3")
        mixed = self.write_episode("2.mcap", "T3", "T4")
        before = {p: p.read_bytes() for p in (valid, mixed)}
        result, output = self.run_update(
            "--match-task-id", "T3", "--task-description", "new", "--apply"
        )
        self.assertEqual(result, 1)
        self.assertIn("mixed task_ids", output)
        self.assertFalse(list(self.root.rglob("*.bak*")))
        for p in before:
            self.assertEqual(p.read_bytes(), before[p])

    def test_rename_only_keeps_description(self) -> None:
        path = self.write_episode("1.mcap", "T3")
        result, output = self.run_update("--match-task-id", "T3", "--task-id", "renamed", "--apply")
        self.assertEqual(result, 0, output)
        self.assertEqual(read_task_info(path)[0]["task_description"], "old ")
        self.assertEqual(read_task_info(path)[0]["task_id"], "renamed")

    def test_range_still_supports_timestamp_names_and_both_fields(self) -> None:
        path = self.write_episode("69/2026-10-19_16-05.mcap", "T3")
        result, output = self.run_update(
            "--range", "69-69", "--task-id", "T3", "--task-description", "", "--apply"
        )
        self.assertEqual(result, 0, output)
        self.assertEqual(read_task_info(path)[0]["task_description"], "")


if __name__ == "__main__":
    unittest.main()
