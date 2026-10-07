"""Collect /tianji_arm/performance_metrics into JSONL + summary JSON."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import time
from typing import Optional

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from std_msgs.msg import String

from .common import get_default_qos
from .performance_metrics import MetricsSummary


DEFAULT_OUTPUT_DIR = Path(
    "/home/wuji/datasets/tianji_wuji/diagnostics"
)


def _safe_label(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip())
    return result.strip("_") or "baseline"


class PerformanceCollector(Node):
    def __init__(self, output: Path, label: str) -> None:
        super().__init__("tianji_performance_collector")
        self.output = output
        self.label = label
        self.summary = MetricsSummary()
        self.received = 0
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.output.open("w", encoding="utf-8")
        self.create_subscription(
            String,
            "/tianji_arm/performance_metrics",
            self._on_metrics,
            get_default_qos(),
        )

    def _on_metrics(self, message: String) -> None:
        try:
            snapshot = json.loads(message.data)
            if not isinstance(snapshot, dict):
                raise ValueError("metrics payload is not an object")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().warning(f"Rejected metrics payload: {exc}")
            return
        self.summary.add(snapshot)
        self.received += 1
        self._stream.write(
            json.dumps(snapshot, separators=(",", ":"), sort_keys=True)
            + "\n"
        )
        self._stream.flush()

        rates = snapshot.get("callback_rates_hz", {})
        series = snapshot.get("series", {})
        control_p99 = (
            series.get("control.duration_ms", {}).get("p99", 0.0)
        )
        state_read_p99 = (
            series.get("driver.state.sdk_read_ms", {}).get("p99", 0.0)
        )
        self.get_logger().info(
            "sample=%d control=%.1fHz state=%.1fHz "
            "control_p99=%.2fms state_sdk_read_p99=%.2fms"
            % (
                self.received,
                float(rates.get("control", 0.0)),
                float(rates.get("state", 0.0)),
                float(control_p99),
                float(state_read_p99),
            )
        )

    def finish(self) -> Path:
        if not self._stream.closed:
            self._stream.close()
        summary_path = self.output.with_suffix(".summary.json")
        summary_path.write_text(
            json.dumps(
                self.summary.result(self.label),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return summary_path


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Tianji controller performance baseline"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="collection duration in seconds; 0 waits for Ctrl+C",
    )
    parser.add_argument("--label", default="baseline")
    parser.add_argument(
        "--output",
        default=None,
        help="JSONL path (default: dataset diagnostics directory)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    program_name = sys.argv[0] if sys.argv else "tianji_performance_collector"
    raw_argv = sys.argv if argv is None else [program_name, *argv]
    args = _parse_args(remove_ros_args(raw_argv)[1:])
    if args.duration < 0.0:
        raise SystemExit("--duration must be >= 0")
    label = _safe_label(args.label)
    if args.output:
        output = Path(args.output).expanduser()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = DEFAULT_OUTPUT_DIR / f"{stamp}_{label}.jsonl"

    rclpy.init(args=raw_argv)
    node = PerformanceCollector(output, label)
    started_at = time.monotonic()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)
            if args.duration > 0.0 and time.monotonic() - started_at >= args.duration:
                break
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        summary_path = node.finish()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print(f"Metrics JSONL: {output}")
    print(f"Summary JSON: {summary_path}")
