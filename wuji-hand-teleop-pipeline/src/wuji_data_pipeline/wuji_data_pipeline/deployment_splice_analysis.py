"""Offline EEF splice audit; never connects to ROS, a policy server or hardware.

Replay uses recorded alignment indices and the production smoothstep function.
Alternative blend lengths are fixed-input comparisons, not closed-loop forecasts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .deployment_protocol import blend_action_prefix


def _xyz(actions, side):
    return np.asarray([a[f"arm_action_{side}"]["ee_pos"] for a in actions])


def _max_step_cm(points):
    if len(points) < 2:
        return None
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).max() * 100)


def analyze_records(records, side="right"):
    """Report missing evidence explicitly instead of treating it as zero motion."""
    snapshots = {}
    chunks = []
    dispatched = {}
    last_command = None
    publish_intervals = []
    previous_publish = None
    counts = {}
    for record in records:
        event = record["event"]
        counts[event] = counts.get(event, 0) + 1
        if event == "action_stream_clear":
            last_command = None
            previous_publish = None
        elif event == "policy_action_chunk":
            key = (record["generation"], record["request_id"])
            snapshots.setdefault(key, {})[record["stage"]] = record
        elif event == "command_publish":
            if side in record.get("arm_eef", {}):
                last_command = record
            timestamp = record["published_at"]
            if previous_publish is not None:
                publish_intervals.append((timestamp - previous_publish) * 1000)
            previous_publish = timestamp
        elif event == "pending_chunk_activate":
            chunks.append((record, last_command))
        elif event == "action_dispatch":
            dispatched.setdefault(record["chunk_id"], []).append(record)

    results = []
    for activation, anchor in chunks:
        result = {
            "chunk_id": activation["chunk_id"],
            "request_id": activation["request_id"],
            "generation": activation["generation"],
            "skipped_actions": activation["skipped_actions"],
            "blend_steps": activation["blend_steps"],
            "boundary_gap_cm": (
                activation.get("raw_position_jump_m", {}).get(side, 0) * 100
                if side in activation.get("raw_position_jump_m", {}) else None
            ),
        }
        results.append(result)
        key = (activation["generation"], activation["request_id"])
        stages = snapshots.get(key, {})
        if "server_output" not in stages:
            result["unavailable_reason"] = "no recorded server_output"
            continue
        raw = stages["server_output"]["actions"]
        result["server_internal_max_step_cm"] = _max_step_cm(_xyz(raw, side))
        smoothing = activation.get("action_smoothing_method", "none")
        if smoothing != "none" and "post_smoothing" not in stages:
            result["unavailable_reason"] = "missing post_smoothing snapshot"
            continue
        actions = stages["post_smoothing"]["actions"] if smoothing != "none" else raw
        result["client_internal_max_step_cm"] = _max_step_cm(_xyz(actions, side))
        start = activation["skipped_actions"]
        window = actions[start:start + activation["installed_actions"]]
        result["aligned_internal_max_step_cm"] = _max_step_cm(_xyz(window, side))
        actual = dispatched.get(activation["chunk_id"], [])
        if any(side not in a.get("arm_eef", {}) for a in actual):
            result["unavailable_reason"] = "requires EEF dispatch records"
            continue
        result["observed_dispatch_count"] = len(actual)
        adjacent_steps = [
            float(np.linalg.norm(np.asarray(b["arm_eef"][side][:3])
                                 - np.asarray(a["arm_eef"][side][:3])) * 100)
            for a, b in zip(actual, actual[1:])
            if b["chunk_action_index"] == a["chunk_action_index"] + 1
        ]
        result["executed_internal_max_step_cm"] = max(adjacent_steps) if adjacent_steps else None
        result["observed_adjacent_step_count"] = len(adjacent_steps)
        if activation.get("early_splice"):
            result["unavailable_reason"] = (
                "early splice retains a committed waypoint and may bridge; "
                "ordinary boundary replay is not applicable"
            )
            continue
        if activation.get("initial_bridge") or activation.get("blend_method") not in ("none", "smoothstep"):
            result["unavailable_reason"] = "replay supports ordinary smoothstep boundaries only"
            continue
        blend_steps = activation["blend_steps"]
        if blend_steps and (anchor is None or side not in anchor.get("hand_deg", {})):
            result["unavailable_reason"] = "missing previous published EEF/hand anchor"
            continue

        def replay(steps):
            if not steps:
                return window
            return blend_action_prefix(
                window, anchor_poses={side: np.asarray(anchor["arm_eef"][side])},
                anchor_hands={side: np.asarray(anchor["hand_deg"][side])},
                anchor_zsp={side: None}, blend_steps=steps, sides=(side,),
            )

        reproduced = replay(blend_steps)
        errors = [
            np.linalg.norm(
                np.asarray(reproduced[a["chunk_action_index"]][f"arm_action_{side}"]["ee_pos"])
                - np.asarray(a["arm_eef"][side][:3])
            ) for a in actual
        ]
        result["replay_max_position_error_m"] = float(max(errors)) if errors else None
        if blend_steps:
            result["fixed_input_alternative_max_step_cm"] = {
                str(steps): _max_step_cm(np.vstack([
                    np.asarray(anchor["arm_eef"][side][:3]), _xyz(replay(steps), side)
                ])) for steps in (1, 4, 8, 16)
            }
    return {
        "side": side, "event_counts": counts,
        "max_publish_interval_ms": max(publish_intervals) if publish_intervals else None,
        "note": "Distances describe commanded EEF targets, not measured robot motion. "
                "Alternatives keep observations, alignment and anchors fixed; not closed-loop replay. "
                "Server output already includes server-side smoothing.",
        "chunks": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.trace.open() as stream:
        report = analyze_records([json.loads(line) for line in stream if line.strip()], args.side)
    report["trace"] = str(args.trace.resolve())
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
