"""Supervised MANUS/Wuji-glove -> WujiHand-only teleoperation session."""

from __future__ import annotations

import argparse
import signal
import sys
import termios
import threading
import tty

import rclpy

from .record_session import (
    SessionClient,
    _launch_graph,
    _read_key,
    _shutdown_launch_graph,
    _validate_documented_environment,
    active_hand_sides,
    required_hand_driver_nodes,
)


CHILD_LOG_PATH = "/tmp/wuji_hand_teleop_session_children.log"


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Interactive supervised WujiHand-only teleoperation"
    )
    parser.add_argument(
        "--active-hand",
        choices=("both", "left", "right"),
        default="right",
    )
    parser.add_argument("--ramp-seconds", type=float, default=5.0)
    return parser.parse_args(argv)


def _print_help(active_hand: str) -> None:
    print(
        "\nControls:\n"
        "  r  recover selected WujiHand to its configured initial posture\n"
        "  a  enable glove teleoperation after Recovery\n"
        "  x  disable selected WujiHand joints\n"
        "  e/q exit and disable all selected hand joints\n"
        "  Ctrl+C same as q\n"
        "  h  show this help\n"
        f"\nActive hand: {active_hand}.\n"
    )


def _required_enable_nodes(active_hand: str) -> set[str]:
    nodes = required_hand_driver_nodes(active_hand)
    nodes.update(
        f"/wujihand_controller_{side}"
        for side in active_hand_sides(active_hand)
    )
    return nodes


def main(argv=None) -> None:
    args = _parse_args(argv)
    if args.ramp_seconds < 0.0:
        raise SystemExit("--ramp-seconds cannot be negative")
    if not sys.stdin.isatty():
        raise SystemExit(
            "hand_teleop_session requires docker exec -it so its control "
            "keys and Ctrl+C are reliable"
        )
    _validate_documented_environment()

    enable_left = args.active_hand in ("both", "left")
    enable_right = args.active_hand in ("both", "right")
    command = [
        "ros2",
        "launch",
        "wuji_teleop_bringup",
        "wuji_teleop_hand.launch.py",
        f"enable_left_hand:={'true' if enable_left else 'false'}",
        f"enable_right_hand:={'true' if enable_right else 'false'}",
        "auto_enable:=false",
        f"command_ramp_duration:={args.ramp_seconds}",
    ]

    process = None
    node = None
    original_terminal = None
    rclpy_started = False
    running = True
    recovery_pending = False
    recovery_failed = False
    shutdown_event = threading.Event()
    exit_announced = False

    def request_exit(received_signal=None, _frame=None):
        nonlocal running, exit_announced
        running = False
        shutdown_event.set()
        if not exit_announced:
            exit_announced = True
            source = "Exit requested" if received_signal is None else "Ctrl+C received"
            print(f"\n{source}; disabling and shutting down hand teleop...", flush=True)

    signal.signal(signal.SIGINT, request_exit)
    signal.signal(signal.SIGTERM, request_exit)
    signal.signal(signal.SIGHUP, request_exit)
    try:
        print("Starting:", " ".join(command))
        print(f"ROS child logs: {CHILD_LOG_PATH}")
        process = _launch_graph(command, CHILD_LOG_PATH)
        rclpy.init()
        rclpy_started = True
        node = SessionClient(
            node_name="wuji_hand_teleop_session",
            include_recorder=False,
            include_hands=True,
        )
        signal.signal(signal.SIGINT, request_exit)
        signal.signal(signal.SIGTERM, request_exit)
        signal.signal(signal.SIGHUP, request_exit)
        original_terminal = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        _print_help(args.active_hand)

        while running and process.poll() is None:
            rclpy.spin_once(node, timeout_sec=0.0)
            if recovery_pending:
                result = node.hand_recovery_result(args.active_hand)
                if result == "ready":
                    print(
                        "RECOVERY COMPLETE: selected WujiHand is holding its "
                        "configured initial posture; press a to enable"
                    )
                    recovery_pending = False
                    recovery_failed = False
                    termios.tcflush(sys.stdin, termios.TCIFLUSH)
                elif result == "failed":
                    print("RECOVERY FAILED: disable and inspect the WujiHand")
                    node.set_hands_enabled(args.active_hand, False)
                    recovery_pending = False
                    recovery_failed = True
                    termios.tcflush(sys.stdin, termios.TCIFLUSH)

            key = _read_key()
            if key is None:
                continue
            if key in ("\x04", "e", "q"):
                request_exit()
            elif key == "h":
                _print_help(args.active_hand)
            elif key == "r":
                if recovery_pending:
                    print("ERROR: Recovery is already running")
                    continue
                if recovery_failed:
                    print("ERROR: Recovery failure is latched; restart after inspection")
                    continue
                missing = sorted(
                    required_hand_driver_nodes(args.active_hand) - node.visible_nodes()
                )
                if missing:
                    print(
                        "ERROR: Recovery preflight failed: "
                        + "; ".join(f"missing ROS node {name}" for name in missing)
                    )
                    continue
                print(
                    "Recovery requested. Keep fingers clear of the physical hand.",
                    flush=True,
                )
                ok, message = node.start_hand_recovery(
                    args.active_hand,
                    abort_event=shutdown_event,
                )
                print(("OK: " if ok else "ERROR: ") + message)
                recovery_pending = ok
                recovery_failed = not ok
            elif key == "a":
                if recovery_pending or node.hand_recovery_result(args.active_hand) != "ready":
                    print("ERROR: Enable requires a completed WujiHand Recovery")
                    continue
                missing = sorted(
                    _required_enable_nodes(args.active_hand) - node.visible_nodes()
                )
                if missing:
                    print(
                        "ERROR: Enable preflight failed: "
                        + "; ".join(f"missing ROS node {name}" for name in missing)
                    )
                    continue
                ok, message = node.set_hands_enabled(
                    args.active_hand,
                    True,
                    abort_event=shutdown_event,
                )
                print(("ENABLE COMPLETE: " if ok else "ERROR: ") + message)
            elif key == "x":
                was_recovering = recovery_pending
                ok, message = node.set_hands_enabled(
                    args.active_hand,
                    False,
                    abort_event=shutdown_event,
                )
                print(("OK: " if ok else "ERROR: ") + message)
                if was_recovering:
                    recovery_pending = False
                    recovery_failed = False
                    print("RECOVERY CANCELLED: WujiHand joints are disabled")
    finally:
        if original_terminal is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original_terminal)
            except (OSError, termios.error):
                pass
        hand_safe = False
        if node is not None:
            try:
                hand_safe, message = node.set_hands_enabled(args.active_hand, False)
            except Exception as exc:
                message = f"WujiHand disable failed: {exc}"
            print(("HANDS DISABLED: " if hand_safe else "WARNING: ") + message)
            node.destroy_node()
        if rclpy_started:
            try:
                rclpy.shutdown()
            except Exception:
                pass
        stopped = _shutdown_launch_graph(process)
        if hand_safe and stopped:
            print("Hand teleop exited; selected WujiHand joints are disabled.")
        else:
            print("Hand teleop exited with shutdown errors.")

    if not hand_safe or not stopped:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
