"""Interactive arm-only HTC Tracker -> Tianji teleoperation cockpit.

This entry point deliberately launches no recorder, camera, MANUS process,
Wuji Hand driver, or hand controller.  It reuses the guarded Tianji
Recovery/Enable/standby primitives from the recording cockpit so the hardware
shutdown contract remains identical.
"""

from __future__ import annotations

import argparse
import signal
import sys
import termios
import threading
import tty
from typing import Optional

import rclpy
from std_srvs.srv import SetBool, Trigger

from .record_session import (
    ENABLE_TERMINAL_STATES,
    LIFECYCLE_NAMES,
    RECOVERY_TERMINAL_STATES,
    SessionClient,
    _launch_graph,
    _read_key,
    _request_tianji_standby,
    _shutdown_launch_graph,
    _validate_documented_environment,
)


CHILD_LOG_PATH = "/tmp/wuji_arm_teleop_session_children.log"
ARM_ONLY_REQUIRED_NODES = frozenset({
    "/openvr_input",
    "/tianji_arm_controller",
    "/left_chest_base_tf",
    "/left_chest_tf",
    "/right_chest_base_tf",
    "/right_chest_tf",
    "/tianji_left_tf",
    "/tianji_right_tf",
})


def arm_only_required_nodes(active_arm: str) -> frozenset[str]:
    nodes = {"/openvr_input", "/tianji_arm_controller"}
    sides = ("left", "right") if active_arm == "both" else (active_arm,)
    for side in sides:
        nodes.update({
            f"/{side}_chest_base_tf",
            f"/{side}_chest_tf",
            f"/tianji_{side}_tf",
        })
    return frozenset(nodes)


def arm_only_tf_pairs(active_arm: str) -> tuple[tuple[str, str], ...]:
    sides = ("left", "right") if active_arm == "both" else (active_arm,)
    return tuple(
        pair
        for side in sides
        for pair in (
            (f"{side}_chest", f"tianji_{side}"),
            (f"{side}_chest", f"{side}_arm"),
        )
    )


def _print_help(active_arm: str) -> None:
    print(
        "\nControls:\n"
        "  r  guarded Tianji Recovery (moves one arm at a time)\n"
        "  a  Enable Tianji impedance/Tracker teleop after Recovery\n"
        "  x  Disable Tianji and request standby\n"
        "  e/q exit: confirm Tianji state=0, then stop this session\n"
        "  Ctrl+C same as q\n"
        "  h  show this help\n"
        f"\nActive arm: {active_arm}.\n"
        "Arm-only mode launches no hands, MANUS, cameras, or recorder.\n"
        "Recovery/Enable follow TIANJI_VIVE_TELEOP_RECORD.md exactly.\n"
    )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Interactive Tracker -> Tianji arm-only teleoperation")
    parser.add_argument(
        "active_arm",
        nargs="?",
        choices=("both", "left", "right"),
        default="both",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    if not sys.stdin.isatty():
        raise SystemExit(
            "arm_teleop_session requires an interactive terminal; start it "
            "with docker exec -it so Ctrl+C and the control keys are reliable"
        )
    _validate_documented_environment()

    command = [
        "ros2",
        "launch",
        "wuji_teleop_bringup",
        "arm_only_teleop.launch.py",
        f"active_arm:={args.active_arm}",
    ]
    launch_process = None
    node: Optional[SessionClient] = None
    original_terminal = None
    rclpy_started = False
    running = True
    shutdown_event = threading.Event()
    exit_announced = False
    recovery_pending = False
    recovery_failed = False
    enable_pending = False
    last_lifecycle = None

    def request_exit(_signal=None, _frame=None):
        nonlocal running, exit_announced
        running = False
        shutdown_event.set()
        if not exit_announced:
            exit_announced = True
            source = "Exit requested" if _signal is None else "Ctrl+C received"
            print(f"\n{source}; disabling and shutting down...", flush=True)

    signal.signal(signal.SIGINT, request_exit)
    signal.signal(signal.SIGTERM, request_exit)
    signal.signal(signal.SIGHUP, request_exit)
    try:
        print("Starting:", " ".join(command))
        print(f"ROS child logs: {CHILD_LOG_PATH}")
        launch_process = _launch_graph(command, CHILD_LOG_PATH)
        rclpy.init()
        rclpy_started = True
        node = SessionClient(
            node_name="wuji_arm_teleop_session",
            include_recorder=False,
        )
        # rclpy installs signal handlers during init; the cockpit owns the
        # ordered standby -> child-process shutdown sequence.
        signal.signal(signal.SIGINT, request_exit)
        signal.signal(signal.SIGTERM, request_exit)
        signal.signal(signal.SIGHUP, request_exit)
        original_terminal = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        _print_help(args.active_arm)

        while running and launch_process.poll() is None:
            rclpy.spin_once(node, timeout_sec=0.0)
            if node.lifecycle != last_lifecycle:
                last_lifecycle = node.lifecycle
                if last_lifecycle is not None:
                    print(
                        f"\nTianji lifecycle: {last_lifecycle}/"
                        f"{LIFECYCLE_NAMES.get(last_lifecycle, 'UNKNOWN')}"
                    )
            if recovery_pending and node.lifecycle in RECOVERY_TERMINAL_STATES:
                message = RECOVERY_TERMINAL_STATES[node.lifecycle]
                if node.lifecycle == 9:
                    message = (
                        "RECOVERY PARTIAL: configured-arm teleoperation cannot continue"
                    )
                elif node.lifecycle == 7:
                    if args.active_arm == "right":
                        message = (
                            "RECOVERY COMPLETE: left arm is vertically parked, "
                            "right arm is at init; press a to enable"
                        )
                    else:
                        message = (
                            f"RECOVERY COMPLETE: {args.active_arm} arm mode is "
                            "at init in standby; press a to enable"
                        )
                print(message)
                recovery_pending = False
                recovery_failed = node.lifecycle != 7
                termios.tcflush(sys.stdin, termios.TCIFLUSH)
            if enable_pending and node.lifecycle in ENABLE_TERMINAL_STATES:
                print(ENABLE_TERMINAL_STATES[node.lifecycle])
                enable_pending = False
                termios.tcflush(sys.stdin, termios.TCIFLUSH)

            key = _read_key()
            if key is None:
                continue
            if key in ("\x04", "e", "q"):
                request_exit()
            elif key == "h":
                _print_help(args.active_arm)
            elif key == "r":
                if recovery_pending:
                    print(
                        "ERROR: Recovery is already running; wait for "
                        "lifecycle 7 or 8"
                    )
                    continue
                if recovery_failed:
                    print(
                        "ERROR: Recovery failure is latched; inspect hardware "
                        "and restart"
                    )
                    continue
                print(
                    "r received; checking Tianji controller and arm status...",
                    flush=True,
                )
                errors = node.recovery_preflight_nodes(
                    frozenset({"/tianji_arm_controller"}),
                    abort_event=shutdown_event,
                    tf_pairs=(),
                )
                if errors:
                    if not shutdown_event.is_set():
                        print(
                            "ERROR: Recovery preflight failed: "
                            + "; ".join(errors)
                        )
                    continue
                print(
                    "\nRecovery requested. Clear the workspace and keep "
                    "E-stop reachable."
                )
                recovery_client = {
                    "both": "recover",
                    "left": "recover_left",
                    "right": "recover_right",
                }[args.active_arm]
                ok, message = node.call(
                    recovery_client,
                    Trigger.Request(),
                    timeout_s=3.0,
                    abort_event=shutdown_event,
                )
                print(("OK: " if ok else "ERROR: ") + message)
                recovery_pending = ok
            elif key == "a":
                if node.lifecycle != 7:
                    current = LIFECYCLE_NAMES.get(
                        node.lifecycle, str(node.lifecycle)
                    )
                    print(
                        "ERROR: Enable requires lifecycle 7/RECOVERY_READY, "
                        f"got {current}"
                    )
                    continue
                print(
                    "a received; checking Tracker nodes and TF...",
                    flush=True,
                )
                errors = node.enable_preflight_nodes(
                    arm_only_required_nodes(args.active_arm),
                    abort_event=shutdown_event,
                    tf_pairs=arm_only_tf_pairs(args.active_arm),
                )
                if errors:
                    if not shutdown_event.is_set():
                        print(
                            "ERROR: Enable preflight failed: "
                            + "; ".join(errors)
                        )
                    continue
                request = SetBool.Request()
                request.data = True
                ok, message = node.call(
                    "enable",
                    request,
                    timeout_s=3.0,
                    abort_event=shutdown_event,
                )
                print(("OK: " if ok else "ERROR: ") + message)
                enable_pending = ok
            elif key == "x":
                request = SetBool.Request()
                request.data = False
                ok, message = node.call(
                    "enable",
                    request,
                    timeout_s=8.0,
                    service_wait_s=0.5,
                    abort_event=shutdown_event,
                )
                print(("OK: " if ok else "ERROR: ") + message)
    finally:
        if original_terminal is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original_terminal)
            except (OSError, termios.error):
                pass

        standby_confirmed = False
        if node is not None:
            print("Requesting Tianji standby and verifying state=0...", flush=True)
            try:
                standby_confirmed, message = _request_tianji_standby(node)
            except Exception as exc:
                message = f"Tianji standby verification failed: {exc}"
            if standby_confirmed:
                print(f"STANDBY: {message}", flush=True)
            else:
                print(f"CRITICAL: {message}", flush=True)
                print(
                    "Tianji state=0 was not confirmed; use the physical E-stop "
                    "before approaching the robot.",
                    flush=True,
                )
            try:
                node.destroy_node()
            except Exception:
                pass
        if rclpy_started:
            try:
                rclpy.shutdown()
            except Exception:
                pass

        print("Stopping ROS child processes...", flush=True)
        children_stopped = _shutdown_launch_graph(launch_process)
        if not children_stopped:
            print(
                "ERROR: the ROS launch process group still did not exit "
                "after SIGKILL.",
                flush=True,
            )
        if standby_confirmed and children_stopped:
            print(
                "Session exited; Tianji state=0 confirmed; all arm-only "
                "session processes stopped."
            )
        else:
            print("Session exited with shutdown errors.", flush=True)

    if not standby_confirmed or not children_stopped:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
