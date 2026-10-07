"""One-command interactive Tianji + Wuji replay session."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import termios
import threading
import tty
from typing import Optional

import numpy as np
import rclpy
from std_srvs.srv import SetBool, Trigger

from .config import default_config_path, load_config, section
from .episode import load_episode
from .record_session import (
    LIFECYCLE_NAMES,
    SessionClient,
    _launch_graph,
    _read_key,
    _request_tianji_standby,
    _shutdown_launch_graph,
    _validate_documented_environment,
    arm_status_errors,
)
from .schema import RobotLayout


REPLAY_SERVER_LOG = "/tmp/wuji_replay_server.log"
REPLAY_DEPLOYMENT_LOG = "/tmp/wuji_replay_deployment.log"
DEFAULT_PLAYBACK_RATE_HZ = 6.0
DEFAULT_ACTION_RATE_HZ = 30.0
TIANJI_CONTROL_RATE_HZ = 120.0
TIANJI_STATE_RATE_HZ = 500.0
REPLAY_CHUNK_SIZE = 30
RECOVERY_TERMINAL_STATES = {
    7: "RECOVERY COMPLETE: configured arm mode is ready; press a to replay",
    8: "RECOVERY FAILED: inspect arm status and restart the session",
    9: "RECOVERY PARTIAL: dual-arm replay cannot continue",
    5: "RECOVERY FAILED: Tianji SDK/hardware error",
}
ENABLE_TERMINAL_STATES = {
    2: "REPLAY STARTED: Tianji is READY; the episode is now playing",
    4: "ENABLE FAILED: Tianji requested standby",
    5: "ENABLE FAILED: Tianji SDK/hardware error",
}


def required_replay_nodes(active_hand: str) -> set[str]:
    nodes = {"/tianji_arm_controller", "/wuji_deployment"}
    if active_hand in ("both", "left"):
        nodes.add("/left_hand/wujihand_driver")
    if active_hand in ("both", "right"):
        nodes.add("/right_hand/wujihand_driver")
    return nodes


def replay_recovery_preflight(
    node: SessionClient,
    active_hand: str,
    abort_event: Optional[threading.Event] = None,
) -> list[str]:
    missing = sorted(required_replay_nodes(active_hand) - node.visible_nodes())
    if missing:
        return [f"missing ROS node {name}" for name in missing]
    ok, message = node.call(
        "arm_status",
        Trigger.Request(),
        timeout_s=3.0,
        abort_event=abort_event,
    )
    if not ok:
        return [f"arm_status failed: {message}"]
    try:
        status = json.loads(message)
    except json.JSONDecodeError:
        return [f"arm_status returned invalid JSON: {message}"]
    return arm_status_errors(status)


def load_first_arm_joint_targets(
    episode_dir: str,
    active_arm: str,
) -> dict[str, list[float]]:
    """Return the episode's first measured arm qpos in Tianji degrees."""
    arrays, metadata = load_episode(episode_dir)
    layout = RobotLayout.from_metadata(metadata)
    qpos = np.asarray(arrays.get("qpos"), dtype=np.float64)
    expected = (layout.state_dim,)
    if qpos.ndim != 2 or qpos.shape[0] == 0 or qpos.shape[1:] != expected:
        raise ValueError(
            "absolute Replay requires non-empty qpos with shape "
            f"Nx{layout.state_dim}, got {qpos.shape}"
        )

    active_sides = set(
        layout.sides if active_arm == "both" else (active_arm,)
    )
    targets: dict[str, list[float]] = {}
    cursor = 0
    for side in layout.sides:
        arm_rad = qpos[0, cursor : cursor + layout.arm_dof]
        if side in active_sides:
            if arm_rad.shape != (layout.arm_dof,) or not np.all(
                np.isfinite(arm_rad)
            ):
                raise ValueError(
                    f"episode first-frame {side} arm qpos is invalid"
                )
            targets[side] = np.degrees(arm_rad).tolist()
        cursor += layout.arm_dof + layout.hand_dof

    missing = active_sides - set(targets)
    if missing:
        raise ValueError(
            "episode layout is missing active arm side(s): "
            + ", ".join(sorted(missing))
        )
    return targets


def _print_help() -> None:
    print(
        "\nControls:\n"
        "  r  guarded Tianji Recovery\n"
        "  a  Enable Tianji and start replay after Recovery\n"
        "  x  Disable Tianji and request standby\n"
        "  e/q exit\n"
        "  Ctrl+C disable Tianji and stop every replay process\n"
        "  h  show this help\n"
    )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Interactive LMDB replay session")
    parser.add_argument("--episode-dir", required=True)
    parser.add_argument(
        "--active-hand",
        choices=("both", "left", "right"),
        default="both",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="deployment config override (default: dedicated replay.yaml)",
    )
    parser.add_argument("--with-camera", action="store_true")
    parser.add_argument(
        "--arm-command-mode",
        choices=("eef", "joint"),
        default="eef",
        help="arm replay source: recorded EEF action (default) or motor qpos",
    )
    parser.add_argument(
        "--arm-hardware-mode",
        choices=("position", "impedance"),
        default="position",
        help="Tianji SDK mode used during replay (default: position)",
    )
    parser.add_argument(
        "--playback-rate-hz",
        "--rate-hz",
        dest="playback_rate_hz",
        type=float,
        default=DEFAULT_PLAYBACK_RATE_HZ,
        help=(
            "recorded source frames advanced per second (default: 6 Hz, "
            "which is 0.2x for a 30 Hz episode)"
        ),
    )
    parser.add_argument(
        "--action-rate-hz",
        type=float,
        default=DEFAULT_ACTION_RATE_HZ,
        help="replay waypoint stream rate (default: 30 Hz)",
    )
    parser.add_argument(
        "--start-hold-s",
        type=float,
        default=1.0,
        help="hold the rebased/recorded first frame before playback",
    )
    args = parser.parse_args(argv)
    if args.playback_rate_hz <= 0.0:
        parser.error("--playback-rate-hz must be positive")
    if args.action_rate_hz <= 0.0:
        parser.error("--action-rate-hz must be positive")
    if args.start_hold_s < 0.0:
        parser.error("--start-hold-s must be non-negative")
    return args


def _resolve_replay_config(config: Optional[str]) -> str:
    return (
        os.path.abspath(os.path.expanduser(config))
        if config
        else str(default_config_path().with_name("replay.yaml"))
    )


def _validate_replay_config(
    config_path: str,
    action_rate_hz: float,
) -> None:
    config = load_config(config_path)
    deployment = section(config, "deployment")
    configured_rate = float(deployment.get("action_rate_hz", 30.0))
    publish_rate = float(deployment.get("publish_rate_hz", 120.0))
    horizon = int(deployment.get("open_loop_horizon", 25))
    if not math.isclose(configured_rate, action_rate_hz):
        raise ValueError(
            "Replay --action-rate-hz must match "
            "deployment.action_rate_hz in the replay config "
            f"({action_rate_hz:g} != {configured_rate:g})"
        )
    if publish_rate < configured_rate:
        raise ValueError(
            "Replay deployment.publish_rate_hz must be at least its "
            "action_rate_hz"
        )
    if not 1 <= horizon <= REPLAY_CHUNK_SIZE:
        raise ValueError(
            "Replay open_loop_horizon must be in "
            f"[1, {REPLAY_CHUNK_SIZE}], got {horizon}"
        )


def _build_commands(
    args,
    episode_dir: str,
    config_path: Optional[str] = None,
    first_joint_targets: Optional[dict[str, list[float]]] = None,
) -> tuple[list[str], list[str]]:
    server_command = [
        "ros2",
        "run",
        "wuji_data_pipeline",
        "replay_server",
        "--episode-dir",
        episode_dir,
        "--bind",
        "tcp://0.0.0.0:5555",
        "--playback-rate-hz",
        str(args.playback_rate_hz),
        "--action-rate-hz",
        str(args.action_rate_hz),
        "--start-hold-s",
        str(args.start_hold_s),
        "--chunk-size",
        str(REPLAY_CHUNK_SIZE),
        "--arm-command-mode",
        args.arm_command_mode,
    ]
    # Both replay modes reproduce the original absolute trajectory.  Enable
    # first moves Tianji to the episode's recorded first qpos, so applying a
    # Cartesian rebase here would incorrectly translate/rotate the episode.
    server_command.append("--no-rebase")

    deployment_command = [
        "ros2",
        "launch",
        "wuji_data_pipeline",
        "deployment.launch.py",
        f"enable_camera:={'true' if args.with_camera else 'false'}",
        f"active_hand:={args.active_hand}",
        f"arm_command_mode:={args.arm_command_mode}",
        f"arm_hardware_mode:={args.arm_hardware_mode}",
        "replay_first_joint_targets_json:="
        + json.dumps(first_joint_targets or {}, separators=(",", ":")),
        "server:=tcp://127.0.0.1:5555",
    ]
    config_path = config_path or _resolve_replay_config(args.config)
    deployment_command.append(f"pipeline_config:={config_path}")
    return server_command, deployment_command


def main(argv=None) -> None:
    args = _parse_args(argv)
    if not sys.stdin.isatty():
        raise SystemExit("replay_session requires docker exec -it")
    _validate_documented_environment()
    episode_dir = os.path.abspath(os.path.expanduser(args.episode_dir))
    if not os.path.isdir(episode_dir):
        raise SystemExit(f"episode directory does not exist: {episode_dir}")

    config_path = _resolve_replay_config(args.config)
    try:
        _validate_replay_config(config_path, args.action_rate_hz)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid Replay config: {exc}") from exc
    try:
        first_joint_targets = load_first_arm_joint_targets(
            episode_dir, args.active_hand
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid absolute Replay episode: {exc}") from exc
    server_command, deployment_command = _build_commands(
        args,
        episode_dir,
        config_path,
        first_joint_targets=first_joint_targets,
    )

    server_process = None
    deployment_process = None
    node = None
    original_terminal = None
    rclpy_started = False
    running = True
    shutdown_event = threading.Event()
    exit_announced = False
    recovery_pending = False
    recovery_failed = False
    arm_recovery_terminal = None
    enable_pending = False
    hands_enabled = False
    last_lifecycle = None

    def request_exit(received_signal=None, _frame=None):
        nonlocal running, exit_announced
        running = False
        shutdown_event.set()
        if not exit_announced:
            exit_announced = True
            source = "Exit requested" if received_signal is None else "Ctrl+C received"
            print(f"\n{source}; disabling and shutting down replay...", flush=True)

    signal.signal(signal.SIGINT, request_exit)
    signal.signal(signal.SIGTERM, request_exit)
    signal.signal(signal.SIGHUP, request_exit)
    try:
        print(f"Episode: {episode_dir}")
        print(f"Active hand: {args.active_hand}")
        print(f"Active arm/hand mode: {args.active_hand}")
        print(
            "Absolute Replay startup: configured Recovery -> recorded first "
            "qpos -> original EEF trajectory (rebase disabled)"
        )
        if args.arm_command_mode == "eef":
            print("Arm replay source: recorded EEF pose action -> Tianji IK")
        else:
            print("Arm replay source: recorded qpos motor joints (7 DoF)")
        if args.arm_hardware_mode == "position":
            print("Tianji replay mode: rigid position (SDK state=1)")
        else:
            print("Tianji replay mode: joint impedance (SDK state=3)")
        print(
            "Replay timing: "
            f"trajectory={args.playback_rate_hz:.2f} source-frames/s, "
            f"waypoints={args.action_rate_hz:.2f}Hz, "
            f"Tianji control={TIANJI_CONTROL_RATE_HZ:.0f}Hz, "
            f"state read={TIANJI_STATE_RATE_HZ:.0f}Hz"
        )
        print(f"Replay first-frame hold: {args.start_hold_s:.2f}s")
        print(f"Replay log: {REPLAY_SERVER_LOG}")
        print(f"Deployment log: {REPLAY_DEPLOYMENT_LOG}")
        server_process = _launch_graph(server_command, REPLAY_SERVER_LOG)
        deployment_process = _launch_graph(deployment_command, REPLAY_DEPLOYMENT_LOG)
        rclpy.init()
        rclpy_started = True
        node = SessionClient(
            node_name="wuji_replay_session",
            include_recorder=False,
            include_hands=True,
            include_deployment=True,
        )
        signal.signal(signal.SIGINT, request_exit)
        signal.signal(signal.SIGTERM, request_exit)
        signal.signal(signal.SIGHUP, request_exit)
        original_terminal = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        _print_help()

        while running:
            if server_process.poll() is not None:
                print(f"ERROR: replay server exited; inspect {REPLAY_SERVER_LOG}")
                break
            if deployment_process.poll() is not None:
                print(f"ERROR: deployment exited; inspect {REPLAY_DEPLOYMENT_LOG}")
                break
            rclpy.spin_once(node, timeout_sec=0.0)
            if node.lifecycle != last_lifecycle:
                last_lifecycle = node.lifecycle
                if last_lifecycle is not None:
                    print(
                        f"\nTianji lifecycle: {last_lifecycle}/"
                        f"{LIFECYCLE_NAMES.get(last_lifecycle, 'UNKNOWN')}"
                    )
            if recovery_pending:
                if node.lifecycle in RECOVERY_TERMINAL_STATES:
                    arm_recovery_terminal = node.lifecycle
                hand_result = node.hand_recovery_result(args.active_hand)
                if arm_recovery_terminal not in (None, 7):
                    print(RECOVERY_TERMINAL_STATES[arm_recovery_terminal])
                    node.set_hands_enabled(args.active_hand, False)
                    recovery_pending = False
                    recovery_failed = True
                    termios.tcflush(sys.stdin, termios.TCIFLUSH)
                elif hand_result == "failed":
                    print("RECOVERY FAILED: WujiHand initial-position move failed")
                    node.set_hands_enabled(args.active_hand, False)
                    _request_tianji_standby(node)
                    recovery_pending = False
                    recovery_failed = True
                    termios.tcflush(sys.stdin, termios.TCIFLUSH)
                elif arm_recovery_terminal == 7 and hand_result == "ready":
                    print(
                        "RECOVERY COMPLETE: Tianji and selected WujiHand are "
                        "at their configured initial postures; press a to replay"
                    )
                    recovery_pending = False
                    recovery_failed = False
                    termios.tcflush(sys.stdin, termios.TCIFLUSH)
            if enable_pending and node.lifecycle in ENABLE_TERMINAL_STATES:
                if node.lifecycle == 2:
                    hand_ok, hand_message = node.set_hands_enabled(
                        args.active_hand,
                        True,
                        abort_event=shutdown_event,
                    )
                    if hand_ok:
                        hands_enabled = True
                        print(
                            "REPLAY STARTED: Tianji is READY; "
                            f"{hand_message}; the episode is now playing"
                        )
                    else:
                        print(
                            "ERROR: Arm reached READY but hand enable failed: "
                            + hand_message
                        )
                        standby_ok, standby_message = _request_tianji_standby(node)
                        print(
                            ("STANDBY: " if standby_ok else "CRITICAL: ")
                            + standby_message
                        )
                else:
                    print(ENABLE_TERMINAL_STATES[node.lifecycle])
                    node.set_hands_enabled(args.active_hand, False)
                    hands_enabled = False
                enable_pending = False
                termios.tcflush(sys.stdin, termios.TCIFLUSH)

            if hands_enabled and node.lifecycle in (0, 3, 4, 5, 8, 9):
                hand_ok, hand_message = node.set_hands_enabled(
                    args.active_hand,
                    False,
                    abort_event=shutdown_event,
                )
                print(("HAND SAFE: " if hand_ok else "WARNING: ") + hand_message)
                hands_enabled = False

            key = _read_key()
            if key is None:
                continue
            if key in ("\x04", "e", "q"):
                request_exit()
            elif key == "h":
                _print_help()
            elif key == "r":
                if recovery_pending:
                    print("ERROR: Recovery is already running")
                    continue
                if recovery_failed:
                    print("ERROR: Recovery failure is latched; restart after inspection")
                    continue
                print("r received; checking replay nodes and arm status...", flush=True)
                errors = replay_recovery_preflight(
                    node, args.active_hand, abort_event=shutdown_event
                )
                if errors:
                    if not shutdown_event.is_set():
                        print("ERROR: Recovery preflight failed: " + "; ".join(errors))
                    continue
                print("Recovery requested. Clear the workspace and keep E-stop reachable.")
                recovery_client = {
                    "both": "recover",
                    "left": "recover_left",
                    "right": "recover_right",
                }[args.active_hand]
                ok, message = node.call(
                    recovery_client,
                    Trigger.Request(),
                    timeout_s=3.0,
                    abort_event=shutdown_event,
                )
                print(("OK: " if ok else "ERROR: ") + message)
                if ok:
                    hand_ok, hand_message = node.start_hand_recovery(
                        args.active_hand,
                        abort_event=shutdown_event,
                    )
                    print(("OK: " if hand_ok else "ERROR: ") + hand_message)
                    if not hand_ok:
                        node.set_hands_enabled(args.active_hand, False)
                        _request_tianji_standby(node)
                        recovery_failed = True
                    recovery_pending = hand_ok
                    arm_recovery_terminal = None
                else:
                    recovery_pending = False
            elif key == "a":
                if node.lifecycle != 7:
                    current = LIFECYCLE_NAMES.get(node.lifecycle, str(node.lifecycle))
                    print(f"ERROR: Enable requires RECOVERY_READY, got {current}")
                    continue
                if recovery_pending or node.hand_recovery_result(args.active_hand) != "ready":
                    print(
                        "ERROR: Enable requires WujiHand Recovery READY; "
                        "wait for the combined RECOVERY COMPLETE message"
                    )
                    continue
                server_ok, server_message = node.call(
                    "deployment_ready",
                    Trigger.Request(),
                    timeout_s=2.0,
                    service_wait_s=0.5,
                    abort_event=shutdown_event,
                )
                if not server_ok:
                    print("ERROR: Replay server is not ready: " + server_message)
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
                was_recovering = recovery_pending
                hand_ok, hand_message = node.set_hands_enabled(
                    args.active_hand,
                    False,
                    abort_event=shutdown_event,
                )
                print(("OK: " if hand_ok else "WARNING: ") + hand_message)
                hands_enabled = False
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
                if was_recovering:
                    recovery_pending = False
                    recovery_failed = False
                    arm_recovery_terminal = None
                    print("RECOVERY CANCELLED: Tianji/WujiHand returned to standby")
    finally:
        if original_terminal is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original_terminal)
            except (OSError, termios.error):
                pass
        standby_confirmed = False
        if node is not None:
            hand_ok, hand_message = node.set_hands_enabled(
                args.active_hand,
                False,
            )
            print(("HAND SAFE: " if hand_ok else "WARNING: ") + hand_message)
            print("Requesting Tianji standby and verifying state=0...", flush=True)
            try:
                standby_confirmed, message = _request_tianji_standby(node)
            except Exception as exc:
                message = f"Tianji standby verification failed: {exc}"
            print(("STANDBY: " if standby_confirmed else "CRITICAL: ") + message)
            try:
                node.destroy_node()
            except Exception:
                pass
        if rclpy_started:
            try:
                rclpy.shutdown()
            except Exception:
                pass
        deployment_stopped = _shutdown_launch_graph(deployment_process)
        server_stopped = _shutdown_launch_graph(server_process)
        if standby_confirmed and deployment_stopped and server_stopped:
            print("Replay session exited; Tianji state=0 confirmed; all processes stopped.")
        else:
            print("Replay session exited with shutdown errors.")

    if not standby_confirmed or not deployment_stopped or not server_stopped:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
