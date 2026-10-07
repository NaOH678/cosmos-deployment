"""Interactive supervised session for a remote Tianji/Wuji policy server."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys
import termios
import threading
import time
import tty

import rclpy
from ament_index_python.packages import get_package_share_directory
from std_srvs.srv import SetBool, Trigger

from .config import load_config, section
from .policy_transport import policy_transport_kind
from .record_session import (
    LIFECYCLE_NAMES,
    SessionClient,
    _launch_graph,
    _read_key,
    _request_tianji_standby,
    _shutdown_launch_graph,
    _validate_documented_environment,
)
from .replay_session import (
    ENABLE_TERMINAL_STATES,
    RECOVERY_TERMINAL_STATES,
    required_replay_nodes,
    replay_recovery_preflight,
)
from .schema import (
    ARM_ACTION_SPACE_JOINT,
    normalize_arm_action_space,
)


DEPLOYMENT_LOG = "/tmp/wuji_cloud_deployment.log"
PI_PROTOCOL_MODE = "pi_v2"
PROTOCOL_V2_MODE = "protocol_v2"
SYNCHRONOUS_PROTOCOL_MODES = (PROTOCOL_V2_MODE, PI_PROTOCOL_MODE)
DEPLOYMENT_GRAPH_READY_TIMEOUT_S = 15.0


def _wait_for_deployment_nodes(
    node: SessionClient,
    active_hand: str,
    *,
    timeout_s: float = DEPLOYMENT_GRAPH_READY_TIMEOUT_S,
    abort_event: threading.Event | None = None,
) -> list[str]:
    """Wait for launch/discovery startup before accepting Recovery input."""

    required = required_replay_nodes(active_hand)
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        missing = sorted(required - node.visible_nodes())
        if not missing:
            return []
        if abort_event is not None and abort_event.is_set():
            return missing
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return missing
        rclpy.spin_once(node, timeout_sec=min(0.05, remaining))


def _synchronous_http_startup_handoff_enabled(args) -> bool:
    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else Path(get_package_share_directory("wuji_data_pipeline"))
        / "config"
        / "cosmos_protocol_v2.yaml"
    )
    config = load_config(str(config_path))
    deployment = section(config, "deployment")
    server = str(
        args.server or deployment.get("server", "")
    ).strip()
    return bool(
        deployment.get("startup_handoff_gate_enabled", True)
        and str(
            deployment.get("protocol_mode", PI_PROTOCOL_MODE)
        ).strip().lower()
        in SYNCHRONOUS_PROTOCOL_MODES
        and policy_transport_kind(server) == "http"
    )


def _fdm_state_history_hand_first_enabled(args) -> bool:
    """Limit hand-first Enable ordering to the FDM state-history profile."""

    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else Path(get_package_share_directory("wuji_data_pipeline"))
        / "config"
        / "cosmos_protocol_v2.yaml"
    )
    config = load_config(str(config_path))
    deployment = section(config, "deployment")
    fdm_async = deployment.get("fdm_async", {})
    if not isinstance(fdm_async, dict):
        return False
    state_history = fdm_async.get("state_history", {})
    return bool(
        str(deployment.get("protocol_mode", "")).strip().lower()
        == "fdm_async"
        and isinstance(state_history, dict)
        and state_history.get("enabled", False) is True
    )


def _configured_arm_command_mode(args) -> str:
    """Select the controller command mode from the deployment profile.

    LingBot FDM retains its nested action-mode setting. Hosted synchronous
    HTTP profiles (Pi or Cosmos) may negotiate protocol-v2 joint actions
    through ``expected_arm_action_space``. TCP replay retains its established
    EEF command path.
    """

    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else Path(get_package_share_directory("wuji_data_pipeline"))
        / "config"
        / "cosmos_protocol_v2.yaml"
    )
    config = load_config(str(config_path))
    deployment = section(config, "deployment")
    protocol_mode = str(
        deployment.get("protocol_mode", PI_PROTOCOL_MODE)
    ).strip().lower()
    if protocol_mode == "fdm_async":
        fdm_async = deployment.get("fdm_async", {})
        if not isinstance(fdm_async, dict):
            return "eef"
        action_mode = str(
            fdm_async.get("action_mode", "eef")
        ).strip().lower()
        if action_mode not in ("eef", "joint"):
            raise ValueError(
                "deployment.fdm_async.action_mode must be eef or joint"
            )
        return action_mode

    server = str(args.server or deployment.get("server", "")).strip()
    if (
        protocol_mode == PROTOCOL_V2_MODE
        and policy_transport_kind(server) == "http"
        and "arm_command_mode" in deployment
    ):
        command_mode = str(deployment["arm_command_mode"]).strip().lower()
        if command_mode not in ("eef", "joint"):
            raise ValueError(
                "deployment.arm_command_mode must be eef or joint"
            )
        return command_mode
    if (
        protocol_mode == PI_PROTOCOL_MODE
        and policy_transport_kind(server) == "http"
        and "expected_arm_action_space" in deployment
    ):
        action_space = normalize_arm_action_space(
            deployment["expected_arm_action_space"],
            "deployment.expected_arm_action_space",
        )
        return (
            "joint"
            if action_space == ARM_ACTION_SPACE_JOINT
            else "eef"
        )
    return "eef"


def _request_ordered_enable(
    node,
    active_hand: str,
    *,
    abort_event: threading.Event | None = None,
) -> tuple[bool, bool, bool, str]:
    """Enable WujiHand before Tianji can enter READY and dispatch actions.

    Returns ``(accepted, hands_enabled, arm_requested, message)``.  A failed
    Tianji request rolls the hand back immediately; ``hands_enabled`` remains
    true only when that rollback could not be confirmed.
    """

    hand_ok, hand_message = node.set_hands_enabled(
        active_hand, True, abort_event=abort_event
    )
    if not hand_ok:
        return False, False, False, hand_message

    request = SetBool.Request()
    request.data = True
    arm_ok, arm_message = node.call(
        "enable",
        request,
        timeout_s=3.0,
        abort_event=abort_event,
    )
    if arm_ok:
        return True, True, True, f"{hand_message}; {arm_message}"

    rollback_ok, rollback_message = node.set_hands_enabled(
        active_hand, False, abort_event=abort_event
    )
    return (
        False,
        not rollback_ok,
        True,
        f"{arm_message}; hand rollback: {rollback_message}",
    )


def _print_help() -> None:
    print(
        "\nControls:\n"
        "  r  guarded Tianji Recovery\n"
        "  a  Enable Tianji/Wuji and start cloud policy execution\n"
        "  x  Disable Wuji hands and request Tianji standby\n"
        "  e/q exit\n"
        "  Ctrl+C disable all hardware and stop the deployment graph\n"
        "  h  show this help\n"
    )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Interactive Tianji/Wuji cloud deployment session"
    )
    parser.add_argument(
        "--server",
        default=None,
        help=(
            "tcp://host:port or http(s)://host; omit to use "
            "deployment.server from config"
        ),
    )
    parser.add_argument(
        "--active-hand", choices=("both", "left", "right"), default="both"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-camera", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    if not sys.stdin.isatty():
        raise SystemExit("deployment_session requires docker exec -it")
    _validate_documented_environment()

    command = [
        "ros2",
        "launch",
        "wuji_data_pipeline",
        "deployment.launch.py",
        f"enable_camera:={'false' if args.no_camera else 'true'}",
        f"active_hand:={args.active_hand}",
    ]
    startup_handoff_enabled = _synchronous_http_startup_handoff_enabled(args)
    state_history_hand_first = _fdm_state_history_hand_first_enabled(args)
    arm_command_mode = _configured_arm_command_mode(args)
    command.append(
        "startup_handoff_gate_enabled:="
        + ("true" if startup_handoff_enabled else "false")
    )
    command.append(f"arm_command_mode:={arm_command_mode}")
    if args.server:
        command.append(f"server:={args.server}")
    if args.config:
        command.append(
            f"pipeline_config:={os.path.abspath(os.path.expanduser(args.config))}"
        )

    launch_process = None
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
            print(f"\n{source}; disabling and shutting down deployment...", flush=True)

    signal.signal(signal.SIGINT, request_exit)
    signal.signal(signal.SIGTERM, request_exit)
    signal.signal(signal.SIGHUP, request_exit)
    try:
        print("Starting:", " ".join(command))
        print(f"Deployment log: {DEPLOYMENT_LOG}")
        launch_process = _launch_graph(command, DEPLOYMENT_LOG)
        rclpy.init()
        rclpy_started = True
        node = SessionClient(
            node_name="wuji_cloud_deployment_session",
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
        print("Waiting for deployment ROS graph to become ready...", flush=True)
        missing_startup_nodes = _wait_for_deployment_nodes(
            node,
            args.active_hand,
            abort_event=shutdown_event,
        )
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
        if missing_startup_nodes:
            print(
                "ERROR: Deployment ROS graph startup timed out: "
                + "; ".join(
                    f"missing ROS node {name}"
                    for name in missing_startup_nodes
                ),
                flush=True,
            )
        else:
            print("Deployment ROS graph ready; press r for Recovery", flush=True)

        while running and launch_process.poll() is None:
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
                        "at their configured initial postures; press a to "
                        "start cloud deployment"
                    )
                    recovery_pending = False
                    recovery_failed = False
                    termios.tcflush(sys.stdin, termios.TCIFLUSH)
            if enable_pending and node.lifecycle in ENABLE_TERMINAL_STATES:
                if node.lifecycle == 2:
                    if hands_enabled:
                        print(
                            "CLOUD DEPLOYMENT STARTED: Tianji is READY; "
                            "WujiHand was enabled before Tianji Enable"
                        )
                    else:
                        hand_ok, hand_message = node.set_hands_enabled(
                            args.active_hand, True, abort_event=shutdown_event
                        )
                        if hand_ok:
                            hands_enabled = True
                            print(
                                "CLOUD DEPLOYMENT STARTED: Tianji is READY; "
                                + hand_message
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

            # TARGET_HOLD is intentionally excluded: a short cloud jitter may
            # recover by receiving a fresh chunk.  Terminal/standby states
            # remove hand actuator power immediately.
            if hands_enabled and node.lifecycle in (0, 3, 4, 5, 8, 9):
                hand_ok, hand_message = node.set_hands_enabled(
                    args.active_hand, False, abort_event=shutdown_event
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
                print("r received; checking deployment nodes and arm status...", flush=True)
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
                    print("ERROR: Cloud policy is not ready: " + server_message)
                    continue
                print("POLICY READY: " + server_message)
                if state_history_hand_first:
                    ok, hands_enabled, arm_requested, message = (
                        _request_ordered_enable(
                            node,
                            args.active_hand,
                            abort_event=shutdown_event,
                        )
                    )
                    print(("OK: " if ok else "ERROR: ") + message)
                    enable_pending = ok
                    if not ok and arm_requested:
                        standby_ok, standby_message = _request_tianji_standby(node)
                        print(
                            ("STANDBY: " if standby_ok else "CRITICAL: ")
                            + standby_message
                        )
                else:
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
                    args.active_hand, False, abort_event=shutdown_event
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
        hand_safe = False
        if node is not None:
            hand_safe, hand_message = node.set_hands_enabled(
                args.active_hand, False
            )
            print(("HAND SAFE: " if hand_safe else "WARNING: ") + hand_message)
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
        deployment_stopped = _shutdown_launch_graph(launch_process)
        if hand_safe and standby_confirmed and deployment_stopped:
            print("Deployment exited; Tianji/Wuji safe state confirmed.")
        else:
            print("Deployment exited with shutdown errors.")

    if not hand_safe or not standby_confirmed or not deployment_stopped:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
