"""One-command ROS 2 teleoperation + multi-episode recording cockpit."""

from __future__ import annotations

import argparse
import json
import os
import select
import signal
import subprocess
import sys
import termios
import threading
import time
import tty
from typing import Any, Optional

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Int8, String
from std_srvs.srv import SetBool, Trigger
from tf2_ros import Buffer, TransformListener
from wujihand_msgs.srv import SetEnabled as HandSetEnabled


LIFECYCLE_NAMES = {
    0: "INITIALIZING",
    1: "ENABLING",
    2: "READY",
    3: "DISABLED",
    4: "ENABLE_FAILED",
    5: "SDK_ERROR",
    6: "RECOVERING",
    7: "RECOVERY_READY",
    8: "RECOVERY_FAILED",
    9: "RECOVERY_PARTIAL",
    10: "TARGET_HOLD",
    11: "CLUTCH_DISCONNECTED",
}

DOCUMENTED_ROS_DOMAIN_ID = "112"
DOCUMENTED_RMW = "rmw_fastrtps_cpp"
DEFAULT_CHILD_LOG_PATH = "/tmp/wuji_record_session_children.log"
LAUNCH_SIGINT_TIMEOUT_S = 8.0
LAUNCH_SIGTERM_TIMEOUT_S = 2.0
LAUNCH_SIGKILL_TIMEOUT_S = 2.0
RECOVERY_TF_PAIRS = (
    ("left_chest", "tianji_left"),
    ("right_chest", "tianji_right"),
    ("left_chest", "left_arm"),
    ("right_chest", "right_arm"),
)
RECOVERY_REQUIRED_NODES = frozenset({"/tianji_arm_controller"})
RECOVERY_TERMINAL_STATES = {
    7: "RECOVERY COMPLETE: configured arm mode is ready; press a to enable",
    8: "RECOVERY FAILED: do not retry; inspect arm_status and restart the session",
    9: "RECOVERY PARTIAL: dual-arm recording cannot continue",
    5: "RECOVERY FAILED: Tianji SDK/hardware error",
}
ENABLE_TERMINAL_STATES = {
    2: "ENABLE COMPLETE: Tianji tracker teleoperation is READY",
    4: "ENABLE FAILED: Tianji requested standby",
    5: "ENABLE FAILED: Tianji SDK/hardware error",
}
HAND_RECOVERY_IDLE = 0
HAND_RECOVERY_MOVING = 1
HAND_RECOVERY_READY = 2
HAND_RECOVERY_FAILED = 3


def _fully_qualified_node_name(name: str, namespace: str) -> str:
    namespace = str(namespace).rstrip("/")
    return f"{namespace}/{name}" if namespace else f"/{name}"


def active_arm_tf_pairs(
    active_arm: str,
) -> tuple[tuple[str, str], ...]:
    sides = ("left", "right") if active_arm == "both" else (active_arm,)
    return tuple(
        pair
        for side in sides
        for pair in (
            (f"{side}_chest", f"tianji_{side}"),
            (f"{side}_chest", f"{side}_arm"),
        )
    )


def required_control_nodes(
    active_hand: str,
    active_arm: Optional[str] = None,
) -> set[str]:
    active_arm = active_hand if active_arm is None else active_arm
    nodes = {
        "/openvr_input",
        "/tianji_arm_controller",
        "/wuji_teleop_recorder",
        "/manus_data_publisher",
    }
    arm_sides = (
        ("left", "right") if active_arm == "both" else (active_arm,)
    )
    for side in arm_sides:
        nodes.update({
            f"/{side}_chest_base_tf",
            f"/{side}_chest_tf",
            f"/tianji_{side}_tf",
        })
    if active_hand in ("both", "left"):
        nodes.update({"/left_hand/wujihand_driver", "/wujihand_controller_left"})
    if active_hand in ("both", "right"):
        nodes.update({"/right_hand/wujihand_driver", "/wujihand_controller_right"})
    return nodes


def active_hand_sides(active_hand: str) -> tuple[str, ...]:
    if active_hand == "both":
        return ("left", "right")
    if active_hand in ("left", "right"):
        return (active_hand,)
    raise ValueError(f"unsupported active_hand: {active_hand!r}")


def required_hand_driver_nodes(active_hand: str) -> set[str]:
    return {
        f"/{side}_hand/wujihand_driver"
        for side in active_hand_sides(active_hand)
    }


def arm_status_errors(
    status: Any,
    sides: tuple[str, ...] = ("left", "right"),
) -> list[str]:
    """Validate the documented state=0/error-free Recovery precondition."""
    if not isinstance(status, dict):
        return ["arm_status response is not an object"]
    errors = []
    for side in sides:
        arm = status.get(side)
        if not isinstance(arm, dict):
            errors.append(f"{side}:missing status")
            continue
        state = int(arm.get("state", -1))
        error_code = int(arm.get("err_code", -1))
        if state != 0:
            errors.append(f"{side}:state={state} (required 0)")
        if error_code != 0:
            errors.append(f"{side}:err_code={error_code}")
        servo_faults = []
        for code in arm.get("servo_errors") or []:
            text = str(code).strip()
            if not text:
                continue
            try:
                is_zero = int(text, 16 if text.lower().startswith("0x") else 10) == 0
            except ValueError:
                is_zero = False
            if not is_zero:
                servo_faults.append(text)
        if servo_faults:
            errors.append(f"{side}:servo_faults={servo_faults}")
    return errors


def arm_standby_errors(status: Any) -> list[str]:
    """Return only errors that mean Tianji servo-off was not confirmed.

    Shutdown follows TIANJI_VIVE_TELEOP_RECORD.md section 5.9: the launch tree
    may be stopped only after both arms report ``state=0``.  Fault codes remain
    useful diagnostics, but they do not change whether servo-off was reached.
    """
    if not isinstance(status, dict):
        return ["arm_status response is not an object"]
    errors = []
    for side in ("left", "right"):
        arm = status.get(side)
        if not isinstance(arm, dict):
            errors.append(f"{side}:missing status")
            continue
        try:
            state = int(arm.get("state", -1))
        except (TypeError, ValueError):
            state = -1
        if state != 0:
            errors.append(f"{side}:state={state} (required 0)")
    return errors


class SessionClient(Node):
    def __init__(
        self,
        node_name: str = "wuji_record_session",
        include_recorder: bool = True,
        include_hands: bool = True,
        include_deployment: bool = False,
    ) -> None:
        super().__init__(node_name)
        self.lifecycle = None
        self.teleop_status = "waiting for Tianji controller"
        self.recorder_status = "waiting for recorder"
        self.hand_recovery_states = {"left": None, "right": None}
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.create_subscription(
            Int8,
            "/tianji_arm/lifecycle_state",
            self._lifecycle_callback,
            10,
        )
        self.create_subscription(
            String,
            "/tianji_arm/teleop_status",
            lambda msg: setattr(self, "teleop_status", msg.data),
            10,
        )
        if include_recorder:
            self.create_subscription(
                String,
                "/wuji_teleop_recorder/status_text",
                lambda msg: setattr(self, "recorder_status", msg.data),
                10,
            )
        # ``Node.clients`` is a read-only rclpy introspection property; keep
        # application-owned service clients under a distinct private name.
        self._service_clients = {
            "recover": self.create_client(
                Trigger, "/tianji_arm_controller/recover_to_init"
            ),
            "recover_left": self.create_client(
                Trigger, "/tianji_arm_controller/recover_left_to_init"
            ),
            "recover_right": self.create_client(
                Trigger, "/tianji_arm_controller/recover_right_to_init"
            ),
            "arm_status": self.create_client(
                Trigger, "/tianji_arm_controller/arm_status"
            ),
            "enable": self.create_client(
                SetBool, "/tianji_arm_controller/set_enabled"
            ),
        }
        if include_recorder:
            self._service_clients.update({
                "start": self.create_client(
                    Trigger, "/wuji_teleop_recorder/start"
                ),
                "toggle_capture": self.create_client(
                    Trigger, "/wuji_teleop_recorder/toggle_capture"
                ),
                "save": self.create_client(
                    Trigger, "/wuji_teleop_recorder/save"
                ),
                "discard": self.create_client(
                    Trigger, "/wuji_teleop_recorder/discard"
                ),
                "stop": self.create_client(
                    Trigger, "/wuji_teleop_recorder/stop"
                ),
                "status": self.create_client(
                    Trigger, "/wuji_teleop_recorder/status"
                ),
            })
        self._service_clients["clutch"] = self.create_client(
            Trigger, "/tianji_arm_controller/toggle_tracker_clutch"
        )
        if include_hands:
            for side in ("left", "right"):
                self.create_subscription(
                    Int8,
                    f"/{side}_hand/recovery_state",
                    lambda msg, s=side: self.hand_recovery_states.__setitem__(
                        s, int(msg.data)
                    ),
                    10,
                )
                self._service_clients[f"hand_enable_{side}"] = (
                    self.create_client(
                        HandSetEnabled,
                        f"/{side}_hand/set_enabled",
                    )
                )
                self._service_clients[f"hand_recover_{side}"] = (
                    self.create_client(
                        Trigger,
                        f"/{side}_hand/recover_to_initial",
                    )
                )
        if include_deployment:
            self._service_clients["deployment_ready"] = self.create_client(
                Trigger, "/wuji_deployment/ready"
            )

    def _lifecycle_callback(self, message: Int8) -> None:
        self.lifecycle = int(message.data)

    def call(
        self,
        name: str,
        request: Any,
        timeout_s: float = 3.0,
        service_wait_s: Optional[float] = None,
        abort_event: Optional[threading.Event] = None,
    ):
        client = self._service_clients[name]
        wait_s = timeout_s if service_wait_s is None else service_wait_s
        service_deadline = time.monotonic() + max(0.0, wait_s)
        while True:
            if abort_event is not None and abort_event.is_set():
                return False, f"service call interrupted by shutdown: {client.srv_name}"
            remaining = service_deadline - time.monotonic()
            if remaining <= 0.0:
                if client.wait_for_service(timeout_sec=0.0):
                    break
                return False, f"service unavailable: {client.srv_name}"
            if client.wait_for_service(timeout_sec=min(0.05, remaining)):
                break

        future = client.call_async(request)
        response_deadline = time.monotonic() + max(0.0, timeout_s)
        while not future.done():
            if abort_event is not None and abort_event.is_set():
                future.cancel()
                return False, f"service call interrupted by shutdown: {client.srv_name}"
            remaining = response_deadline - time.monotonic()
            if remaining <= 0.0:
                future.cancel()
                return False, f"service timeout: {client.srv_name}"
            rclpy.spin_once(self, timeout_sec=min(0.05, remaining))
        try:
            response = future.result()
        except Exception as exc:
            return False, str(exc)
        return bool(response.success), str(response.message)

    def visible_nodes(self) -> set[str]:
        return {
            _fully_qualified_node_name(name, namespace)
            for name, namespace in self.get_node_names_and_namespaces()
        }

    def missing_recovery_transforms(
        self,
        timeout_s: float = 3.0,
        abort_event: Optional[threading.Event] = None,
        tf_pairs: tuple[tuple[str, str], ...] = RECOVERY_TF_PAIRS,
    ) -> list[str]:
        """Actively spin until the documented TF chains arrive or time out."""
        deadline = time.monotonic() + timeout_s
        missing = list(tf_pairs)
        while missing and time.monotonic() < deadline:
            if abort_event is not None and abort_event.is_set():
                return ["shutdown requested"]
            rclpy.spin_once(self, timeout_sec=0.05)
            missing = [
                (target, source)
                for target, source in tf_pairs
                if not self._tf_buffer.can_transform(target, source, Time())
            ]
        return [f"missing TF {target} <- {source}" for target, source in missing]

    def recovery_preflight(
        self,
        active_hand: str,
        active_arm: str,
        abort_event: Optional[threading.Event] = None,
    ) -> list[str]:
        del active_arm
        return self.recovery_preflight_nodes(
            RECOVERY_REQUIRED_NODES | required_hand_driver_nodes(active_hand),
            abort_event=abort_event,
            tf_pairs=(),
        )

    def recovery_preflight_nodes(
        self,
        required_nodes: set[str] | frozenset[str],
        abort_event: Optional[threading.Event] = None,
        tf_pairs: tuple[tuple[str, str], ...] = (),
        status_sides: tuple[str, ...] = ("left", "right"),
    ) -> list[str]:
        """Check Tianji hardware readiness before joint-space Recovery."""
        missing = sorted(set(required_nodes) - self.visible_nodes())
        errors = [f"missing ROS node {name}" for name in missing]
        if errors:
            return errors
        errors.extend(self.missing_recovery_transforms(
            abort_event=abort_event,
            tf_pairs=tf_pairs,
        ))
        if errors:
            return errors
        ok, message = self.call(
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
        return arm_status_errors(status, sides=status_sides)

    def enable_preflight(
        self,
        active_hand: str,
        active_arm: str,
        abort_event: Optional[threading.Event] = None,
    ) -> list[str]:
        return self.enable_preflight_nodes(
            required_control_nodes(active_hand, active_arm),
            abort_event=abort_event,
            tf_pairs=active_arm_tf_pairs(active_arm),
        )

    def enable_preflight_nodes(
        self,
        required_nodes: set[str] | frozenset[str],
        abort_event: Optional[threading.Event] = None,
        tf_pairs: tuple[tuple[str, str], ...] = RECOVERY_TF_PAIRS,
    ) -> list[str]:
        """Check Tracker/input graph readiness immediately before Enable."""
        missing = sorted(set(required_nodes) - self.visible_nodes())
        errors = [f"missing ROS node {name}" for name in missing]
        if errors:
            return errors
        return self.missing_recovery_transforms(
            abort_event=abort_event,
            tf_pairs=tf_pairs,
        )

    def set_hands_enabled(
        self,
        active_hand: str,
        enabled: bool,
        abort_event: Optional[threading.Event] = None,
    ) -> tuple[bool, str]:
        """Enable/disable selected hands as one supervised session action."""
        completed = []
        errors = []
        for side in active_hand_sides(active_hand):
            request = HandSetEnabled.Request()
            request.finger_id = 255
            request.joint_id = 255
            request.enabled = enabled
            ok, message = self.call(
                f"hand_enable_{side}",
                request,
                timeout_s=3.0,
                service_wait_s=0.5,
                abort_event=abort_event,
            )
            if ok:
                completed.append(side)
            else:
                errors.append(f"{side}: {message}")

        if enabled and errors:
            # Do not leave one hand energized if a dual-hand enable only
            # partially succeeds.
            for side in completed:
                rollback = HandSetEnabled.Request()
                rollback.finger_id = 255
                rollback.joint_id = 255
                rollback.enabled = False
                self.call(
                    f"hand_enable_{side}",
                    rollback,
                    timeout_s=3.0,
                    service_wait_s=0.2,
                )

        if errors:
            action = "enable" if enabled else "disable"
            return False, f"WujiHand {action} failed: " + "; ".join(errors)
        sides = ",".join(completed)
        state = "enabled" if enabled else "disabled"
        return True, f"WujiHand {state}: {sides}"

    def start_hand_recovery(
        self,
        active_hand: str,
        abort_event: Optional[threading.Event] = None,
    ) -> tuple[bool, str]:
        """Start fixed-pose Recovery on every selected physical hand."""
        completed = []
        errors = []
        for side in active_hand_sides(active_hand):
            self.hand_recovery_states[side] = None
            ok, message = self.call(
                f"hand_recover_{side}",
                Trigger.Request(),
                timeout_s=3.0,
                service_wait_s=0.5,
                abort_event=abort_event,
            )
            if ok:
                completed.append(side)
            else:
                errors.append(f"{side}: {message}")

        if errors:
            # A dual-hand Recovery is atomic from the operator's point of
            # view. Cancel power on any side that already accepted Recovery.
            for side in completed:
                request = HandSetEnabled.Request()
                request.finger_id = 255
                request.joint_id = 255
                request.enabled = False
                self.call(
                    f"hand_enable_{side}",
                    request,
                    timeout_s=3.0,
                    service_wait_s=0.2,
                )
            return False, "WujiHand Recovery failed: " + "; ".join(errors)
        return True, "WujiHand Recovery started: " + ",".join(completed)

    def hand_recovery_result(self, active_hand: str) -> str:
        """Return pending, ready, or failed for the selected hands."""
        states = [
            self.hand_recovery_states[side]
            for side in active_hand_sides(active_hand)
        ]
        if any(state == HAND_RECOVERY_FAILED for state in states):
            return "failed"
        if states and all(state == HAND_RECOVERY_READY for state in states):
            return "ready"
        return "pending"


def _read_key(timeout_s: float = 0.1) -> Optional[str]:
    readable, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not readable:
        return None
    value = sys.stdin.read(1)
    # An interactive stdin reaching EOF means the controlling terminal went
    # away. Treat it exactly like an exit request instead of spinning forever.
    return value.lower() if value else "\x04"


def _launch_graph(
    command: list[str], child_log_path: str = DEFAULT_CHILD_LOG_PATH
) -> subprocess.Popen:
    """Launch ROS quietly in an owned process group.

    The interactive terminal is reserved for key acknowledgements and Tianji
    lifecycle transitions.  The launch graph still has a complete diagnostic
    log, but high-rate child output can no longer hide an ``r`` response.
    """
    log_stream = open(child_log_path, "w", encoding="utf-8")
    try:
        return subprocess.Popen(
            command,
            start_new_session=True,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
        )
    finally:
        log_stream.close()


def _signal_process_group(process: subprocess.Popen, sig: signal.Signals) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def _shutdown_launch_graph(process: Optional[subprocess.Popen]) -> bool:
    """Stop the complete ROS launch tree, escalating only after grace periods."""
    if process is None or process.poll() is not None:
        return True
    _signal_process_group(process, signal.SIGINT)
    try:
        process.wait(timeout=LAUNCH_SIGINT_TIMEOUT_S)
        return True
    except subprocess.TimeoutExpired:
        print(
            "ROS children did not exit after SIGINT; sending SIGTERM...",
            flush=True,
        )
    _signal_process_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=LAUNCH_SIGTERM_TIMEOUT_S)
        return True
    except subprocess.TimeoutExpired:
        print(
            "ROS children did not exit after SIGTERM; sending SIGKILL...",
            flush=True,
        )
    _signal_process_group(process, signal.SIGKILL)
    try:
        process.wait(timeout=LAUNCH_SIGKILL_TIMEOUT_S)
        return True
    except subprocess.TimeoutExpired:
        return False


def _request_tianji_standby(
    node: SessionClient,
    verify_timeout_s: float = 2.0,
) -> tuple[bool, str]:
    """Disable through the documented service and verify both arms at state=0."""
    request = SetBool.Request()
    request.data = False
    disable_ok, disable_message = node.call(
        "enable",
        request,
        timeout_s=8.0,
        service_wait_s=0.5,
    )

    deadline = time.monotonic() + max(0.0, verify_timeout_s)
    last_diagnostic = f"disable service: {disable_message}"
    while True:
        ok, message = node.call(
            "arm_status",
            Trigger.Request(),
            timeout_s=0.75,
            service_wait_s=0.25,
        )
        if ok:
            try:
                status = json.loads(message)
            except json.JSONDecodeError:
                last_diagnostic = f"arm_status returned invalid JSON: {message}"
            else:
                errors = arm_standby_errors(status)
                if not errors:
                    suffix = "" if disable_ok else f"; disable reply was: {disable_message}"
                    return True, "both Tianji arms confirmed state=0" + suffix
                last_diagnostic = "; ".join(errors)
        else:
            last_diagnostic = message

        if time.monotonic() >= deadline:
            break
        time.sleep(0.05)

    return False, f"Tianji standby not confirmed: {last_diagnostic}"


def _print_help() -> None:
    print(
        "\nControls:\n"
        "  r  guarded Tianji Recovery (moves one arm at a time)\n"
        "  a  Enable Tianji, then WujiHand, after Recovery\n"
        "  x  Disable WujiHand and request Tianji standby\n"
        "  s  pedal 1: start capture / finish capture and wait for decision\n"
        "  z  pedal 2: save the finished episode\n"
        "  c  pedal 3: disconnect / reconnect Tracker clutch (not while recording)\n"
        "  e/q exit and discard any active/pending unsaved episode\n"
        "  Ctrl+C same exit semantics; finish then use z when data must be kept\n"
        "  h  show this help\n"
        "\nRecovery/Enable follow TIANJI_VIVE_TELEOP_RECORD.md exactly.\n"
    )


def _validate_documented_environment() -> None:
    errors = []
    if os.environ.get("ROS_DOMAIN_ID") != DOCUMENTED_ROS_DOMAIN_ID:
        errors.append(
            f"ROS_DOMAIN_ID must be {DOCUMENTED_ROS_DOMAIN_ID}, got "
            f"{os.environ.get('ROS_DOMAIN_ID')!r}"
        )
    if os.environ.get("RMW_IMPLEMENTATION") != DOCUMENTED_RMW:
        errors.append(
            f"RMW_IMPLEMENTATION must be {DOCUMENTED_RMW}, got "
            f"{os.environ.get('RMW_IMPLEMENTATION')!r}"
        )
    if os.environ.get("CYCLONEDDS_URI"):
        errors.append("CYCLONEDDS_URI must be unset in the Tianji controller session")
    if errors:
        raise SystemExit("documented Tianji ROS environment is not active: " + "; ".join(errors))


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Launch MANUS + HTC + Tianji + Wuji and record LMDB episodes"
    )
    parser.add_argument("--config", default=None, help="pipeline.yaml override")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--handoff-ramp-sec", type=float, default=None)
    parser.add_argument(
        "--camera-transport",
        choices=("direct", "ros"),
        default="direct",
        help="Stage-D shared-memory path or legacy ROS image-topic fallback",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="record scalar data at 30Hz without launching/requiring cameras",
    )
    parser.add_argument(
        "--no-teleop",
        action="store_true",
        help="launch only the recorder (for mocked ROS topic testing)",
    )
    parser.add_argument(
        "--active-hand",
        choices=("both", "left", "right"),
        default="both",
        help=(
            "physical Wuji Hand side(s) connected; the inactive side is not "
            "launched and is zero-filled in the 54-D recording"
        ),
    )
    parser.add_argument(
        "--active-arm",
        choices=("both", "left", "right"),
        default="both",
        help="Tianji arm side(s) controlled by the Tracker",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    if not sys.stdin.isatty():
        raise SystemExit(
            "record_session requires an interactive terminal; start it with "
            "docker exec -it so Ctrl+C and the control keys are reliable"
        )
    if not args.no_teleop:
        _validate_documented_environment()
    command = [
        "ros2",
        "launch",
        "wuji_data_pipeline",
        "record.launch.py",
        f"enable_camera:={'false' if args.no_camera else 'true'}",
        f"enable_teleop:={'false' if args.no_teleop else 'true'}",
        f"active_hand:={args.active_hand}",
        f"active_arm:={args.active_arm}",
        f"camera_transport:={args.camera_transport}",
    ]
    if args.config:
        command.append(f"pipeline_config:={os.path.abspath(os.path.expanduser(args.config))}")
    if args.output_dir:
        command.append(f"output_dir:={args.output_dir}")
    if args.task_name:
        command.append(f"task_name:={args.task_name}")
    if args.handoff_ramp_sec is not None:
        if args.handoff_ramp_sec < 0.0:
            raise SystemExit("--handoff-ramp-sec cannot be negative")
        command.append(f"handoff_ramp_sec:={args.handoff_ramp_sec}")

    launch_process: Optional[subprocess.Popen] = None
    node: Optional[SessionClient] = None
    original_terminal = None
    rclpy_started = False
    running = True
    shutdown_event = threading.Event()
    exit_announced = False
    recovery_pending = False
    recovery_failed = False
    arm_recovery_terminal = None
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
        print(f"ROS child logs: {DEFAULT_CHILD_LOG_PATH}")
        launch_process = _launch_graph(command)
        rclpy.init()
        rclpy_started = True
        node = SessionClient()
        # rclpy installs its own handlers during init; the cockpit must remain
        # the single owner of ordered save/disable/launch shutdown.
        signal.signal(signal.SIGINT, request_exit)
        signal.signal(signal.SIGTERM, request_exit)
        signal.signal(signal.SIGHUP, request_exit)
        original_terminal = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        _print_help()
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
                    print(
                        "RECOVERY FAILED: WujiHand could not reach its "
                        "configured initial posture"
                    )
                    node.set_hands_enabled(args.active_hand, False)
                    _request_tianji_standby(node)
                    recovery_pending = False
                    recovery_failed = True
                    termios.tcflush(sys.stdin, termios.TCIFLUSH)
                elif arm_recovery_terminal == 7 and hand_result == "ready":
                    if args.active_arm == "right":
                        message = (
                            "RECOVERY COMPLETE: left arm is vertically parked; "
                            "right arm and selected WujiHand are at their "
                            "configured initial postures; press a to enable"
                        )
                    else:
                        message = (
                            "RECOVERY COMPLETE: configured arm mode and selected "
                            "WujiHand are at their initial postures; press a to enable"
                        )
                    print(message)
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
                        print(
                            "ENABLE COMPLETE: Tianji tracker teleoperation "
                            f"is READY; {hand_message}"
                        )
                    else:
                        print(
                            "ERROR: Arm reached READY but hand enable failed: "
                            + hand_message
                        )
                        print(
                            "Requesting Tianji standby because combined "
                            "Arm + Hand Enable did not complete...",
                            flush=True,
                        )
                        standby_ok, standby_message = (
                            _request_tianji_standby(node)
                        )
                        print(
                            ("STANDBY: " if standby_ok else "CRITICAL: ")
                            + standby_message
                        )
                else:
                    print(ENABLE_TERMINAL_STATES[node.lifecycle])
                    # Normally the hand is still disabled because it is
                    # enabled only after READY. Keep this idempotent cleanup
                    # for interrupted or future enable sequences.
                    node.set_hands_enabled(args.active_hand, False)
                enable_pending = False
                termios.tcflush(sys.stdin, termios.TCIFLUSH)
            key = _read_key()
            if key is None:
                continue
            if key in ("\x04", "e", "q"):
                if key in ("e", "q"):
                    ok, message = node.call(
                        "discard",
                        Trigger.Request(),
                        timeout_s=5.0,
                        service_wait_s=0.2,
                    )
                    print(("DISCARDED: " if ok else "WARNING: ") + message)
                request_exit()
            elif key == "h":
                _print_help()
            elif key == "r":
                if recovery_pending:
                    print("ERROR: Recovery is already running; wait for lifecycle 7 or 8")
                    continue
                if recovery_failed:
                    print("ERROR: Recovery failure is latched; inspect hardware and restart")
                    continue
                print(
                    "r received; checking Tianji controller and arm status...",
                    flush=True,
                )
                errors = node.recovery_preflight(
                    args.active_hand,
                    args.active_arm,
                    abort_event=shutdown_event,
                )
                if errors:
                    if shutdown_event.is_set():
                        continue
                    print("ERROR: Recovery preflight failed: " + "; ".join(errors))
                    continue
                print("\nRecovery requested. Clear the workspace and keep E-stop reachable.")
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
                    print(f"ERROR: Enable requires lifecycle 7/RECOVERY_READY, got {current}")
                    continue
                print(
                    "a received; checking Tracker, hand, recorder, and TF...",
                    flush=True,
                )
                errors = node.enable_preflight(
                    args.active_hand,
                    args.active_arm,
                    abort_event=shutdown_event,
                )
                if errors:
                    if not shutdown_event.is_set():
                        print(
                            "ERROR: Enable preflight failed: "
                            + "; ".join(errors)
                        )
                    continue
                if recovery_pending or node.hand_recovery_result(args.active_hand) != "ready":
                    print(
                        "ERROR: Enable requires WujiHand Recovery READY; "
                        "wait for the combined RECOVERY COMPLETE message"
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
                was_recovering = recovery_pending
                hand_ok, hand_message = node.set_hands_enabled(
                    args.active_hand,
                    False,
                    abort_event=shutdown_event,
                )
                print(("OK: " if hand_ok else "WARNING: ") + hand_message)
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
            elif key == "s":
                # Snapshot lifecycle once. TARGET_HOLD can clear while the
                # recorder status service is in flight; reading the live value
                # again produced the contradictory message "is READY;
                # requires READY" even though the key arrived during HOLD.
                capture_lifecycle = node.lifecycle
                if not args.no_teleop and capture_lifecycle != 2:
                    ok, message = node.call(
                        "status",
                        Trigger.Request(),
                        timeout_s=3.0,
                        service_wait_s=0.2,
                        abort_event=shutdown_event,
                    )
                    try:
                        capture_active = (
                            ok and json.loads(message).get("recording", False)
                        )
                    except json.JSONDecodeError:
                        capture_active = False
                    if not capture_active:
                        current = LIFECYCLE_NAMES.get(
                            capture_lifecycle, str(capture_lifecycle)
                        )
                        print(
                            "ERROR: Tianji lifecycle is "
                            f"{current}; starting capture requires READY"
                        )
                        continue
                ok, message = node.call(
                    "toggle_capture",
                    Trigger.Request(),
                    timeout_s=5.0,
                    service_wait_s=0.5,
                    abort_event=shutdown_event,
                )
                print(("CAPTURE: " if ok else "ERROR: ") + message)
            elif key == "z":
                ok, message = node.call(
                    "save",
                    Trigger.Request(),
                    timeout_s=60.0,
                    service_wait_s=0.5,
                    abort_event=shutdown_event,
                )
                print(("SAVED: " if ok else "ERROR: ") + message)
            elif key == "c":
                # Reconnect must remain available even if the recorder has
                # gone away. A *new* disconnect, however, is permitted only
                # after at least one pedal-1 capture has been finished.
                if node.lifecycle != 11:
                    ok, message = node.call(
                        "status",
                        Trigger.Request(),
                        timeout_s=3.0,
                        service_wait_s=0.2,
                        abort_event=shutdown_event,
                    )
                    if not ok:
                        print(
                            "ERROR: Cannot verify recorder state: " + message
                        )
                        continue
                    try:
                        recorder_state = json.loads(message)
                    except json.JSONDecodeError:
                        print("ERROR: Recorder returned invalid status")
                        continue
                    if recorder_state.get("recording"):
                        print(
                            "ERROR: Tracker disconnect is forbidden while "
                            "recording; press pedal 1 to finish capture first"
                        )
                        continue
                    if not recorder_state.get("capture_has_finished", False):
                        print(
                            "ERROR: Finish a capture with pedal 1 before "
                            "disconnecting Tracker control"
                        )
                        continue
                ok, message = node.call(
                    "clutch",
                    Trigger.Request(),
                    timeout_s=5.0,
                    service_wait_s=0.2,
                    abort_event=shutdown_event,
                )
                print(("CLUTCH: " if ok else "ERROR: ") + message)
    finally:
        if original_terminal is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original_terminal)
            except (OSError, termios.error):
                pass
        standby_confirmed = args.no_teleop
        if node is not None:
            if not args.no_teleop:
                print("Disabling WujiHand joints...", flush=True)
                try:
                    hand_ok, hand_message = node.set_hands_enabled(
                        args.active_hand,
                        False,
                    )
                except Exception as exc:
                    hand_ok = False
                    hand_message = f"WujiHand disable failed: {exc}"
                print(
                    ("HANDS DISABLED: " if hand_ok else "WARNING: ")
                    + hand_message,
                    flush=True,
                )
                print("Requesting Tianji standby and verifying state=0...", flush=True)
                try:
                    standby_confirmed, message = _request_tianji_standby(node)
                except Exception as exc:
                    standby_confirmed = False
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
                "ERROR: the ROS launch process group still did not exit after SIGKILL.",
                flush=True,
            )
        if standby_confirmed and children_stopped:
            detail = "Tianji state=0 confirmed; " if not args.no_teleop else ""
            print(f"Session exited; {detail}all session processes stopped.")
        else:
            print("Session exited with shutdown errors.", flush=True)

    if not standby_confirmed or not children_stopped:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
