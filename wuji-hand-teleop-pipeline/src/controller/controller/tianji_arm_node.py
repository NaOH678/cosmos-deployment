"""Tianji arm controller node: TF -> IK -> hardware."""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy,
)
from rclpy.utilities import remove_ros_args
from rclpy._rclpy_pybind11 import RCLError
from tf2_ros import Buffer, TransformListener
import tf2_ros
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Int8, String
from std_srvs.srv import SetBool, Trigger

from tianji_output import TianjiChestDriver
from .common import (
    ArmState,
    ROS2LoggerAdapter,
    get_default_qos,
    load_yaml_config,
    get_package_config_path,
)
from .performance_metrics import PerformanceWindow


# Topic names
LEFT_ARM_CMD_TOPIC = "/left_arm/joint_commands"
RIGHT_ARM_CMD_TOPIC = "/right_arm/joint_commands"
LEFT_ARM_STATE_TOPIC = "/left_arm/joint_states"
RIGHT_ARM_STATE_TOPIC = "/right_arm/joint_states"

# Latched lifecycle state on /tianji_arm/lifecycle_state (Int8). Replaces the
# old Bool /tianji_arm/teleop_ready — that signal was edge-only ("ready or
# not"), so a Monitor waiting on it couldn't tell the difference between
# "still initializing" and "_do_enable failed during the hardware transition".
# The Monitor's Stop button got stuck disabled in the second case. Lifecycle
# is the state machine the operator actually needs to see:
#
#   INITIALIZING  - node __init__ running; controller not yet armed
#   ENABLING      - _do_enable validating init and entering configured mode
#   READY         - _do_enable completed; teleop loop running
#   DISABLED      - _do_disable called (set_standby); waiting for re-enable
#   ENABLE_FAILED - _do_enable raised; operator should Stop or call retry
#   SDK_ERROR     - _on_sdk_error called; controller wedged
#   RECOVERING    - guarded state=1 single-arm recovery is running
#   RECOVERY_READY - both arms are at init in state=0; enable is permitted
#   RECOVERY_FAILED - recovery aborted and standby was requested
#   RECOVERY_PARTIAL - selected arm recovered; another arm still needs recovery
#   TARGET_HOLD - external command stream is briefly stale; last command is held
#   CLUTCH_DISCONNECTED - impedance stays active while Tracker input is paused
LIFECYCLE_INITIALIZING = 0
LIFECYCLE_ENABLING = 1
LIFECYCLE_READY = 2
LIFECYCLE_DISABLED = 3
LIFECYCLE_ENABLE_FAILED = 4
LIFECYCLE_SDK_ERROR = 5
LIFECYCLE_RECOVERING = 6
LIFECYCLE_RECOVERY_READY = 7
LIFECYCLE_RECOVERY_FAILED = 8
LIFECYCLE_RECOVERY_PARTIAL = 9
LIFECYCLE_TARGET_HOLD = 10
LIFECYCLE_CLUTCH_DISCONNECTED = 11

EXTERNAL_HANDOFF_IDLE = 'IDLE'
EXTERNAL_HANDOFF_WAITING_TARGET = 'WAITING_TARGET'
EXTERNAL_HANDOFF_ACTIVE = 'ACTIVE'
EXTERNAL_HANDOFF_COMPLETE = 'COMPLETE'
EXTERNAL_HANDOFF_FAILED = 'FAILED'
EXTERNAL_HANDOFF_STATES = {
    EXTERNAL_HANDOFF_IDLE,
    EXTERNAL_HANDOFF_WAITING_TARGET,
    EXTERNAL_HANDOFF_ACTIVE,
    EXTERNAL_HANDOFF_COMPLETE,
    EXTERNAL_HANDOFF_FAILED,
}

ALIGNMENT_JOINT_NAMES = (
    [f'Joint{i}_L' for i in range(1, 8)]
    + [f'Joint{i}_R' for i in range(1, 8)]
)


class ExternalCommandStale(RuntimeError):
    """A recoverable external-stream delay: hold and keep checking."""


def _observe_duration(node, name: str, started_ns: int) -> None:
    metrics = getattr(node, '_performance_metrics', None)
    if metrics is not None:
        metrics.observe(name, (time.perf_counter_ns() - started_ns) / 1e6)


def _observe_driver_timings(node, prefix: str, timings) -> None:
    metrics = getattr(node, '_performance_metrics', None)
    if metrics is None or not isinstance(timings, dict):
        return
    for name, value in timings.items():
        metrics.observe(f'{prefix}.{name}', value)


class TianjiArmControllerNode(Node):
    """Tianji arm controller node."""

    def __init__(self, robot_ip: str = '192.168.1.190'):
        super().__init__("tianji_arm_controller")

        # Hardware motion always requires an explicit service call. The node
        # may connect on startup, but defaults to servo-off standby.
        self.declare_parameter('auto_enable', False)
        # ``both`` preserves the production dual-arm behavior.  ``left`` and
        # ``right`` are explicit development modes: only the selected arm may
        # recover, enable, receive commands, or consume Tracker TF.
        self.declare_parameter('active_arm', 'both')
        # Control loop / state-publish rates (Hz, two timers decoupled):
        # - control_rate: TF -> IK control path. 120Hz matches the upper bound
        #   of HTC tracker / MANUS / wuji_glove input streams; higher adds no
        #   new command information.
        # - state_publish_rate: /{side}_arm/joint_states publish rate.
        #   Downstream inference + data capture want finer state feedback;
        #   500Hz is stable under Python + DDS.
        # No rebuild required to change rates — launch / param yaml overrides work.
        self.declare_parameter('control_rate', 120.0)
        self.declare_parameter('state_publish_rate', 500.0)
        # Handoff after hardware enable: hold the measured reference pose for
        # `handoff_hold_sec`, then run a `handoff_ramp_sec` smoothstep ramp from
        # that snapshot to the live IK target.
        # Set either to 0.0 to disable that phase. Conservative defaults keep
        # the tracker handoff visibly slow.
        self.declare_parameter('handoff_hold_sec', 1.0)
        self.declare_parameter('handoff_ramp_sec', 5.0)
        self.declare_parameter('enable_init_tolerance_deg', 3.0)
        self.declare_parameter('recovery_velocity_ratio', 10)
        self.declare_parameter('recovery_acceleration_ratio', 10)
        self.declare_parameter('recovery_max_speed_deg_s', 5.0)
        self.declare_parameter('recovery_max_accel_deg_s2', 10.0)
        self.declare_parameter('recovery_control_period_sec', 0.02)
        self.declare_parameter('recovery_tracking_error_deg', 2.0)
        self.declare_parameter('recovery_command_lead_deg', 0.5)
        self.declare_parameter('recovery_reverse_motion_deg', 0.2)
        self.declare_parameter('recovery_arrival_tolerance_deg', 0.5)
        self.declare_parameter('recovery_encoder_agreement_deg', 1.0)
        self.declare_parameter('recovery_hold_sec', 2.0)
        self.declare_parameter('recovery_stall_timeout_sec', 5.0)
        self.declare_parameter('impedance_stability_sec', 3.0)
        self.declare_parameter('impedance_max_drift_deg', 1.0)
        self.declare_parameter('impedance_velocity_ratio', 30)
        self.declare_parameter('impedance_acceleration_ratio', 30)
        self.declare_parameter(
            'impedance_k', [14.0, 14.0, 14.0, 10.5, 5.6, 5.6, 5.6])
        self.declare_parameter(
            'impedance_d', [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
        # Tracker motion is applied as a delta around a neutral captured at
        # every enable. This avoids requiring the operator to physically match
        # the robot's Cartesian init pose before impedance can be enabled.
        self.declare_parameter('teleop_position_scale', 1.0)
        # The existing tracker path remains the default.  ``external`` is an
        # explicitly selected deployment/replay source; the two sources never
        # write hardware commands concurrently.
        self.declare_parameter('control_source', 'tracker')
        self.declare_parameter('external_command_mode', 'eef')
        # Hardware actuation stays impedance by default. Position mode is
        # accepted only for an explicit external replay/deployment source;
        # its command may be EEF->IK or a recorded joint target.
        self.declare_parameter('arm_hardware_mode', 'impedance')
        # Deployment may explicitly coordinate a fixed-target startup ramp.
        # Keep this false by default so tracker, TCP replay, and asynchronous
        # deployment protocols preserve their established handoff behavior.
        self.declare_parameter('external_handoff_gate_enabled', False)
        # Replay-only absolute startup target. Recovery itself still ends at
        # the configured init; Enable may then move to the episode first qpos.
        self.declare_parameter('replay_first_joint_targets_json', '{}')
        self.declare_parameter('external_command_timeout_sec', 0.25)
        self.declare_parameter('external_standby_timeout_sec', 1.0)
        self.declare_parameter('eef_publish_rate', 60.0)
        # Stage-A diagnostics are passive: one compact JSON snapshot per
        # second, with no change to control rates, mapping, IK, or guards.
        self.declare_parameter('performance_metrics_enabled', True)
        self.declare_parameter('performance_publish_rate', 1.0)
        # Stage-B GUI status is built only from in-memory controller/feedback
        # caches.  It must never trigger an additional Marvin SDK query.
        self.declare_parameter('status_snapshot_rate', 2.0)
        auto_enable = bool(self.get_parameter('auto_enable').value)
        if auto_enable:
            self.get_logger().error(
                "auto_enable ignored: guarded recover_to_init must complete "
                "before every hardware enable")
            auto_enable = False
        self._control_rate_hz = float(self.get_parameter('control_rate').value)
        self._state_publish_rate_hz = float(self.get_parameter('state_publish_rate').value)
        self._handoff_hold_sec = max(0.0, float(self.get_parameter('handoff_hold_sec').value))
        self._handoff_ramp_sec = max(0.0, float(self.get_parameter('handoff_ramp_sec').value))
        self._enable_init_tolerance_deg = float(
            self.get_parameter('enable_init_tolerance_deg').value)
        self._recovery_velocity_ratio = int(
            self.get_parameter('recovery_velocity_ratio').value)
        self._recovery_acceleration_ratio = int(
            self.get_parameter('recovery_acceleration_ratio').value)
        self._recovery_max_speed_deg_s = float(
            self.get_parameter('recovery_max_speed_deg_s').value)
        self._recovery_max_accel_deg_s2 = float(
            self.get_parameter('recovery_max_accel_deg_s2').value)
        self._recovery_control_period_sec = float(
            self.get_parameter('recovery_control_period_sec').value)
        self._recovery_tracking_error_deg = float(
            self.get_parameter('recovery_tracking_error_deg').value)
        self._recovery_command_lead_deg = float(
            self.get_parameter('recovery_command_lead_deg').value)
        self._recovery_reverse_motion_deg = float(
            self.get_parameter('recovery_reverse_motion_deg').value)
        self._recovery_arrival_tolerance_deg = float(
            self.get_parameter('recovery_arrival_tolerance_deg').value)
        self._recovery_encoder_agreement_deg = float(
            self.get_parameter('recovery_encoder_agreement_deg').value)
        self._recovery_hold_sec = float(
            self.get_parameter('recovery_hold_sec').value)
        self._recovery_stall_timeout_sec = float(
            self.get_parameter('recovery_stall_timeout_sec').value)
        self._impedance_stability_sec = float(
            self.get_parameter('impedance_stability_sec').value)
        self._impedance_max_drift_deg = float(
            self.get_parameter('impedance_max_drift_deg').value)
        self._impedance_velocity_ratio = int(
            self.get_parameter('impedance_velocity_ratio').value)
        self._impedance_acceleration_ratio = int(
            self.get_parameter('impedance_acceleration_ratio').value)
        self._impedance_k = np.asarray(
            self.get_parameter('impedance_k').value, dtype=float)
        self._impedance_d = np.asarray(
            self.get_parameter('impedance_d').value, dtype=float)
        self._teleop_position_scale = float(
            self.get_parameter('teleop_position_scale').value)
        self._control_source = str(
            self.get_parameter('control_source').value).strip().lower()
        self._external_command_mode = str(
            self.get_parameter('external_command_mode').value
        ).strip().lower()
        self._arm_hardware_mode = str(
            self.get_parameter('arm_hardware_mode').value
        ).strip().lower()
        self._external_handoff_gate_enabled = bool(
            self.get_parameter('external_handoff_gate_enabled').value)
        replay_targets_raw = str(
            self.get_parameter('replay_first_joint_targets_json').value
        ).strip()
        try:
            replay_targets_value = json.loads(replay_targets_raw or '{}')
        except json.JSONDecodeError as exc:
            raise ValueError(
                'replay_first_joint_targets_json must be valid JSON') from exc
        if not isinstance(replay_targets_value, dict):
            raise ValueError(
                'replay_first_joint_targets_json must contain a JSON object')
        self._replay_first_joint_targets: dict[str, list[float]] = {}
        for side, values in replay_targets_value.items():
            if side not in ('left', 'right'):
                raise ValueError(f'unknown Replay first-qpos side {side!r}')
            vector = np.asarray(values, dtype=float)
            if vector.shape != (7,) or not np.all(np.isfinite(vector)):
                raise ValueError(
                    f'Replay first-qpos {side} must contain 7 finite degrees')
            self._replay_first_joint_targets[side] = vector.tolist()
        self._external_command_timeout_sec = float(
            self.get_parameter('external_command_timeout_sec').value)
        self._external_standby_timeout_sec = float(
            self.get_parameter('external_standby_timeout_sec').value)
        self._eef_publish_rate_hz = float(
            self.get_parameter('eef_publish_rate').value)
        self._performance_metrics_enabled = bool(
            self.get_parameter('performance_metrics_enabled').value)
        self._performance_publish_rate_hz = float(
            self.get_parameter('performance_publish_rate').value)
        self._status_snapshot_rate_hz = float(
            self.get_parameter('status_snapshot_rate').value)
        self._active_arm_mode = str(
            self.get_parameter('active_arm').value).strip().lower()
        active_arm_config = {
            'both': (('left', 'right'), ('A', 'B')),
            'left': (('left',), ('A',)),
            'right': (('right',), ('B',)),
        }
        if self._active_arm_mode not in active_arm_config:
            raise ValueError(
                "active_arm must be both, left, or right, got "
                f"{self._active_arm_mode!r}")
        self._configured_sides, self._configured_sdk_arms = (
            active_arm_config[self._active_arm_mode])
        if self._control_source not in ('tracker', 'external'):
            raise ValueError(
                f"control_source must be tracker or external, got {self._control_source!r}")
        if self._external_command_mode not in ('eef', 'joint'):
            raise ValueError(
                "external_command_mode must be eef or joint, got "
                f"{self._external_command_mode!r}")
        if self._arm_hardware_mode not in ('impedance', 'position'):
            raise ValueError(
                "arm_hardware_mode must be impedance or position, got "
                f"{self._arm_hardware_mode!r}")
        if (
            self._arm_hardware_mode == 'position'
            and self._control_source != 'external'
        ):
            raise ValueError(
                "arm_hardware_mode=position is restricted to "
                "control_source=external")
        if self._replay_first_joint_targets:
            if self._control_source != 'external':
                raise ValueError(
                    'Replay first-qpos targets require external control')
            configured = set(self._configured_sides)
            provided = set(self._replay_first_joint_targets)
            if configured != provided:
                raise ValueError(
                    'Replay first-qpos sides must exactly match active_arm: '
                    f'missing={sorted(configured - provided)}, '
                    f'extra={sorted(provided - configured)}')
        if self._control_rate_hz <= 0.0 or self._control_rate_hz > 120.0:
            raise ValueError(
                f"control_rate must be in (0, 120], got {self._control_rate_hz}")
        if self._state_publish_rate_hz <= 0.0:
            raise ValueError(
                f"state_publish_rate must be > 0, got {self._state_publish_rate_hz}")
        if not 1 <= self._recovery_velocity_ratio <= 10:
            raise ValueError("recovery_velocity_ratio must be in [1, 10]")
        if not 1 <= self._recovery_acceleration_ratio <= 10:
            raise ValueError("recovery_acceleration_ratio must be in [1, 10]")
        if min(
            self._enable_init_tolerance_deg,
            self._recovery_max_speed_deg_s,
            self._recovery_max_accel_deg_s2,
            self._recovery_control_period_sec,
            self._recovery_tracking_error_deg,
            self._recovery_command_lead_deg,
            self._recovery_reverse_motion_deg,
            self._recovery_arrival_tolerance_deg,
            self._recovery_encoder_agreement_deg,
            self._recovery_stall_timeout_sec,
            self._impedance_stability_sec,
            self._impedance_max_drift_deg,
        ) <= 0.0 or self._recovery_hold_sec < 0.0:
            raise ValueError("Recovery and impedance safety limits are invalid")
        if self._recovery_command_lead_deg >= self._recovery_tracking_error_deg:
            raise ValueError(
                "recovery_command_lead_deg must be smaller than "
                "recovery_tracking_error_deg")
        if not 1 <= self._impedance_velocity_ratio <= 100:
            raise ValueError("impedance_velocity_ratio must be in [1, 100]")
        if not 1 <= self._impedance_acceleration_ratio <= 100:
            raise ValueError("impedance_acceleration_ratio must be in [1, 100]")
        if (self._impedance_k.shape != (7,)
                or not np.all(np.isfinite(self._impedance_k))
                or np.any(self._impedance_k < 0.0)
                or np.any(self._impedance_k > 22.0)):
            raise ValueError("impedance_k must contain 7 finite values in [0, 22]")
        if (self._impedance_d.shape != (7,)
                or not np.all(np.isfinite(self._impedance_d))
                or np.any(self._impedance_d <= 0.0)
                or np.any(self._impedance_d > 1.0)):
            raise ValueError("impedance_d must contain 7 finite values in (0, 1]")
        if self._teleop_position_scale <= 0.0:
            raise ValueError("teleop_position_scale must be > 0")
        if (
            self._external_command_timeout_sec <= 0.0
            or self._external_standby_timeout_sec
            <= self._external_command_timeout_sec
            or self._eef_publish_rate_hz <= 0.0
            or self._performance_publish_rate_hz <= 0.0
            or self._status_snapshot_rate_hz <= 0.0
        ):
            raise ValueError(
                "external timeout, publish rates, or status snapshot rate "
                "parameters are invalid")
        self._arm_state = ArmState.IMPEDANCE
        self._logger_adapter = ROS2LoggerAdapter(self.get_logger())
        self._log_counter = 0
        self._state_error_count = 0  # consecutive get_current_joints failures
        self._last_lifecycle_state = LIFECYCLE_INITIALIZING
        self._last_teleop_status = 'INITIALIZING: controller is starting'
        self._status_snapshot_sequence = 0
        self._last_state_feedback_at = 0.0
        self._last_arm_state_codes = (None, None)
        self._last_arm_error_codes = (None, None)
        self._last_detailed_arm_status = None
        self._last_detailed_arm_status_at = 0.0
        self._performance_metrics = (
            PerformanceWindow()
            if self._performance_metrics_enabled
            else None
        )

        # Initialize the controller.
        self.get_logger().info(f"Connecting to robot {robot_ip}...")
        self.controller = TianjiChestDriver(robot_ip=robot_ip, logger=self._logger_adapter)

        # STANDBY / ACTIVE state management.
        self._arm_enabled = False
        self._enable_lock = threading.Lock()
        self._enable_cancel_event = threading.Event()
        self._enable_in_progress = threading.Event()
        self._enable_thread: Optional[threading.Thread] = None
        self._recovery_cancel_event = threading.Event()
        self._recovery_in_progress = threading.Event()
        self._recovery_thread: Optional[threading.Thread] = None
        self._recovery_complete = False

        # Pose and joint caches.
        self.left_pose = None
        self.right_pose = None
        self.left_y_axis = None
        self.right_y_axis = None
        self._last_left_state_joints = None
        self._last_right_state_joints = None

        # Handoff ramp state — populated at the end of _do_enable, consumed by
        # _teleop_control. `_handoff_start_at is None` means no ramp pending.
        self._handoff_start_at: Optional[float] = None
        self._handoff_start_left: Optional[list] = None
        self._handoff_start_right: Optional[list] = None
        # External deployment has a two-phase handoff.  Enable snapshots the
        # measured joints, but the ramp clock starts only after the first
        # complete external target has produced a valid IK solution.  That
        # first joint target is frozen for the whole ramp so a policy clock
        # cannot move the endpoint underneath the controller.
        self._external_handoff_state = EXTERNAL_HANDOFF_IDLE
        self._external_handoff_target_left: Optional[list] = None
        self._external_handoff_target_right: Optional[list] = None
        self._external_handoff_completed_at = 0.0
        self._neutral_tracker_left: Optional[np.ndarray] = None
        self._neutral_tracker_right: Optional[np.ndarray] = None
        self._neutral_arm_left: Optional[np.ndarray] = None
        self._neutral_arm_right: Optional[np.ndarray] = None
        self._neutral_robot_left: Optional[np.ndarray] = None
        self._neutral_robot_right: Optional[np.ndarray] = None
        self._target_hold_reason: Optional[str] = None
        self._target_hold_log_at = 0.0
        self._tracker_connected = True
        self._external_lock = threading.Lock()
        self._external_pose = {'left': None, 'right': None}
        self._external_pose_at = {'left': 0.0, 'right': 0.0}
        self._external_zsp = {'left': None, 'right': None}
        self._external_joint = {'left': None, 'right': None}
        self._external_joint_at = {'left': 0.0, 'right': 0.0}
        self._external_stream_started = False
        self._external_hold_pose = {'left': None, 'right': None}
        self._external_hold_joint = {'left': None, 'right': None}

        # Always enter standby first. set_standby failure (e.g. state=100
        # under e-stop) must not block node startup, otherwise launch
        # respawn=True crash-loops until the fault is cleared externally.
        try:
            self.controller.set_standby()
            self.get_logger().info("Arm connected; servos in standby")
        except Exception as exc:
            self.get_logger().error(
                f"set_standby failed at startup (node will continue, "
                f"awaiting operator intervention): {exc}"
            )

        # If auto_enable is ever restored, schedule it outside __init__ so
        # shutdown always has a fully constructed node. The current hard lock
        # forces auto_enable false.
        self._auto_enable_timer = None
        if auto_enable:
            self._auto_enable_timer = self.create_timer(0.1, self._auto_enable_once)

        # TF listener.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        qos = get_default_qos()
        alignment_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Publishers.
        self.left_cmd_pub = self.create_publisher(JointState, LEFT_ARM_CMD_TOPIC, qos)
        self.right_cmd_pub = self.create_publisher(JointState, RIGHT_ARM_CMD_TOPIC, qos)
        self.left_state_pub = self.create_publisher(JointState, LEFT_ARM_STATE_TOPIC, qos)
        self.right_state_pub = self.create_publisher(JointState, RIGHT_ARM_STATE_TOPIC, qos)
        self._alignment_state_pub = self.create_publisher(
            JointState, '/tianji_alignment/joint_states', alignment_qos)
        self._alignment_target_pub = self.create_publisher(
            JointState, '/tianji_alignment/target_joint_states', alignment_qos)

        # zsp_para and pose publishers.
        self.left_zsp_para_pub = self.create_publisher(Float64MultiArray, '/left_arm/zsp_para', qos)
        self.right_zsp_para_pub = self.create_publisher(Float64MultiArray, '/right_arm/zsp_para', qos)
        self.left_ee_pose_pub = self.create_publisher(Float64MultiArray, '/left_arm/ee_pose', qos)
        self.right_ee_pose_pub = self.create_publisher(Float64MultiArray, '/right_arm/ee_pose', qos)
        self.left_target_pose_pub = self.create_publisher(
            PoseStamped, '/left_arm/target_ee_pose', qos)
        self.right_target_pose_pub = self.create_publisher(
            PoseStamped, '/right_arm/target_ee_pose', qos)
        self.left_actual_pose_pub = self.create_publisher(
            PoseStamped, '/left_arm/actual_ee_pose', qos)
        self.right_actual_pose_pub = self.create_publisher(
            PoseStamped, '/right_arm/actual_ee_pose', qos)

        # External deployment topics are subscribed in all modes so a launch
        # can be inspected before enable.  Callbacks discard data unless the
        # node was explicitly launched with control_source=external.
        self.create_subscription(
            PoseStamped, '/left_arm/external_target_pose',
            lambda msg: self._external_pose_callback('left', msg), qos)
        self.create_subscription(
            PoseStamped, '/right_arm/external_target_pose',
            lambda msg: self._external_pose_callback('right', msg), qos)
        self.create_subscription(
            Float64MultiArray, '/left_arm/external_zsp',
            lambda msg: self._external_zsp_callback('left', msg), qos)
        self.create_subscription(
            Float64MultiArray, '/right_arm/external_zsp',
            lambda msg: self._external_zsp_callback('right', msg), qos)
        self.create_subscription(
            JointState, '/left_arm/external_joint_target',
            lambda msg: self._external_joint_callback('left', msg), qos)
        self.create_subscription(
            JointState, '/right_arm/external_joint_target',
            lambda msg: self._external_joint_callback('right', msg), qos)

        # Latched lifecycle state. TRANSIENT_LOCAL so late-joining
        # subscribers (e.g. the Monitor UI opened after the controller is up)
        # immediately receive the last value, rather than racing the next
        # state transition. Monitor uses this to gate the Stop Teleop button
        # on real controller state (see the LIFECYCLE_* constants at top).
        lifecycle_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._lifecycle_pub = self.create_publisher(
            Int8, '/tianji_arm/lifecycle_state', lifecycle_qos)
        self._teleop_status_pub = self.create_publisher(
            String, '/tianji_arm/teleop_status', lifecycle_qos)
        self._handoff_state_pub = self.create_publisher(
            String, '/tianji_arm/handoff_state', lifecycle_qos)
        self._status_snapshot_pub = self.create_publisher(
            String, '/tianji_arm/status_snapshot', lifecycle_qos)
        self._performance_metrics_pub = (
            self.create_publisher(
                String, '/tianji_arm/performance_metrics', qos)
            if self._performance_metrics is not None
            else None
        )
        self._publish_lifecycle(LIFECYCLE_INITIALIZING)
        self._publish_teleop_status('INITIALIZING: controller is starting')
        self._publish_external_handoff_state(EXTERNAL_HANDOFF_IDLE)

        # Services.
        self.create_service(SetBool, '~/set_enabled', self._set_enabled_callback)
        self.create_service(
            Trigger, '~/recover_to_init', self._recover_all_callback)
        self.create_service(
            Trigger, '~/recover_left_to_init', self._recover_left_callback)
        self.create_service(
            Trigger, '~/recover_right_to_init', self._recover_right_callback)
        self.create_service(
            Trigger, '~/toggle_tracker_clutch',
            self._toggle_tracker_clutch_callback)
        self.create_service(SetBool, '/tianji_arm/set_arm_state', self._set_arm_state_callback)
        self.create_service(SetBool, '~/left/brake', self._left_brake_cb)
        self.create_service(SetBool, '~/right/brake', self._right_brake_cb)
        self.create_service(Trigger, '~/arm_status', self._arm_status_cb)
        self.create_service(SetBool, '~/left/clear_error', self._left_clear_error_cb)
        self.create_service(SetBool, '~/right/clear_error', self._right_clear_error_cb)

        # Control loop + state-publish: two timers (rates read from ROS2 params; see top of __init__).
        self.create_timer(1.0 / self._control_rate_hz, self._control_loop)
        self.create_timer(1.0 / self._state_publish_rate_hz, self._state_publish_loop)
        self.create_timer(1.0 / self._eef_publish_rate_hz, self._eef_publish_loop)
        self.create_timer(
            1.0 / self._status_snapshot_rate_hz,
            self._publish_status_snapshot,
        )
        if self._performance_metrics is not None:
            self.create_timer(
                1.0 / self._performance_publish_rate_hz,
                self._publish_performance_metrics,
            )

        self.get_logger().info(
            f"Init complete: control_rate={self._control_rate_hz:.1f}Hz, "
            f"state_publish_rate={self._state_publish_rate_hz:.1f}Hz, "
            f"control_source={self._control_source}, "
            f"arm_hardware_mode={self._arm_hardware_mode}, "
            f"active_arm={self._active_arm_mode}, "
            f"performance_metrics={self._performance_metrics_enabled}"
        )

    # -------------------- Latched lifecycle state --------------------

    def _publish_lifecycle(self, state: int):
        """Push the latched lifecycle state. Safe to call before the
        publisher exists; silently no-ops in that case."""
        self._last_lifecycle_state = int(state)
        pub = getattr(self, '_lifecycle_pub', None)
        if pub is None:
            return
        msg = Int8()
        msg.data = int(state)
        try:
            pub.publish(msg)
        except Exception as exc:
            self.get_logger().debug(f"lifecycle_state publish skipped: {exc}")

    def _publish_teleop_status(self, status: str):
        self._last_teleop_status = str(status)
        publisher = getattr(self, '_teleop_status_pub', None)
        if publisher is None:
            return
        message = String()
        message.data = self._last_teleop_status
        try:
            publisher.publish(message)
        except Exception as exc:
            self.get_logger().debug(f"teleop_status publish skipped: {exc}")

    def _external_handoff_payload(self) -> dict:
        """Return the cached external-handoff state without SDK access."""

        state = str(getattr(
            self, '_external_handoff_state', EXTERNAL_HANDOFF_IDLE))
        started_at = getattr(self, '_handoff_start_at', None)
        now = time.monotonic()
        elapsed = (
            max(0.0, now - float(started_at))
            if started_at is not None
            else 0.0
        )
        total = max(
            0.0,
            float(getattr(self, '_handoff_hold_sec', 0.0))
            + float(getattr(self, '_handoff_ramp_sec', 0.0)),
        )
        progress = 0.0
        if state == EXTERNAL_HANDOFF_COMPLETE:
            progress = 1.0
        elif state == EXTERNAL_HANDOFF_ACTIVE and total > 0.0:
            progress = min(1.0, elapsed / total)
        return {
            'schema_version': 1,
            'state': state,
            'progress': progress,
            'elapsed_sec': elapsed,
            'hold_sec': float(getattr(self, '_handoff_hold_sec', 0.0)),
            'ramp_sec': float(getattr(self, '_handoff_ramp_sec', 0.0)),
            'completed_at_monotonic': float(getattr(
                self, '_external_handoff_completed_at', 0.0)),
            'control_source': str(getattr(
                self, '_control_source', 'tracker')),
            'enabled': bool(getattr(
                self, '_external_handoff_gate_enabled', False)),
        }

    def _publish_external_handoff_state(self, state: Optional[str] = None):
        """Publish an external-only, latched handoff state transition."""

        if state is not None:
            normalized = str(state).strip().upper()
            if normalized not in EXTERNAL_HANDOFF_STATES:
                raise ValueError(f'invalid external handoff state: {state!r}')
            self._external_handoff_state = normalized
        publisher = getattr(self, '_handoff_state_pub', None)
        if publisher is None:
            return
        message = String()
        message.data = json.dumps(
            self._external_handoff_payload(),
            separators=(',', ':'),
            sort_keys=True,
        )
        try:
            publisher.publish(message)
        except Exception as exc:
            self.get_logger().debug(f"handoff_state publish skipped: {exc}")

    def _build_status_snapshot(self) -> dict:
        """Return a GUI-safe status snapshot without touching the SDK."""
        now_monotonic = time.monotonic()
        feedback_at = float(
            getattr(self, '_last_state_feedback_at', 0.0) or 0.0)
        feedback_age = (
            max(0.0, now_monotonic - feedback_at)
            if feedback_at > 0.0
            else None
        )
        diagnostic_at = float(
            getattr(self, '_last_detailed_arm_status_at', 0.0) or 0.0)
        diagnostic_age = (
            max(0.0, now_monotonic - diagnostic_at)
            if diagnostic_at > 0.0
            else None
        )
        state_codes = getattr(
            self, '_last_arm_state_codes', (None, None))
        error_codes = getattr(
            self, '_last_arm_error_codes', (None, None))
        left_joints = getattr(self, '_last_left_state_joints', None)
        right_joints = getattr(self, '_last_right_state_joints', None)

        def arm_snapshot(index: int, joints) -> dict:
            state = state_codes[index]
            error = error_codes[index]
            return {
                'feedback_available': joints is not None,
                'state': None if state is None else int(state),
                'err_code': None if error is None else int(error),
            }

        arm_state = getattr(self, '_arm_state', ArmState.IMPEDANCE)
        arm_state_value = (
            arm_state.value if isinstance(arm_state, ArmState)
            else str(arm_state)
        )
        lifecycle = int(getattr(
            self, '_last_lifecycle_state', LIFECYCLE_INITIALIZING))
        return {
            'schema_version': 1,
            'sequence': int(
                getattr(self, '_status_snapshot_sequence', 0)),
            'published_at_unix_s': time.time(),
            'lifecycle_state': lifecycle,
            'teleop_status': str(getattr(
                self, '_last_teleop_status',
                'INITIALIZING: controller is starting',
            )),
            'arm_enabled': bool(getattr(self, '_arm_enabled', False)),
            'arm_state': arm_state_value,
            'active_arm': str(getattr(
                self, '_active_arm_mode', 'both')),
            'control_source': str(getattr(
                self, '_control_source', 'tracker')),
            'tracker_connected': bool(getattr(
                self, '_tracker_connected', True)),
            'motion_in_progress': bool(self._motion_in_progress()),
            'target_hold': getattr(
                self, '_target_hold_reason', None) is not None,
            'target_hold_reason': getattr(
                self, '_target_hold_reason', None),
            'handoff': TianjiArmControllerNode._external_handoff_payload(
                self),
            'sdk_fault_latched': lifecycle == LIFECYCLE_SDK_ERROR,
            'feedback_age_s': feedback_age,
            'arms': {
                'left': arm_snapshot(0, left_joints),
                'right': arm_snapshot(1, right_joints),
            },
            'detailed_hardware_status': {
                'available': getattr(
                    self, '_last_detailed_arm_status', None) is not None,
                'age_s': diagnostic_age,
            },
        }

    def _publish_status_snapshot(self):
        """Publish cached status for observers; never perform SDK I/O."""
        publisher = getattr(self, '_status_snapshot_pub', None)
        if publisher is None:
            return
        started_ns = time.perf_counter_ns()
        self._status_snapshot_sequence = int(
            getattr(self, '_status_snapshot_sequence', 0)) + 1
        message = String()
        message.data = json.dumps(
            self._build_status_snapshot(),
            separators=(',', ':'),
            sort_keys=True,
        )
        try:
            publisher.publish(message)
            metrics = getattr(self, '_performance_metrics', None)
            if metrics is not None:
                metrics.increment('status_snapshot.publishes')
        finally:
            _observe_duration(
                self, 'status_snapshot.duration_ms', started_ns)

    def _publish_performance_metrics(self):
        metrics = getattr(self, '_performance_metrics', None)
        publisher = getattr(self, '_performance_metrics_pub', None)
        if metrics is None or publisher is None:
            return
        context = {
            'arm_enabled': bool(getattr(self, '_arm_enabled', False)),
            'tracker_connected': bool(
                getattr(self, '_tracker_connected', True)),
            'motion_in_progress': bool(self._motion_in_progress()),
            'target_hold': getattr(self, '_target_hold_reason', None) is not None,
            'lifecycle_state': int(
                getattr(self, '_last_lifecycle_state',
                        LIFECYCLE_INITIALIZING)),
            'active_arm': self._active_arm_mode,
            'control_source': self._control_source,
            'target_rates_hz': {
                'control': self._control_rate_hz,
                'state': self._state_publish_rate_hz,
                'eef': self._eef_publish_rate_hz,
            },
        }
        message = String()
        message.data = json.dumps(
            metrics.snapshot(context),
            separators=(',', ':'),
            sort_keys=True,
        )
        publisher.publish(message)

    # -------------------- SDK exception handling --------------------

    def _on_sdk_error(self, context: str, error: Exception):
        """On SDK exception, auto-disable the arm to avoid a deadlocked state.
        Respawn recovers automatically."""
        metrics = getattr(self, '_performance_metrics', None)
        if metrics is not None:
            metrics.increment('sdk.errors')
        self.get_logger().error(f"[{context}] SDK exception: {error}; auto-disabling arm")
        # An SDK failure invalidates every motion authorization derived from
        # earlier feedback. Wake any guarded transition before touching the
        # controller again so it cannot continue with stale state.
        self._enable_cancel_event.set()
        self._recovery_cancel_event.set()
        self._arm_enabled = False
        self._recovery_complete = False
        self._tracker_connected = True
        self._arm_state = ArmState.IMPEDANCE
        self.left_pose = None
        self.right_pose = None
        self._reset_control_targets()
        self._publish_lifecycle(LIFECYCLE_SDK_ERROR)
        self._publish_teleop_status(f'ERROR: {context}: {error}')
        try:
            self.controller.set_standby()
        except Exception as e2:
            # Error-recovery path; do not re-raise (would clobber the outer
            # timer callback). Must still log: SDK is already faulted and
            # set_standby also failing means a real hardware fault.
            self.get_logger().error(
                f"[{context}] set_standby also failed during recovery: {e2} — "
                f"SDK may be wedged, awaiting respawn"
            )

    # -------------------- Enable / disable --------------------

    def _control_sides(self):
        """Configured ROS-side arm names; default keeps legacy dual-arm tests."""
        return tuple(getattr(self, '_configured_sides', ('left', 'right')))

    def _control_sdk_arms(self):
        """Configured SDK arm names (A=left, B=right)."""
        return tuple(getattr(self, '_configured_sdk_arms', ('A', 'B')))

    def _control_indices(self):
        return tuple(0 if side == 'left' else 1 for side in self._control_sides())

    def _init_pose_errors(self):
        feedback, init_errors = self.controller.get_verified_init_feedback(
            encoder_agreement_deg=self._recovery_encoder_agreement_deg,
            arms=list(self._control_sdk_arms()),
            active_arm=getattr(self, '_active_arm_mode', 'both'),
        )
        current = np.asarray(feedback, dtype=float)
        errors = np.asarray(init_errors, dtype=float)
        if current.shape != (2, 7) or errors.shape != (2,):
            raise RuntimeError(
                "invalid dual-arm feedback during verified init-pose check")
        if not np.all(np.isfinite(current)) or not np.all(np.isfinite(errors)):
            raise RuntimeError(
                "non-finite dual-arm feedback during verified init-pose check")
        return current, errors

    def _do_enable(self):
        """Bring the configured arm(s) from STANDBY to control-ready.

        Sequence:
            1. Require a completed recover_to_init cycle and verify state=0.
            2. Enter configured hardware mode (state=3 impedance, or explicit
               external replay state=1 position).
            3. In impedance mode, hold and reject any startup drift.
            4. Snapshot measured joints for the guarded tracker handoff.

        Idempotent under _enable_lock: if already enabled, returns
        immediately without re-publishing lifecycle.
        """
        with self._enable_lock:
            active_sides = self._control_sides()
            active_arms = self._control_sdk_arms()
            hardware_mode = getattr(
                self, '_arm_hardware_mode', 'impedance')
            expected_active_state = 1 if hardware_mode == 'position' else 3
            if self._arm_enabled:
                left_state, right_state = self.controller.get_arm_states_only()
                states = {'left': left_state, 'right': right_state}
                if all(
                    states[side] == (
                        expected_active_state if side in active_sides else 0)
                    for side in ('left', 'right')
                ):
                    return
                self._arm_enabled = False
                self.get_logger().error(
                    "Software enable flag disagrees with hardware state "
                    f"(left={left_state}, right={right_state}); running full "
                    "safe enable sequence"
                )
            self._publish_lifecycle(LIFECYCLE_ENABLING)
            try:
                if not self._recovery_complete:
                    raise RuntimeError(
                        "recover_to_init has not completed; hardware enable is refused")
                left_state, right_state = self.controller.get_arm_states_only()
                if left_state != 0 or right_state != 0:
                    raise RuntimeError(
                        "enable requires all arms in standby before the selected "
                        f"arm enters {hardware_mode} mode "
                        f"(left={left_state}, right={right_state})")

                _, side_errors = self._init_pose_errors()
                active_error = float(np.max(side_errors[list(self._control_indices())]))
                if active_error > self._enable_init_tolerance_deg:
                    raise RuntimeError(
                        "recovered pose no longer satisfies enable tolerance; "
                        f"left={side_errors[0]:.1f} deg, "
                        f"right={side_errors[1]:.1f} deg, "
                        f"limit={self._enable_init_tolerance_deg:.1f} deg")

                # Absolute LMDB Replay starts from the episode's measured
                # first qpos. Recovery above remains unchanged and still ends
                # at the configured init; this guarded move belongs only to
                # the operator-triggered Enable sequence.
                self._move_to_replay_first_qpos()

                if hardware_mode == 'position':
                    self.controller.set_position_mode(
                        velRatio=self._recovery_velocity_ratio,
                        AccRatio=self._recovery_acceleration_ratio,
                        arms=list(active_arms),
                        replay=True,
                    )
                else:
                    self.controller.set_active(
                        mode='joint',
                        K=self._impedance_k.tolist(),
                        D=self._impedance_d.tolist(),
                        velRatio=self._impedance_velocity_ratio,
                        AccRatio=self._impedance_acceleration_ratio,
                        arms=list(active_arms),
                    )
                    self._monitor_impedance_stability()
                self._arm_handoff_snapshot()
                if getattr(self, '_control_source', 'tracker') == 'external':
                    self._capture_external_neutral()
                else:
                    self._capture_teleop_neutral()
            except Exception as exc:
                self._arm_enabled = False
                self._recovery_complete = False
                self._arm_state = ArmState.IMPEDANCE
                self.left_pose = None
                self.right_pose = None
                self._reset_control_targets()
                self._publish_lifecycle(LIFECYCLE_ENABLE_FAILED)
                try:
                    self.controller.set_standby()
                except Exception as standby_exc:
                    raise RuntimeError(
                        f"enable failed ({exc}); CRITICAL: standby recovery also "
                        f"failed ({standby_exc})") from exc
                raise

            self._arm_enabled = True
            self._tracker_connected = True
            self._arm_state = (
                ArmState.POSITION
                if hardware_mode == 'position'
                else ArmState.IMPEDANCE
            )
            self._publish_lifecycle(LIFECYCLE_READY)
            source = getattr(self, '_control_source', 'tracker')
            self._publish_teleop_status(
                f'READY: {source} control active in {hardware_mode} mode '
                f'({getattr(self, "_active_arm_mode", "both")})')

    def _move_to_replay_first_qpos(self):
        """Guardedly move from configured Recovery init to Replay frame zero."""
        targets = getattr(self, '_replay_first_joint_targets', {})
        if not targets:
            return
        if getattr(self, '_control_source', 'tracker') != 'external':
            raise RuntimeError(
                'Replay first-qpos transition requires external control')

        side_to_arm = {'left': 'A', 'right': 'B'}
        for side in self._control_sides():
            if self._enable_cancel_event.is_set():
                raise RuntimeError('Replay first-qpos transition cancelled')
            target = targets.get(side)
            if target is None:
                raise RuntimeError(
                    f'Replay first-qpos target is missing for {side}')
            arm = side_to_arm[side]
            self.get_logger().info(
                f'Absolute Replay startup: moving {side} from configured '
                'Recovery init to episode first qpos')
            self.controller.recover_arm_to_init(
                arm,
                target_joints=target,
                target_label='episode first qpos',
                vel_ratio=self._recovery_velocity_ratio,
                acc_ratio=self._recovery_acceleration_ratio,
                max_speed_deg_s=self._recovery_max_speed_deg_s,
                max_accel_deg_s2=self._recovery_max_accel_deg_s2,
                dt=self._recovery_control_period_sec,
                tracking_error_deg=self._recovery_tracking_error_deg,
                command_lead_deg=self._recovery_command_lead_deg,
                reverse_motion_deg=self._recovery_reverse_motion_deg,
                arrival_tolerance_deg=self._recovery_arrival_tolerance_deg,
                encoder_agreement_deg=self._recovery_encoder_agreement_deg,
                hold_sec=self._recovery_hold_sec,
                stall_timeout_sec=self._recovery_stall_timeout_sec,
                cancel_event=self._enable_cancel_event,
            )
            self.controller.set_standby(arms=[arm])
        self.controller.set_standby()
        self.get_logger().info(
            'Absolute Replay startup: episode first qpos reached; '
            'enabling original absolute trajectory')

    def _monitor_impedance_stability(self):
        """Hold state=3 without tracker commands and reject startup drift."""
        baseline, _ = self._init_pose_errors()
        active_sides = self._control_sides()
        active_indices = list(self._control_indices())
        deadline = time.monotonic() + self._impedance_stability_sec
        while time.monotonic() < deadline:
            if self._enable_cancel_event.is_set():
                raise RuntimeError("impedance enable cancelled by stop request")
            left_state, right_state = self.controller.get_arm_states_only()
            states = {'left': left_state, 'right': right_state}
            if any(
                states[side] != (3 if side in active_sides else 0)
                for side in ('left', 'right')
            ):
                raise RuntimeError(
                    "impedance stability check lost the configured arm state "
                    f"(left={left_state}, right={right_state})")
            current_left, current_right = self.controller.get_current_joints()
            current = np.asarray([current_left, current_right], dtype=float)
            if current.shape != (2, 7) or not np.all(np.isfinite(current)):
                raise RuntimeError("invalid feedback during impedance stability check")
            drift = np.max(np.abs(current - baseline), axis=1)
            if float(np.max(drift[active_indices])) > self._impedance_max_drift_deg:
                raise RuntimeError(
                    "impedance startup drift exceeded safety limit: "
                    f"left={drift[0]:.2f} deg, right={drift[1]:.2f} deg, "
                    f"limit={self._impedance_max_drift_deg:.2f} deg")
            if self._enable_cancel_event.wait(0.05):
                raise RuntimeError("impedance enable cancelled by stop request")

    def _do_recover_to_init(self, arms):
        """Recover/park arms one at a time, ending with all arms in state=0.

        Right-arm-only operation has one additional guarded step: arm A is
        first moved to the configured vertical parking pose. Arm B still
        recovers to the normal Tracker init pose and is the only arm enabled
        afterwards.
        """
        normalized = self.controller._normalize_arms(arms)
        init_left, init_right = self.controller.get_init_joints(
            getattr(self, '_active_arm_mode', 'both'))
        init_targets = {'A': init_left, 'B': init_right}
        recovery_plan = [
            (arm, init_targets[arm], 'init')
            for arm in normalized
        ]
        if (
            getattr(self, '_active_arm_mode', 'both') == 'right'
            and normalized == ['B']
        ):
            recovery_plan.insert(
                0,
                (
                    'A',
                    self.controller.get_park_joints('A'),
                    'vertical park',
                ),
            )
        with self._enable_lock:
            self._arm_enabled = False
            self._recovery_complete = False
            self._tracker_connected = True
            self.left_pose = None
            self.right_pose = None
            self._reset_control_targets()
            self._publish_lifecycle(LIFECYCLE_RECOVERING)
            try:
                self.controller.set_standby()
                for arm, target_joints, target_label in recovery_plan:
                    if self._recovery_cancel_event.is_set():
                        raise RuntimeError("recovery cancelled by stop request")
                    recovery_kwargs = dict(
                        vel_ratio=self._recovery_velocity_ratio,
                        acc_ratio=self._recovery_acceleration_ratio,
                        max_speed_deg_s=self._recovery_max_speed_deg_s,
                        max_accel_deg_s2=self._recovery_max_accel_deg_s2,
                        dt=self._recovery_control_period_sec,
                        tracking_error_deg=self._recovery_tracking_error_deg,
                        command_lead_deg=self._recovery_command_lead_deg,
                        reverse_motion_deg=self._recovery_reverse_motion_deg,
                        arrival_tolerance_deg=self._recovery_arrival_tolerance_deg,
                        encoder_agreement_deg=(
                            self._recovery_encoder_agreement_deg),
                        hold_sec=self._recovery_hold_sec,
                        stall_timeout_sec=self._recovery_stall_timeout_sec,
                        cancel_event=self._recovery_cancel_event,
                    )
                    if target_joints is not None:
                        recovery_kwargs.update(
                            target_joints=target_joints,
                            target_label=target_label,
                        )
                    self.controller.recover_arm_to_init(
                        arm,
                        **recovery_kwargs,
                    )
                    self.controller.set_standby(arms=[arm])

                self.controller.set_standby()
                _, side_errors = self._init_pose_errors()
                selected_indices = [0 if arm == 'A' else 1 for arm in normalized]
                selected_error = float(np.max(side_errors[selected_indices]))
                if selected_error > self._enable_init_tolerance_deg:
                    raise RuntimeError(
                        "selected recovery did not reach enable tolerance: "
                        f"left={side_errors[0]:.2f} deg, "
                        f"right={side_errors[1]:.2f} deg, "
                        f"limit={self._enable_init_tolerance_deg:.2f} deg")
                ready_indices = list(self._control_indices())
                self._recovery_complete = (
                    float(np.max(side_errors[ready_indices]))
                    <= self._enable_init_tolerance_deg)
                self._publish_lifecycle(
                    LIFECYCLE_RECOVERY_READY
                    if self._recovery_complete
                    else LIFECYCLE_RECOVERY_PARTIAL)
                return side_errors.tolist(), self._recovery_complete
            except Exception:
                self._recovery_complete = False
                self._publish_lifecycle(LIFECYCLE_RECOVERY_FAILED)
                try:
                    self.controller.set_standby()
                except Exception as standby_exc:
                    self.get_logger().error(
                        f"CRITICAL: recovery failed and standby also failed: "
                        f"{standby_exc}")
                raise

    def _do_disable(self):
        """Standby (servo-off, keep connection). Idempotent under _enable_lock."""
        # Set this before waiting for _enable_lock so an in-progress transition
        # can abort and release the lock promptly.
        self._enable_cancel_event.set()
        self._recovery_cancel_event.set()
        with self._enable_lock:
            self._arm_enabled = False
            self._recovery_complete = False
            self._tracker_connected = True
            self._arm_state = ArmState.IMPEDANCE
            self.left_pose = None
            self.right_pose = None
            self._reset_control_targets()
            self._publish_lifecycle(LIFECYCLE_DISABLED)
            self._publish_teleop_status('DISABLED: arms are in standby')
            self.controller.set_standby()

    def _auto_enable_once(self):
        """One-shot timer callback: run _do_enable asynchronously. A failure
        here leaves the node up (operator can retry via ~/set_enabled) but
        publishes LIFECYCLE_ENABLE_FAILED so the Monitor can re-enable Stop
        Teleop instead of waiting forever for a READY signal that never
        comes."""
        # Cancel + destroy the timer first so a slow _do_enable doesn't get
        # re-entered by the next tick.
        if self._auto_enable_timer is not None:
            try:
                self._auto_enable_timer.cancel()
                self.destroy_timer(self._auto_enable_timer)
            except Exception:
                pass
            self._auto_enable_timer = None
        if self._start_enable_worker():
            self.get_logger().info("Arm auto-enable sequence started")

    def _start_enable_worker(self) -> bool:
        """Start the cancellable enable sequence without blocking ROS services."""
        if self._motion_in_progress():
            return False
        self._enable_cancel_event.clear()
        self._enable_in_progress.set()
        self._enable_thread = threading.Thread(
            target=self._enable_worker,
            name='tianji-safe-enable',
            daemon=True,
        )
        self._enable_thread.start()
        return True

    def _enable_worker(self):
        try:
            self._do_enable()
            self.get_logger().info(
                "Arms enabled in "
                f"{getattr(self, '_arm_hardware_mode', 'impedance')} mode")
        except Exception as exc:
            self.get_logger().error(
                f"Enable failed and recovery was requested: {exc}. "
                "Inspect the hardware state before retrying."
            )
            self._publish_lifecycle(LIFECYCLE_ENABLE_FAILED)
        finally:
            self._enable_in_progress.clear()

    def _motion_in_progress(self):
        return self._enable_in_progress.is_set() or self._recovery_in_progress.is_set()

    def _start_recovery_worker(self, arms) -> bool:
        if self._motion_in_progress():
            return False
        self._recovery_cancel_event.clear()
        self._enable_cancel_event.clear()
        self._recovery_in_progress.set()
        self._recovery_thread = threading.Thread(
            target=self._recovery_worker,
            args=(tuple(arms),),
            name='tianji-guarded-recovery',
            daemon=True,
        )
        self._recovery_thread.start()
        return True

    def _recovery_worker(self, arms):
        try:
            errors, all_ready = self._do_recover_to_init(arms)
            if all_ready:
                if getattr(self, '_active_arm_mode', 'both') == 'right':
                    self.get_logger().info(
                        "Recovery complete; left arm is in vertical park and "
                        "right arm is at init, both in standby "
                        f"(right init error={errors[1]:.2f} deg)")
                else:
                    self.get_logger().info(
                        "Recovery complete; configured arm(s) are in standby "
                        "at init "
                        f"(left={errors[0]:.2f} deg, "
                        f"right={errors[1]:.2f} deg)")
            else:
                self.get_logger().info(
                    "Selected-arm recovery complete; another arm still needs "
                    f"recovery (left={errors[0]:.2f} deg, "
                    f"right={errors[1]:.2f} deg)")
        except Exception as exc:
            self.get_logger().error(
                f"Recovery failed and standby was requested: {exc}")
        finally:
            self._recovery_in_progress.clear()

    def _arm_handoff_snapshot(self):
        """Snapshot current hardware joints to seed the handoff ramp.

        The arm must already be near the configured reference before the mode
        transition. Snapshot its measured pose here, then _teleop_control
        blends from that snapshot to live IK over handoff_hold_sec plus
        handoff_ramp_sec.
        """
        try:
            left, right = self.controller.get_current_joints()
            joints = {'left': left, 'right': right}
            active_sides = self._control_sides()
            for side in ('left', 'right'):
                value = joints[side] if side in active_sides else None
                setattr(
                    self,
                    f'_handoff_start_{side}',
                    list(value) if value is not None else None,
                )
            missing = [
                side for side in active_sides
                if getattr(self, f'_handoff_start_{side}') is None
            ]
            if missing:
                raise RuntimeError(
                    "missing joint feedback for configured arm(s): "
                    + ', '.join(missing))
            if (
                getattr(self, '_control_source', 'tracker') == 'external'
                and getattr(
                    self, '_external_handoff_gate_enabled', False)
            ):
                self._handoff_start_at = None
                self._external_handoff_target_left = None
                self._external_handoff_target_right = None
                self._external_handoff_completed_at = 0.0
                self._publish_external_handoff_state(
                    EXTERNAL_HANDOFF_WAITING_TARGET)
                self.get_logger().info(
                    "External handoff waiting for the first valid arm target")
                return
            if self._handoff_hold_sec <= 0.0 and self._handoff_ramp_sec <= 0.0:
                self._handoff_start_at = None
                return
            self._handoff_start_at = time.monotonic()
            self.get_logger().info(
                f"Handoff ramp armed: hold={self._handoff_hold_sec:.2f}s, "
                f"ramp={self._handoff_ramp_sec:.2f}s"
            )
        except Exception as exc:
            self._handoff_start_at = None
            raise RuntimeError(f"handoff snapshot failed: {exc}") from exc

    def _reset_control_targets(self):
        self._handoff_start_at = None
        self._handoff_start_left = None
        self._handoff_start_right = None
        self._external_handoff_target_left = None
        self._external_handoff_target_right = None
        self._external_handoff_completed_at = 0.0
        self._publish_external_handoff_state(EXTERNAL_HANDOFF_IDLE)
        self._neutral_tracker_left = None
        self._neutral_tracker_right = None
        self._neutral_arm_left = None
        self._neutral_arm_right = None
        self._neutral_robot_left = None
        self._neutral_robot_right = None
        self._target_hold_reason = None
        self._target_hold_log_at = 0.0
        if hasattr(self, '_external_lock'):
            with self._external_lock:
                self._external_pose = {'left': None, 'right': None}
                self._external_pose_at = {'left': 0.0, 'right': 0.0}
                self._external_zsp = {'left': None, 'right': None}
                self._external_joint = {'left': None, 'right': None}
                self._external_joint_at = {'left': 0.0, 'right': 0.0}
                self._external_stream_started = False
                self._external_hold_pose = {'left': None, 'right': None}
                self._external_hold_joint = {'left': None, 'right': None}

    def _capture_teleop_neutral(self):
        """Anchor current tracker poses to the measured robot EE poses."""
        active_sides = self._control_sides()
        frames = {}
        arm_frames = {}
        for side in active_sides:
            frames[side] = self._lookup_transform(
                f'{side}_chest', f'tianji_{side}')
            arm_frames[side] = self._lookup_transform(
                f'{side}_chest', f'{side}_arm')
        missing = []
        for side in active_sides:
            if frames[side] is None:
                missing.append(f'tianji_{side}')
            if arm_frames[side] is None:
                missing.append(f'{side}_arm')
        if missing:
            raise RuntimeError(
                f"cannot capture teleop neutral; missing TF: {', '.join(missing)}")

        robot_frames = {}
        for side in active_sides:
            joints = getattr(self, f'_handoff_start_{side}')
            if joints is None:
                raise RuntimeError(
                    f"cannot capture {side} teleop neutral without joint snapshot")
            robot_frames[side] = self._robot_fk_matrix(side, joints)
        for side in active_sides:
            setattr(self, f'_neutral_tracker_{side}', frames[side].copy())
            setattr(self, f'_neutral_arm_{side}', arm_frames[side].copy())
            setattr(self, f'_neutral_robot_{side}', robot_frames[side].copy())
        self.get_logger().info(
            "Teleop neutral captured for " + ','.join(active_sides))

    def _capture_external_neutral(self):
        """Start external control by holding the measured robot pose.

        Unlike tracker mode, external deployment does not need OpenVR TFs.
        No network command is trusted during enable; a fresh paired command is
        required after READY before the stream is considered active.
        """
        active_sides = self._control_sides()
        hold = {'left': None, 'right': None}
        for side in active_sides:
            joints = getattr(self, f'_handoff_start_{side}')
            if joints is None:
                raise RuntimeError(
                    f"cannot capture {side} external neutral without joint snapshot")
            hold[side] = self._robot_fk_matrix(side, joints)
        with self._external_lock:
            self._external_pose = {'left': None, 'right': None}
            self._external_pose_at = {'left': 0.0, 'right': 0.0}
            self._external_joint = {'left': None, 'right': None}
            self._external_joint_at = {'left': 0.0, 'right': 0.0}
            self._external_stream_started = False
            self._external_hold_pose = {
                side: None if matrix is None else matrix.copy()
                for side, matrix in hold.items()
            }
            self._external_hold_joint = {
                side: (
                    list(getattr(self, f'_handoff_start_{side}'))
                    if side in active_sides else None
                )
                for side in ('left', 'right')
            }
        self.left_pose = (
            self._matrix_to_pose(hold['left'])
            if hold['left'] is not None else None)
        self.right_pose = (
            self._matrix_to_pose(hold['right'])
            if hold['right'] is not None else None)
        self.get_logger().info(
            "External neutral captured; holding measured pose until fresh "
            "paired deployment commands arrive")

    def _robot_fk_matrix(self, side: str, joints_deg) -> np.ndarray:
        kine = self.controller.kine_left if side == 'left' else self.controller.kine_right
        serial = 0 if side == 'left' else 1
        matrix = kine.fk(robot_serial=serial, joints=list(joints_deg))
        if matrix is False or matrix is None:
            raise RuntimeError(f"{side} FK failed during neutral capture")
        result = np.asarray(matrix, dtype=float)
        if result.shape != (4, 4) or not np.all(np.isfinite(result)):
            raise RuntimeError(f"{side} FK returned invalid neutral pose")
        result = result.copy()
        result[:3, 3] /= 1000.0
        return result

    def _map_tracker_target(self, side: str, current: np.ndarray) -> np.ndarray:
        """Apply tracker motion in the official per-side IK base frame.

        Both ``current`` and ``neutral`` are already expressed in
        ``{side}_chest`` by TF, while ``robot_neutral`` comes from the matching
        Tianji FK model.  No additional left/right alignment belongs here.
        """
        neutral = getattr(self, f'_neutral_tracker_{side}')
        robot_neutral = getattr(self, f'_neutral_robot_{side}')
        if neutral is None or robot_neutral is None:
            raise RuntimeError(f"{side} teleop neutral is not initialized")
        target = robot_neutral.copy()
        delta_position = current[:3, 3] - neutral[:3, 3]
        target[:3, 3] += delta_position * self._teleop_position_scale
        delta_rotation = current[:3, :3] @ neutral[:3, :3].T
        target[:3, :3] = delta_rotation @ robot_neutral[:3, :3]
        target[:3, :3] = R.from_matrix(target[:3, :3]).as_matrix()
        return target

    def _map_upper_arm_direction(self, side: str,
                                 current: np.ndarray) -> np.ndarray:
        neutral = getattr(self, f'_neutral_arm_{side}')
        if neutral is None:
            raise RuntimeError(f"{side} upper-arm neutral is not initialized")
        base = np.array(
            [0.0, -1.0, -0.5] if side == 'left'
            else [0.0, 1.0, -0.5],
            dtype=float)
        delta_rotation = current[:3, :3] @ neutral[:3, :3].T
        direction = delta_rotation @ base
        norm = float(np.linalg.norm(direction))
        if not np.all(np.isfinite(direction)) or norm < 1e-9:
            raise RuntimeError(f"{side} upper-arm direction is invalid")
        return direction / norm

    def _enter_target_hold(self, reason: str):
        """Hold the last command while waiting for a fresh external target."""
        reason = str(reason)
        now = time.monotonic()
        changed = reason != self._target_hold_reason
        metrics = getattr(self, '_performance_metrics', None)
        if metrics is not None:
            metrics.increment('target_hold.frames')
            if changed:
                metrics.increment('target_hold.transitions')
        self._target_hold_reason = reason
        if changed or now - self._target_hold_log_at >= 2.0:
            self._target_hold_log_at = now
            self.get_logger().warning(
                f"External target HOLD: {reason}. Control resumes "
                "automatically when fresh commands arrive.")
        self._publish_lifecycle(LIFECYCLE_TARGET_HOLD)
        self._publish_teleop_status(f'HOLD: {reason}')

    def _clear_target_hold(self):
        if self._target_hold_reason is None:
            return
        self.get_logger().info(
            "Teleop target is valid again; resuming from the last safe command")
        self._target_hold_reason = None
        self._target_hold_log_at = 0.0
        self._publish_lifecycle(LIFECYCLE_READY)
        self._publish_teleop_status('READY: target valid, teleoperation resumed')

    def _set_enabled_callback(self, request: SetBool.Request, response: SetBool.Response):
        """ROS2 service: remote enable/disable. Fast-path based on SDK
        cur_state; do NOT trust the _arm_enabled flag (it can drift from
        hardware). _do_enable / _do_disable own the _enable_lock, so this
        callback doesn't need to grab it itself."""
        if request.data:
            if self._recovery_in_progress.is_set():
                response.success = False
                response.message = "Recovery is still running; enable is refused"
                return response
            if not self._recovery_complete:
                mode = getattr(self, '_active_arm_mode', 'both')
                recovery_service = {
                    'left': 'recover_left_to_init',
                    'right': 'recover_right_to_init',
                }.get(mode, 'recover_to_init')
                response.success = False
                response.message = (
                    "Enable refused: call /tianji_arm_controller/"
                    f"{recovery_service} "
                    "and wait for lifecycle_state=7 first")
                return response
            if self._start_enable_worker():
                self.get_logger().info(
                    "Enabling arms asynchronously: verify init and enter "
                    f"{getattr(self, '_arm_hardware_mode', 'impedance')} mode")
                response.success = True
                response.message = (
                    "Enable sequence started; wait for lifecycle_state=READY")
            else:
                response.success = False
                response.message = "Enable sequence is already running"
        else:
            try:
                self._do_disable()
            except Exception as e:
                self.get_logger().error(f"set_standby error: {e}")
                response.success = False
                response.message = f"set_standby failed: {e}"
                return response
            self.get_logger().info("Arms stopped")
            response.success = True
            response.message = "Arms stopped"
        return response

    def _recover_all_callback(self, request, response):
        return self._recover_callback(['A', 'B'], response)

    def _recover_left_callback(self, request, response):
        return self._recover_callback(['A'], response)

    def _recover_right_callback(self, request, response):
        return self._recover_callback(['B'], response)

    def _recover_callback(self, arms, response):
        if self._arm_enabled:
            response.success = False
            response.message = "Disable teleoperation before starting recovery"
            return response
        normalized = tuple(self.controller._normalize_arms(arms))
        configured = self._control_sdk_arms()
        mode = getattr(self, '_active_arm_mode', 'both')
        if mode != 'both' and normalized != configured:
            selected = self._control_sides()[0]
            response.success = False
            response.message = (
                f"Controller is configured for {selected}-arm teleoperation; "
                f"only recover_{selected}_to_init is permitted")
            return response
        if self._start_recovery_worker(arms):
            sides = ','.join('left' if arm == 'A' else 'right' for arm in arms)
            if mode == 'right':
                sides = 'left(vertical park),right(init)'
            response.success = True
            response.message = (
                f"Guarded recovery started for {sides}; "
                "wait for lifecycle_state=7 before enabling")
        else:
            response.success = False
            response.message = "Another enable or recovery sequence is already running"
        return response

    def _toggle_tracker_clutch_callback(self, _request, response):
        """Disconnect/re-anchor Tracker control without leaving impedance."""
        if getattr(self, '_control_source', 'tracker') != 'tracker':
            response.success = False
            response.message = "Tracker clutch is unavailable in external mode"
            return response
        if self._motion_in_progress():
            response.success = False
            response.message = "Tracker clutch is blocked during Recovery/Enable"
            return response
        if not self._arm_enabled:
            response.success = False
            response.message = "Enable teleoperation before using Tracker clutch"
            return response

        if self._tracker_connected:
            self._tracker_connected = False
            self.left_pose = None
            self.right_pose = None
            self._target_hold_reason = None
            self._publish_lifecycle(LIFECYCLE_CLUTCH_DISCONNECTED)
            self._publish_teleop_status(
                "CLUTCH DISCONNECTED: align the operator with the robot, "
                "then press pedal 3 again"
            )
            response.success = True
            response.message = (
                "Tracker disconnected; impedance remains active"
            )
            return response

        try:
            self._reset_control_targets()
            self._arm_handoff_snapshot()
            self._capture_teleop_neutral()
        except Exception as exc:
            self._tracker_connected = False
            self._publish_lifecycle(LIFECYCLE_CLUTCH_DISCONNECTED)
            self._publish_teleop_status(
                f"CLUTCH RECONNECT FAILED: {exc}"
            )
            response.success = False
            response.message = f"Tracker re-anchor failed: {exc}"
            return response

        self._tracker_connected = True
        self._publish_lifecycle(LIFECYCLE_READY)
        self._publish_teleop_status(
            "READY: Tracker clutch reconnected at current alignment"
        )
        response.success = True
        response.message = (
            "Tracker re-anchored and connected; guarded handoff resumed"
        )
        return response

    # -------------------- Brake (release / hold) --------------------

    def _left_brake_cb(self, request, response):
        return self._brake_cb('A', request, response)

    def _right_brake_cb(self, request, response):
        return self._brake_cb('B', request, response)

    def _brake_cb(self, arm, request, response):
        side = 'left' if arm == 'A' else 'right'
        if self._motion_in_progress():
            response.success = False
            response.message = f"{side} arm brake op rejected: hardware motion in progress"
            return response
        try:
            if request.data:
                self.controller.release_brake(arm)
                response.message = f"{side} arm: brake released"
            else:
                self.controller.hold_brake(arm)
                response.message = f"{side} arm: brake held"
            response.success = True
        except Exception as e:
            response.success = False
            response.message = f"{side} arm brake op failed: {e}"
        return response

    # -------------------- Clear error --------------------

    def _left_clear_error_cb(self, request, response):
        return self._clear_error_cb('A', request, response)

    def _right_clear_error_cb(self, request, response):
        return self._clear_error_cb('B', request, response)

    def _clear_error_cb(self, arm, request, response):
        side = 'left' if arm == 'A' else 'right'
        if self._motion_in_progress():
            response.success = False
            response.message = f"{side} clear-error rejected: hardware motion in progress"
            return response
        try:
            self.controller.clear_arm_error(arm)
            response.success = True
            response.message = f"{side} arm errors cleared"
        except Exception as e:
            response.success = False
            response.message = f"{side} arm clear-error failed: {e}"
        return response

    def _arm_status_cb(self, request, response):
        """ROS2 service: read both arms' state code / error code / servo error code."""
        started_ns = time.perf_counter_ns()
        metrics = getattr(self, '_performance_metrics', None)
        if metrics is not None:
            metrics.increment('arm_status.calls')
        try:
            if self._motion_in_progress():
                response.success = False
                response.message = (
                    "arm_status unavailable while hardware motion is in progress"
                )
                if metrics is not None:
                    metrics.increment('arm_status.rejected')
                return response
            import json
            status = self.controller.get_arm_status()
            self._last_detailed_arm_status = status
            self._last_detailed_arm_status_at = time.monotonic()
            self._last_arm_state_codes = tuple(
                int(status[side]['state'])
                for side in ('left', 'right')
            )
            self._last_arm_error_codes = tuple(
                int(status[side]['err_code'])
                for side in ('left', 'right')
            )
            response.success = True
            response.message = json.dumps(status)
        except Exception as e:
            response.success = False
            response.message = str(e)
            if metrics is not None:
                metrics.increment('arm_status.errors')
        finally:
            _observe_duration(self, 'arm_status.duration_ms', started_ns)
        return response

    # -------------------- Service callbacks --------------------

    def _set_arm_state_callback(self, request: SetBool.Request, response: SetBool.Response):
        """Switch arm control state:
            data=true  -> rejected; replay position mode is launch-controlled
            data=false -> confirm IMPEDANCE (state=3)

        Fast-path based on SDK cur_state; do NOT trust _arm_state / _arm_enabled
        flags (they can drift from hardware).
        """
        if request.data:
            # A forbidden-mode request is treated as a stop request. This
            # makes an accidental UI click fail closed instead of merely
            # returning an error while the arms remain energized.
            try:
                self._do_disable()
                recovery = " Arms were placed in standby."
            except Exception as exc:
                recovery = f" CRITICAL: standby request failed: {exc}"
            response.success = False
            response.message = self.controller.RIGID_MODE_DISABLED + recovery
            self.get_logger().error(response.message)
            return response

        if self._motion_in_progress():
            try:
                self._do_disable()
                response.success = False
                response.message = "Hardware motion cancelled; arms returned to standby"
            except Exception as exc:
                response.success = False
                response.message = f"Enable cancellation failed: {exc}"
            return response

        try:
            left_s, right_s = self.controller.get_arm_states_only()
        except Exception as e:
            response.success = False
            response.message = f"Failed to read hardware state, cannot determine mode: {e}"
            return response

        if left_s == 3 and right_s == 3:
            self._arm_state = ArmState.IMPEDANCE
            response.success = True
            response.message = "Already in compliant (impedance) mode (cur_state=3)"
            return response

        if left_s == 0 or right_s == 0:
            response.success = False
            response.message = (
                f"Arms not enabled (left state={left_s}, right state={right_s}); "
                f"call enable_all first"
            )
            return response

        # Do not transition directly from an unknown or rigid state. Return to
        # standby first and require a fresh, explicit set_enabled request.
        try:
            self._do_disable()
        except Exception as e:
            response.success = False
            response.message = f"Unexpected mode and standby recovery failed: {e}"
            return response
        response.success = False
        response.message = (
            f"Unexpected arm mode (left={left_s}, right={right_s}); "
            "arms returned to standby. Call set_enabled after inspection."
        )
        return response

    # -------------------- Control loop --------------------

    def _state_publish_loop(self):
        """500Hz: publish joint_states only."""
        started_ns = time.perf_counter_ns()
        active = not self._motion_in_progress()
        try:
            if active:
                self._publish_state()
        finally:
            metrics = getattr(self, '_performance_metrics', None)
            if metrics is not None:
                metrics.record_loop(
                    'state',
                    started_ns,
                    time.perf_counter_ns(),
                    self._state_publish_rate_hz,
                    active=active,
                )

    def _eef_publish_loop(self):
        """Publish measured Cartesian state from cached joint feedback."""
        started_ns = time.perf_counter_ns()
        active = not self._motion_in_progress()
        try:
            if not active:
                return
            stamp = self.get_clock().now().to_msg()
            for side, joints, publisher in (
                ('left', self._last_left_state_joints, self.left_actual_pose_pub),
                ('right', self._last_right_state_joints, self.right_actual_pose_pub),
            ):
                if joints is None:
                    continue
                matrix = self._robot_fk_matrix(side, joints)
                publisher.publish(self._matrix_to_pose_stamped(side, matrix, stamp))
        except Exception as exc:
            self.get_logger().warn(f"Failed to publish actual EE pose: {exc}")
        finally:
            metrics = getattr(self, '_performance_metrics', None)
            if metrics is not None:
                metrics.record_loop(
                    'eef',
                    started_ns,
                    time.perf_counter_ns(),
                    self._eef_publish_rate_hz,
                    active=active,
                )

    def _control_loop(self):
        """120Hz: TF -> IK -> hardware. State publish is handled by a separate timer."""
        started_ns = time.perf_counter_ns()
        active = bool(self._arm_enabled) and not (
            getattr(self, '_control_source', 'tracker') == 'tracker'
            and not getattr(self, '_tracker_connected', True)
        )
        try:
            if active:
                self._teleop_control()
        finally:
            metrics = getattr(self, '_performance_metrics', None)
            if metrics is not None:
                metrics.record_loop(
                    'control',
                    started_ns,
                    time.perf_counter_ns(),
                    self._control_rate_hz,
                    active=active,
                )

    def _external_pose_callback(self, side: str, message: PoseStamped):
        if getattr(self, '_control_source', 'tracker') != 'external':
            return
        if side not in self._control_sides():
            return
        expected_frame = f'{side}_chest'
        if message.header.frame_id and message.header.frame_id != expected_frame:
            self.get_logger().warn(
                f"Rejected {side} external pose in frame "
                f"{message.header.frame_id!r}; expected {expected_frame!r}")
            return
        try:
            matrix = self._pose_stamped_to_matrix(message)
        except Exception as exc:
            self.get_logger().warn(
                f"Rejected invalid {side} external pose: {exc}")
            return
        with self._external_lock:
            self._external_pose[side] = matrix
            self._external_pose_at[side] = time.monotonic()

    def _external_zsp_callback(self, side: str, message: Float64MultiArray):
        if getattr(self, '_control_source', 'tracker') != 'external':
            return
        if side not in self._control_sides():
            return
        vector = np.asarray(message.data[:3], dtype=float)
        norm = float(np.linalg.norm(vector)) if vector.shape == (3,) else 0.0
        if vector.shape != (3,) or not np.all(np.isfinite(vector)) or norm < 1e-9:
            self.get_logger().warn(f"Rejected invalid {side} external ZSP")
            return
        with self._external_lock:
            self._external_zsp[side] = vector / norm

    def _external_joint_callback(self, side: str, message: JointState):
        if getattr(self, '_control_source', 'tracker') != 'external':
            return
        if getattr(self, '_external_command_mode', 'eef') != 'joint':
            return
        if side not in self._control_sides():
            return
        joints_rad = np.asarray(message.position, dtype=float).reshape(-1)
        if joints_rad.shape != (7,) or not np.all(np.isfinite(joints_rad)):
            self.get_logger().warn(
                f"Rejected invalid {side} external 7-DoF joint target")
            return
        with self._external_lock:
            self._external_joint[side] = np.degrees(joints_rad).tolist()
            self._external_joint_at[side] = time.monotonic()

    def _refresh_external_joint_targets(
            self) -> tuple[Optional[list], Optional[list]]:
        """Return fresh external joint targets, or the enable-time hold."""
        active_sides = self._control_sides()
        now = time.monotonic()
        with self._external_lock:
            joints = {
                side: (
                    None if self._external_joint[side] is None
                    else list(self._external_joint[side])
                )
                for side in ('left', 'right')
            }
            joint_at = dict(self._external_joint_at)
            stream_started = self._external_stream_started
            hold = {
                side: (
                    None if self._external_hold_joint[side] is None
                    else list(self._external_hold_joint[side])
                )
                for side in ('left', 'right')
            }

        paired = all(joints[side] is not None for side in active_sides)
        ages = {
            side: (
                now - joint_at[side]
                if joint_at[side] > 0.0 else float('inf')
            )
            for side in active_sides
        }
        fresh = paired and max(ages.values()) <= self._external_command_timeout_sec
        if not stream_started:
            if not fresh:
                return hold['left'], hold['right']
            with self._external_lock:
                self._external_stream_started = True
            self.get_logger().info(
                "Fresh external joint command stream acquired for "
                + ','.join(active_sides))
        elif not fresh:
            worst_age = max(ages.values())
            if worst_age > self._external_standby_timeout_sec:
                self.get_logger().error(
                    f"External joint command stream stale for "
                    f"{worst_age:.3f}s; requesting standby")
                self._do_disable()
                return None, None
            raise ExternalCommandStale(
                f"external joint command stale for {worst_age:.3f}s")
        return (
            joints['left'] if 'left' in active_sides else None,
            joints['right'] if 'right' in active_sides else None,
        )

    def _refresh_external_targets(self) -> bool:
        """Copy fresh targets for the configured arm(s) into the IK path."""
        active_sides = self._control_sides()
        now = time.monotonic()
        with self._external_lock:
            poses = {
                side: None if self._external_pose[side] is None
                else self._external_pose[side].copy()
                for side in ('left', 'right')
            }
            pose_at = dict(self._external_pose_at)
            zsp = {
                side: None if self._external_zsp[side] is None
                else self._external_zsp[side].copy()
                for side in ('left', 'right')
            }
            stream_started = self._external_stream_started
            hold = {
                side: None if self._external_hold_pose[side] is None
                else self._external_hold_pose[side].copy()
                for side in ('left', 'right')
            }

        paired = all(poses[side] is not None for side in active_sides)
        ages = {
            side: now - pose_at[side] if pose_at[side] > 0.0 else float('inf')
            for side in active_sides
        }
        fresh = paired and max(ages.values()) <= self._external_command_timeout_sec

        if not stream_started:
            if not fresh:
                # Enable is allowed before the replay/policy server starts.
                # Keep the measured enable-time pose and do not arm the
                # watchdog until a complete paired command has arrived.
                for side in active_sides:
                    if hold[side] is not None:
                        setattr(
                            self, f'{side}_pose',
                            self._matrix_to_pose(hold[side]))
                return True
            with self._external_lock:
                self._external_stream_started = True
            self.get_logger().info(
                "Fresh external command stream acquired for "
                + ','.join(active_sides))
        elif not fresh:
            worst_age = max(ages.values())
            if worst_age > self._external_standby_timeout_sec:
                self.get_logger().error(
                    f"External command stream stale for {worst_age:.3f}s; "
                    "requesting standby")
                self._do_disable()
                return False
            raise ExternalCommandStale(
                f"external command stale for {worst_age:.3f}s")

        for side in ('left', 'right'):
            if side not in active_sides:
                setattr(self, f'{side}_pose', None)
                continue
            setattr(self, f'{side}_pose', self._matrix_to_pose(poses[side]))
            if zsp[side] is not None:
                setattr(self, f'{side}_y_axis', zsp[side])
        return True

    def _teleop_control(self):
        """Look up TF -> local IK -> hardware."""
        mapping_started_ns = time.perf_counter_ns()
        try:
            if getattr(self, '_control_source', 'tracker') == 'external':
                if getattr(self, '_external_command_mode', 'eef') == 'joint':
                    try:
                        l_target, r_target = (
                            self._refresh_external_joint_targets())
                        if (
                            getattr(
                                self,
                                '_external_handoff_gate_enabled',
                                False,
                            )
                            and hasattr(self, '_external_handoff_state')
                        ):
                            l_target, r_target = (
                                self._external_handoff_joint_targets(
                                    l_target, r_target)
                            )
                        else:
                            blend_s = self._handoff_blend_factor()
                            if blend_s is not None:
                                l_target = self._blend_joints(
                                    self._handoff_start_left,
                                    l_target,
                                    blend_s,
                                )
                                r_target = self._blend_joints(
                                    self._handoff_start_right,
                                    r_target,
                                    blend_s,
                                )
                        if l_target is not None or r_target is not None:
                            inactive_arms = [
                                arm for arm in ('A', 'B')
                                if arm not in self._control_sdk_arms()
                            ]
                            self.controller.move_to_joints_direct(
                                left_joints=l_target,
                                right_joints=r_target,
                                active_arms=list(self._control_sdk_arms()),
                                standby_arms=inactive_arms,
                                command_state=(
                                    1
                                    if getattr(
                                        self,
                                        '_arm_hardware_mode',
                                        'impedance',
                                    ) == 'position'
                                    else 3
                                ),
                            )
                            self._publish_command(l_target, r_target)
                            self._clear_target_hold()
                    except ExternalCommandStale as exc:
                        self._enter_target_hold(str(exc))
                    except Exception as exc:
                        self._on_sdk_error("external_joint_control", exc)
                    return
                if not self._refresh_external_targets():
                    return
            else:
                # Query TF and apply motion relative to the enable-time neutral.
                active_sides = self._control_sides()
                for side in ('left', 'right'):
                    if side not in active_sides:
                        setattr(self, f'{side}_pose', None)
                        setattr(self, f'{side}_y_axis', None)
                        continue
                    tracker_tf = self._lookup_transform(
                        f"{side}_chest", f"tianji_{side}")
                    if tracker_tf is not None:
                        setattr(
                            self,
                            f'{side}_pose',
                            self._matrix_to_pose(
                                self._map_tracker_target(side, tracker_tf)),
                        )
                    arm_tf = self._lookup_transform(
                        f"{side}_chest", f"{side}_arm")
                    if arm_tf is not None:
                        setattr(
                            self,
                            f'{side}_y_axis',
                            self._map_upper_arm_direction(side, arm_tf),
                        )
        except ExternalCommandStale as exc:
            self._enter_target_hold(str(exc))
            return
        except Exception as exc:
            self._on_sdk_error("teleop_mapping", exc)
            return
        finally:
            _observe_duration(
                self, 'control.mapping_ms', mapping_started_ns)

        # Log every 3 seconds.
        self._log_counter += 1
        if self._log_counter >= 300:
            self._log_counter = 0
            self.get_logger().info(
                f"TF: left={'OK' if self.left_pose is not None else 'None'}, "
                f"right={'OK' if self.right_pose is not None else 'None'}")

        # Run IK.
        if self.left_pose is not None or self.right_pose is not None:
            # Update null-space parameters.
            if self.left_y_axis is not None:
                self.controller.left_zsp_para = [*self.left_y_axis, 0, 0, 0]
            if self.right_y_axis is not None:
                self.controller.right_zsp_para = [*self.right_y_axis, 0, 0, 0]

            try:
                # Solve IK first so the active-arm routing remains explicit.
                # Match wuji origin/main after handoff: every valid IK solution
                # is dispatched directly, while a failed arm is skipped for
                # this frame by receiving a None joint target.
                solve_started_ns = time.perf_counter_ns()
                try:
                    _, _, l_target, r_target = (
                        self.controller.move_to_pose_direct(
                            left_pose=self.left_pose,
                            right_pose=self.right_pose,
                            unit='m',
                            send=False,
                        )
                    )
                finally:
                    _observe_duration(
                        self, 'control.ik_pipeline_ms', solve_started_ns)
                _observe_driver_timings(
                    self,
                    'driver.pose',
                    getattr(self.controller, '_last_pose_timing', None),
                )
                self._publish_alignment_target(l_target, r_target)
                if (
                    getattr(self, '_control_source', 'tracker') == 'external'
                    and getattr(
                        self, '_external_handoff_gate_enabled', False)
                    and hasattr(self, '_external_handoff_state')
                ):
                    l_joints, r_joints = self._external_handoff_joint_targets(
                        l_target, r_target)
                else:
                    blend_s = self._handoff_blend_factor()
                    if blend_s is None:
                        l_joints, r_joints = l_target, r_target
                    else:
                        l_joints = self._blend_joints(
                            self._handoff_start_left, l_target, blend_s)
                        r_joints = self._blend_joints(
                            self._handoff_start_right, r_target, blend_s)
                if l_joints is not None or r_joints is not None:
                    inactive_arms = [
                        arm for arm in ('A', 'B')
                        if arm not in self._control_sdk_arms()
                    ]
                    command_started_ns = time.perf_counter_ns()
                    try:
                            self.controller.move_to_joints_direct(
                                left_joints=l_joints,
                                right_joints=r_joints,
                                active_arms=list(self._control_sdk_arms()),
                                standby_arms=inactive_arms,
                                command_state=(
                                    1
                                    if getattr(
                                        self,
                                        '_arm_hardware_mode',
                                        'impedance',
                                    ) == 'position'
                                    else 3
                                ),
                            )
                    finally:
                        _observe_duration(
                            self,
                            'control.command_pipeline_ms',
                            command_started_ns,
                        )
                    _observe_driver_timings(
                        self,
                        'driver.command',
                        getattr(
                            self.controller,
                            '_last_joint_command_timing',
                            None,
                        ),
                    )
                self._publish_command(l_joints, r_joints)
                self._clear_target_hold()
            except Exception as e:
                self._on_sdk_error("teleop_control", e)
                return

        # Publish null-space params and end-effector poses (per-arm; no need for both online).
        self._publish_zsp_para_and_pose()

    def _handoff_blend_factor(self) -> Optional[float]:
        """Return the IK-weight for the handoff ramp, or None if the ramp is done.

        0.0 = full hold at snapshot; 1.0 = full live IK target. During the hold
        phase we return 0.0 (hardware tracks the snapshot); during the ramp we
        smoothstep from 0 to 1; after the ramp we clear state and return None
        so the steady-state fused path takes over.
        """
        if self._handoff_start_at is None:
            return None
        elapsed = time.monotonic() - self._handoff_start_at
        ramp_end = self._handoff_hold_sec + self._handoff_ramp_sec
        if elapsed >= ramp_end:
            self._handoff_start_at = None
            self.get_logger().info("Handoff ramp complete; full-rate teleop engaged")
            return None
        if elapsed < self._handoff_hold_sec or self._handoff_ramp_sec <= 0.0:
            return 0.0
        t = (elapsed - self._handoff_hold_sec) / self._handoff_ramp_sec
        # Quintic smoothstep (same shape as move_to_joints_smooth in the driver).
        return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5

    def _external_handoff_joint_targets(
        self,
        left_target: Optional[list],
        right_target: Optional[list],
    ) -> tuple[Optional[list], Optional[list]]:
        """Latch and execute the first external IK target as a fixed ramp.

        The external stream must already be complete and fresh.  Until then,
        the enable-time joint snapshot is held.  Once every configured arm has
        a valid IK result, the target is copied and cannot move until the ramp
        reports COMPLETE.
        """

        state = str(getattr(
            self,
            '_external_handoff_state',
            EXTERNAL_HANDOFF_COMPLETE,
        ))
        if state in (EXTERNAL_HANDOFF_IDLE, EXTERNAL_HANDOFF_COMPLETE):
            return left_target, right_target
        if state == EXTERNAL_HANDOFF_FAILED:
            return self._handoff_start_left, self._handoff_start_right

        active_sides = self._control_sides()
        targets = {'left': left_target, 'right': right_target}
        with self._external_lock:
            stream_started = bool(self._external_stream_started)

        if state == EXTERNAL_HANDOFF_WAITING_TARGET:
            if (
                not stream_started
                or any(targets[side] is None for side in active_sides)
            ):
                return self._handoff_start_left, self._handoff_start_right
            self._external_handoff_target_left = (
                list(left_target) if 'left' in active_sides else None)
            self._external_handoff_target_right = (
                list(right_target) if 'right' in active_sides else None)
            self._handoff_start_at = time.monotonic()
            self._publish_external_handoff_state(EXTERNAL_HANDOFF_ACTIVE)
            self.get_logger().info(
                "External handoff target latched; fixed-target ramp started")

        frozen_left = self._external_handoff_target_left
        frozen_right = self._external_handoff_target_right
        started_at = self._handoff_start_at
        if started_at is None:
            self._publish_external_handoff_state(EXTERNAL_HANDOFF_FAILED)
            self.get_logger().error(
                "External handoff is ACTIVE without a start timestamp")
            return self._handoff_start_left, self._handoff_start_right

        elapsed = max(0.0, time.monotonic() - started_at)
        ramp_end = self._handoff_hold_sec + self._handoff_ramp_sec
        if ramp_end <= 0.0 or elapsed >= ramp_end:
            self._handoff_start_at = None
            self._external_handoff_completed_at = time.monotonic()
            self._publish_external_handoff_state(EXTERNAL_HANDOFF_COMPLETE)
            self.get_logger().info(
                "External handoff complete; live policy targets may advance")
            return frozen_left, frozen_right
        if elapsed < self._handoff_hold_sec or self._handoff_ramp_sec <= 0.0:
            blend_s = 0.0
        else:
            t = (
                (elapsed - self._handoff_hold_sec)
                / self._handoff_ramp_sec
            )
            blend_s = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
        return (
            self._blend_joints(
                self._handoff_start_left, frozen_left, blend_s),
            self._blend_joints(
                self._handoff_start_right, frozen_right, blend_s),
        )

    @staticmethod
    def _blend_joints(start: Optional[list], target: Optional[list],
                      s: float) -> Optional[list]:
        """Return start*(1-s) + target*s; preserve target if start is missing,
        and preserve start if target is missing (lets a per-side TF dropout
        still hold its snapshot instead of going None)."""
        if target is None and start is None:
            return None
        if start is None:
            return list(target)
        if target is None:
            return list(start)
        return [start[i] + s * (target[i] - start[i]) for i in range(len(start))]

    # -------------------- Publish --------------------

    def _publish_command(self, left: Optional[list], right: Optional[list]):
        stamp = self.get_clock().now().to_msg()
        if left is not None:
            msg = JointState()
            msg.header.stamp = stamp
            msg.header.frame_id = "left_base_cmd"
            msg.name = [f'left_joint_{i+1}' for i in range(7)]
            msg.position = list(left)
            self.left_cmd_pub.publish(msg)
        if right is not None:
            msg = JointState()
            msg.header.stamp = stamp
            msg.header.frame_id = "right_base_cmd"
            msg.name = [f'right_joint_{i+1}' for i in range(7)]
            msg.position = list(right)
            self.right_cmd_pub.publish(msg)

    def _publish_alignment_target(self, left: Optional[list],
                                  right: Optional[list]):
        """Publish raw IK targets in URDF radians for the read-only RViz ghost."""
        if left is None or right is None:
            return
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(ALIGNMENT_JOINT_NAMES)
        message.position = np.radians(
            np.asarray(list(left) + list(right), dtype=float)).tolist()
        self._alignment_target_pub.publish(message)

    def _publish_state(self):
        try:
            read_started_ns = time.perf_counter_ns()
            try:
                left_joints, right_joints = (
                    self.controller.get_current_joints()
                )
            finally:
                _observe_duration(
                    self, 'driver.state.sdk_read_ms', read_started_ns)
            self._state_error_count = 0  # reset on success
            self._last_left_state_joints = list(left_joints) if left_joints is not None else None
            self._last_right_state_joints = list(right_joints) if right_joints is not None else None
            cached_states = getattr(
                self.controller, 'latest_arm_state_codes', None)
            cached_errors = getattr(
                self.controller, 'latest_arm_error_codes', None)
            if cached_states is not None and len(cached_states) == 2:
                self._last_arm_state_codes = tuple(
                    int(value) for value in cached_states)
            if cached_errors is not None and len(cached_errors) == 2:
                self._last_arm_error_codes = tuple(
                    int(value) for value in cached_errors)
            self._last_state_feedback_at = time.monotonic()
            stamp = self.get_clock().now().to_msg()

            if left_joints is not None:
                msg = JointState()
                msg.header.stamp = stamp
                msg.header.frame_id = "left_base_state"
                msg.name = [f'left_joint_{i+1}' for i in range(7)]
                msg.position = list(left_joints)
                self.left_state_pub.publish(msg)

            if right_joints is not None:
                msg = JointState()
                msg.header.stamp = stamp
                msg.header.frame_id = "right_base_state"
                msg.name = [f'right_joint_{i+1}' for i in range(7)]
                msg.position = list(right_joints)
                self.right_state_pub.publish(msg)

            if left_joints is not None and right_joints is not None:
                alignment = JointState()
                alignment.header.stamp = stamp
                alignment.name = list(ALIGNMENT_JOINT_NAMES)
                alignment.position = np.radians(
                    np.asarray(
                        list(left_joints) + list(right_joints), dtype=float)
                ).tolist()
                self._alignment_state_pub.publish(alignment)
        except Exception as e:
            self._state_error_count += 1
            # Only escalate to SDK fault while the arm is still enabled (avoid infinite loop after disable).
            if self._arm_enabled and self._state_error_count >= 50:
                self._on_sdk_error("publish_state", e)
                self._state_error_count = 0

    def _publish_zsp_para_and_pose(self):
        """Publish null-space parameters and end-effector poses."""
        try:
            # --- left_zsp_para ---
            # try/except guards against the controller being torn down mid-read.
            if hasattr(self.controller, 'left_zsp_para') and self.controller.left_zsp_para is not None:
                raw_data = self.controller.left_zsp_para
                if len(raw_data) > 0:
                    msg = Float64MultiArray()
                    # Coerce to list[float] for DDS compatibility.
                    msg.data = [float(x) for x in raw_data]
                    self.left_zsp_para_pub.publish(msg)

            # --- right_zsp_para ---
            if hasattr(self.controller, 'right_zsp_para') and self.controller.right_zsp_para is not None:
                raw_data = self.controller.right_zsp_para
                if len(raw_data) > 0:
                    msg = Float64MultiArray()
                    msg.data = [float(x) for x in raw_data]
                    self.right_zsp_para_pub.publish(msg)

            # --- left_pose ---
            if self.left_pose is not None:
                msg = Float64MultiArray()
                msg.data = [float(x) for x in self.left_pose]
                self.left_ee_pose_pub.publish(msg)
                self.left_target_pose_pub.publish(
                    self._matrix_to_pose_stamped(
                        'left', self._pose6_to_matrix(self.left_pose)))

            # --- right_pose ---
            if self.right_pose is not None:
                msg = Float64MultiArray()
                msg.data = [float(x) for x in self.right_pose]
                self.right_ee_pose_pub.publish(msg)
                self.right_target_pose_pub.publish(
                    self._matrix_to_pose_stamped(
                        'right', self._pose6_to_matrix(self.right_pose)))

        except Exception as e:
            # During shutdown, a freed C++ object can still appear briefly.
            # Log only; do not crash the node.
            self.get_logger().warn(f"Failed to publish debug info during shutdown: {e}")

    # -------------------- TF utilities --------------------

    def _lookup_transform(self, from_frame: str, to_frame: str) -> Optional[np.ndarray]:
        try:
            tf = self.tf_buffer.lookup_transform(from_frame, to_frame, rclpy.time.Time())
            return self._transform_to_matrix(tf)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None

    @staticmethod
    def _transform_to_matrix(transform) -> np.ndarray:
        t = transform.transform
        quat = [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R.from_quat(quat).as_matrix()
        T[:3, 3] = [t.translation.x, t.translation.y, t.translation.z]
        return T

    @staticmethod
    def _pose_stamped_to_matrix(message: PoseStamped) -> np.ndarray:
        pose = message.pose
        quaternion = np.asarray(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
            dtype=float,
        )
        norm = float(np.linalg.norm(quaternion))
        position = np.asarray(
            [pose.position.x, pose.position.y, pose.position.z], dtype=float)
        if (
            not np.all(np.isfinite(position))
            or not np.all(np.isfinite(quaternion))
            or norm < 1e-9
        ):
            raise ValueError("pose contains an invalid position or quaternion")
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = R.from_quat(quaternion / norm).as_matrix()
        matrix[:3, 3] = position
        return matrix

    @staticmethod
    def _pose6_to_matrix(pose) -> np.ndarray:
        value = np.asarray(pose, dtype=float)
        if value.shape != (6,) or not np.all(np.isfinite(value)):
            raise ValueError("Tianji pose must contain xyz + rpy degrees")
        matrix = np.eye(4, dtype=float)
        matrix[:3, 3] = value[:3]
        # _matrix_to_pose extracts ZYX and reverses it to XYZ-style storage.
        matrix[:3, :3] = R.from_euler(
            'ZYX', value[3:6][::-1], degrees=True).as_matrix()
        return matrix

    def _matrix_to_pose_stamped(self, side: str, matrix: np.ndarray,
                                stamp=None) -> PoseStamped:
        value = np.asarray(matrix, dtype=float)
        if value.shape != (4, 4) or not np.all(np.isfinite(value)):
            raise ValueError(f"{side} pose matrix is invalid")
        quaternion = R.from_matrix(value[:3, :3]).as_quat()
        message = PoseStamped()
        message.header.stamp = stamp or self.get_clock().now().to_msg()
        message.header.frame_id = f'{side}_chest'
        message.pose.position.x = float(value[0, 3])
        message.pose.position.y = float(value[1, 3])
        message.pose.position.z = float(value[2, 3])
        message.pose.orientation.x = float(quaternion[0])
        message.pose.orientation.y = float(quaternion[1])
        message.pose.orientation.z = float(quaternion[2])
        message.pose.orientation.w = float(quaternion[3])
        return message

    @staticmethod
    def _matrix_to_pose(matrix: np.ndarray) -> np.ndarray:
        """4x4 matrix -> [x, y, z, RX, RY, RZ] (degrees)."""
        xyz = matrix[:3, 3]
        rpy = R.from_matrix(matrix[:3, :3]).as_euler('ZYX', degrees=True)[::-1]
        return np.array([xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2]])

    def shutdown(self):
        self.get_logger().info("Shutting down...")
        cancel_event = getattr(self, '_enable_cancel_event', None)
        if cancel_event is not None:
            cancel_event.set()
        recovery_cancel = getattr(self, '_recovery_cancel_event', None)
        if recovery_cancel is not None:
            recovery_cancel.set()
        enable_thread = getattr(self, '_enable_thread', None)
        if enable_thread is not None and enable_thread.is_alive():
            enable_thread.join(timeout=5.0)
            if enable_thread.is_alive():
                self.get_logger().error(
                    "Enable worker did not exit within 5s; requesting SDK shutdown")
        recovery_thread = getattr(self, '_recovery_thread', None)
        if recovery_thread is not None and recovery_thread.is_alive():
            recovery_thread.join(timeout=5.0)
            if recovery_thread.is_alive():
                self.get_logger().error(
                    "Recovery worker did not exit within 5s; requesting SDK shutdown")
        if hasattr(self, 'controller'):
            self.controller.disable_and_release()
        self.get_logger().info("Exited cleanly")


# -------------------- Entry point --------------------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tianji arm controller")
    parser.add_argument("-c", "--config", help="Config file path")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None):
    program_name = sys.argv[0] if sys.argv else "tianji_arm_controller"
    raw_argv = sys.argv if argv is None else [program_name, *argv]
    cli_argv = remove_ros_args(raw_argv)[1:]
    args = _parse_args(cli_argv)

    # Load config.
    config_path = args.config or get_package_config_path("tianji_output", "tianji_chest.yaml")
    config = load_yaml_config(config_path)

    rclpy.init(args=raw_argv)
    node = None
    try:
        node = TianjiArmControllerNode(robot_ip=config.get("robot_ip", "192.168.1.190"))
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        # KeyboardInterrupt: direct SIGINT to a non-rclpy code path.
        # ExternalShutdownException: rclpy's documented signal-shutdown signal.
        # RCLError: same signal, race window — rclpy's SIGINT handler
        # invalidated the context mid-iteration and spin's next WaitSet
        # tripped on "context is not valid". All three mean "exit cleanly".
        pass
    finally:
        if node is not None:
            try:
                node.shutdown()
            except Exception as e:
                print(f"shutdown() raised: {e}", file=sys.stderr)
            try:
                node.destroy_node()
            except Exception:
                pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


def process_main(argv: Optional[list[str]] = None):
    """Exit deterministically after all robot and ROS cleanup has completed."""
    main(argv)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        # Marvin native destructors can hang or segfault after Robot released.
        # This runs only after main() has completed standby, SDK release,
        # destroy_node(), and rclpy.shutdown().
        os._exit(0)


if __name__ == "__main__":
    process_main()
