#!/usr/bin/env python3
"""
Tianji Arm Chest-frame Driver.

Integrates Cartesian space control (via IK) and joint space control
(direct joint angles). HTC Tracker / PICO teleop uses the Cartesian
path; the joint path is kept for future use.

All tunable parameters load from config/tianji_chest.yaml (single
source of truth).

Notes:
- Requires controller firmware 100342 or newer (otherwise set_tool_kine
  has no effect on IK).
- IK tool offset is set on the kine side (set_tool_kine); the hardware
  side `robot.set_tool` is given kineParams=[0,0,0,0,0,0] so we don't
  double-apply the offset.
"""
try:
    from tianji_output._internal.fx_robot import Marvin_Robot
    from tianji_output._internal.fx_kine import Marvin_Kine
    from tianji_output._internal.structure_data import DCSS
    from tianji_output.fault_codes import describe_fault_codes, read_servo_fault_codes
except ImportError:
    from ._internal.fx_robot import Marvin_Robot
    from ._internal.fx_kine import Marvin_Kine
    from ._internal.structure_data import DCSS
    from .fault_codes import describe_fault_codes, read_servo_fault_codes
import time
import logging
import os
import yaml
import numpy as np
from ament_index_python.packages import get_package_share_directory


class TianjiChestDriver:
    """
    Tianji dual-arm Chest-frame driver.

    Supports Cartesian-space control (via IK) and joint-space control
    (direct angles). HTC/Tracker teleop uses the Cartesian pathway.
    """

    RIGID_MODE_DISABLED = (
        "Rigid position mode (Marvin state=1) is available only inside the "
        "guarded recover-to-init workflow or explicit external replay"
    )
    MAX_RECOVERY_VEL_RATIO = 10
    MAX_RECOVERY_ACC_RATIO = 10
    ARM_INDEX = {'A': 0, 'B': 1}
    ARM_SIDE = {'A': 'left', 'B': 'right'}
    ARM_STATE_ERROR = 100
    ARM_TRANSITION_STATES = frozenset((101, 102, 103, 104, 109))
    ARM_ERROR_DESCRIPTIONS = {
        0: 'ok',
        1: 'bus topology fault',
        2: 'servo fault',
        3: 'PVT fault',
        4: 'request-enter-position failed',
        5: 'enter-position failed',
        6: 'request-enter-torque failed',
        7: 'enter-torque failed',
        8: 'request-enable-servo failed',
        9: 'enable-servo failed',
        10: 'request-disable-servo failed',
        11: 'disable-servo failed',
        12: 'internal error',
        13: 'e-stop',
        14: 'floating-base configuration error',
    }

    def __init__(self, robot_ip='192.168.1.190', config_path=None, logger=None):
        """
        Initialize both arms.

        Args:
            robot_ip: robot controller IP address.
            config_path: kinematics config file path.
                - None: use the default 'ccs_m6.MvKDCfg' shipped in the
                  ROS2 package share dir.
                - absolute path: use it directly.
            logger: optional external logger (e.g. ROS2 node logger).
        """
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger('TianjiChestDriver')
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('[%(name)s] %(message)s'))
                self.logger.addHandler(handler)

        # Load tianji_chest.yaml
        package_share = get_package_share_directory('tianji_output')
        yaml_path = os.path.join(package_share, 'config', 'tianji_chest.yaml')
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)

        init_joints = cfg.get('init_joints')
        if not init_joints or 'left' not in init_joints or 'right' not in init_joints:
            raise ValueError(
                f"tianji_chest.yaml missing init_joints.left/right: {yaml_path}")
        self._init_joints_left = list(init_joints['left'])
        self._init_joints_right = list(init_joints['right'])
        single_arm_init_joints = cfg.get('single_arm_init_joints') or {}
        self._single_arm_init_joints_right = list(
            single_arm_init_joints.get('right', self._init_joints_right)
        )
        park_joints = cfg.get('park_joints') or {}
        if 'left' not in park_joints:
            raise ValueError(
                f"tianji_chest.yaml missing park_joints.left: {yaml_path}")
        self._park_joints_left = list(park_joints['left'])
        self._last_impedance_check_at = 0.0
        self._last_impedance_check_arms = ()
        self._last_position_check_at = 0.0
        self._last_position_check_arms = ()
        # Read by the ROS controller immediately after each realtime call.
        # These dictionaries are diagnostics only; they never affect commands.
        self._last_pose_timing = {}
        self._last_joint_command_timing = {}
        # Updated as a side effect of the existing joint-state SDK subscribe.
        # The Stage-B GUI status snapshot reads these cached values and never
        # performs an additional hardware query.
        self.latest_arm_state_codes = None
        self.latest_arm_error_codes = None

        # Resolve kinematics config path
        if config_path is None:
            config_filename = 'ccs_m6.MvKDCfg'
            config_path = os.path.join(package_share, 'config', config_filename)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self.logger.debug(f"Loading config: {config_path}")

        # ---------------------- Kinematics init ----------------------
        self.logger.debug("[arm A] Initializing kinematics SDK...")
        self.kine_left = Marvin_Kine()
        config_result = self.kine_left.load_config(config_path=config_path)
        self._joint_limits_left = [
            (float(row[1]), float(row[0])) for row in config_result['PNVA'][0]
        ]
        time.sleep(0.3)
        self.kine_left.initial_kine(
            robot_serial=0,
            robot_type=config_result['TYPE'][0],
            dh=config_result['DH'][0],
            pnva=config_result['PNVA'][0],
            j67=config_result['BD'][0]
        )
        # Tool kinematics offset: identity (no offset).
        # New SDK handles tool offset on the IK side; hardware-side
        # robot.set_tool no longer receives kineParams.
        self.kine_left.set_tool_kine(0, [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        self.logger.debug("[arm B] Initializing kinematics SDK...")
        self.kine_right = Marvin_Kine()
        config_result = self.kine_right.load_config(config_path=config_path)
        self._joint_limits_right = [
            (float(row[1]), float(row[0])) for row in config_result['PNVA'][1]
        ]
        time.sleep(0.3)
        self.kine_right.initial_kine(
            robot_serial=1,
            robot_type=config_result['TYPE'][1],
            dh=config_result['DH'][1],
            pnva=config_result['PNVA'][1],
            j67=config_result['BD'][1]
        )
        self.kine_right.set_tool_kine(1, [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        # ---------------------- Robot connection ----------------------
        self.logger.debug("Initializing robot control...")
        self.robot = Marvin_Robot()

        init = self.robot.connect(robot_ip)
        if init == 0:
            raise ConnectionError("Connection failed: port in use")

        time.sleep(0.5)
        self.robot.clear_set()
        self.robot.clear_error('A')
        self.robot.clear_error('B')
        self.robot.send_cmd()
        time.sleep(0.5)

        if not self._verify_connection():
            raise ConnectionError("Robot connection failed")

        self._has_send_wait_response = self.robot.supports_send_cmd_wait_response()
        if not self._has_send_wait_response:
            self.logger.warning(
                "Marvin SDK does not export OnSetSendWaitResponse; using "
                "legacy OnSetSend for standby only. Guarded recovery is "
                "disabled until the SDK is upgraded")

        TianjiChestDriver._shared_robot = self.robot
        TianjiChestDriver._shared_kine_left = self.kine_left
        TianjiChestDriver._shared_kine_right = self.kine_right
        TianjiChestDriver._initialized = True

        # ---------------------- IK parameters (live-tunable) ----------------------
        self.zsp_type = 1                            # nullspace constraint type
        self.left_zsp_para = [0, -1, -0.5, 0, 0, 0]  # left-arm nullspace plane
        self.right_zsp_para = [0, 1, -0.5, 0, 0, 0]  # right-arm nullspace plane
        self.zsp_angle = 0.0                         # nullspace arm-angle rotation
        self.dgr = [5.0, 5.0, 5.0]                   # singularity tolerance (deg)

        # Tool dynamic parameters (Wuji Hand)
        self._set_tool_params()

        self.logger.info("Dual-arm controller initialized")

    def _set_tool_params(self):
        """
        Set tool dynamic params (gravity compensation, hardware side).

        kineParams is forced to [0,0,0,0,0,0]: the geometric tool offset
        is fed to IK via kine.set_tool_kine(); setting it on the hardware
        side too would double-compensate and misalign the wrist tracker
        from the flange.
        """
        # Left Wuji Hand keeps the existing nominal payload.  The right-hand
        # payload was identified on the physical robot with the CCS tool
        # identification trajectory (controller 100343009, 2026-08-13).
        # Layout: [mass, mx, my, mz, ixx, ixy, ixz, iyy, iyz, izz].
        tool_dyn_left = [0.95, 0, 0, 90, 0, 0, 0, 0, 0, 0]
        tool_dyn_right = [
            1.044,
            6.089,
            -21.289,
            101.794,
            0.012,
            0.0,
            0.0,
            0.008,
            0.0,
            0.002,
        ]

        self.logger.debug(
            f"Setting tool dynamics: A={tool_dyn_left}, B={tool_dyn_right}")

        self.robot.clear_set()
        self.robot.set_tool(arm='A', kineParams=[0, 0, 0, 0, 0, 0], dynamicParams=tool_dyn_left)
        self.robot.set_tool(arm='B', kineParams=[0, 0, 0, 0, 0, 0], dynamicParams=tool_dyn_right)
        self.robot.send_cmd()
        time.sleep(0.3)

    def _verify_connection(self):
        """Sanity check: subscribe a few times, look for frame_serial advancing."""
        dcss = DCSS()
        motion_tag = 0
        frame_update = None
        for i in range(5):
            sub_data = self.robot.subscribe(dcss)
            serial = sub_data['outputs'][0]['frame_serial']
            if serial != 0 and frame_update != serial:
                motion_tag += 1
                frame_update = serial
            time.sleep(0.1)
        return motion_tag > 0

    # ==================== State queries ====================

    @classmethod
    def _normalize_arms(cls, arms=None):
        if arms is None:
            return ['A', 'B']
        if isinstance(arms, str):
            arms = [arms]
        normalized = []
        for arm in arms:
            value = str(arm).upper()
            if value not in cls.ARM_INDEX:
                raise ValueError(f"Unsupported Tianji arm: {arm!r}")
            if value not in normalized:
                normalized.append(value)
        if not normalized:
            raise ValueError("At least one Tianji arm must be selected")
        return normalized

    def get_current_joints(self):
        """Return current joints and cache state/error codes from the same frame.

        ``robot.subscribe`` already returns joint feedback, arm state and arm
        error codes together.  Caching the latter two here lets lightweight
        observers reuse the 500 Hz state frame without another SDK call.
        """
        dcss = DCSS()
        sub_data = self.robot.subscribe(dcss)
        states = sub_data["states"]
        self.latest_arm_state_codes = (
            int(states[0]["cur_state"]),
            int(states[1]["cur_state"]),
        )
        self.latest_arm_error_codes = (
            int(states[0]["err_code"]),
            int(states[1]["err_code"]),
        )
        left_joints = sub_data["outputs"][0]["fb_joint_pos"]
        right_joints = sub_data["outputs"][1]["fb_joint_pos"]
        return left_joints, right_joints

    def get_init_joints(self, active_arm='both'):
        """Return copies of the Recovery references for the selected mode."""
        active_arm = str(active_arm).strip().lower()
        if active_arm not in ('both', 'left', 'right'):
            raise ValueError(
                f"active_arm must be both, left, or right, got {active_arm!r}")
        right = (
            self._single_arm_init_joints_right
            if active_arm == 'right'
            else self._init_joints_right
        )
        return list(self._init_joints_left), list(right)

    def get_park_joints(self, arm):
        """Return the configured inactive-arm parking pose."""
        arm = self._normalize_arms([arm])[0]
        if arm != 'A':
            raise ValueError(
                f"No inactive parking pose is configured for arm {arm}")
        return list(self._park_joints_left)

    def get_joint_limits(self):
        """Return configured (lower, upper) joint limits for both arms."""
        return list(self._joint_limits_left), list(self._joint_limits_right)

    def get_arm_snapshot(self):
        """Read state and both joint-feedback channels in one SDK frame."""
        sub_data = self.robot.subscribe(DCSS())
        outputs = sub_data['outputs']
        try:
            external_joints = [
                list(outputs[0]['fb_joint_posE']),
                list(outputs[1]['fb_joint_posE']),
            ]
        except KeyError as exc:
            raise RuntimeError(
                "Marvin SDK snapshot does not expose external-encoder "
                "feedback (fb_joint_posE); guarded recovery is refused"
            ) from exc
        return {
            'states': [int(item['cur_state']) for item in sub_data['states']],
            'errors': [int(item['err_code']) for item in sub_data['states']],
            'joints': [
                list(outputs[0]['fb_joint_pos']),
                list(outputs[1]['fb_joint_pos']),
            ],
            'external_joints': external_joints,
            'command_joints': [
                list(outputs[0].get('fb_joint_cmd', [])),
                list(outputs[1].get('fb_joint_cmd', [])),
            ],
            'velocities': [
                list(outputs[0]['fb_joint_vel']),
                list(outputs[1]['fb_joint_vel']),
            ],
        }

    def get_current_joint_velocities(self):
        """Return (left_velocities, right_velocities), each 7 joint velocities in deg/s."""
        dcss = DCSS()
        sub_data = self.robot.subscribe(dcss)
        left_vel = sub_data["outputs"][0]["fb_joint_vel"]
        right_vel = sub_data["outputs"][1]["fb_joint_vel"]
        return left_vel, right_vel

    def get_current_joint_torques(self):
        """Return (left_torques, right_torques), each 7 joint torques in Nm."""
        dcss = DCSS()
        sub_data = self.robot.subscribe(dcss)
        left_torque = sub_data["outputs"][0]["fb_joint_tor"]
        right_torque = sub_data["outputs"][1]["fb_joint_tor"]
        return left_torque, right_torque

    def get_full_state(self):
        """Return dual-arm joint positions, velocities, and torques as a dict."""
        dcss = DCSS()
        sub_data = self.robot.subscribe(dcss)
        return {
            'left': {
                'joints': sub_data["outputs"][0]["fb_joint_pos"],
                'velocities': sub_data["outputs"][0]["fb_joint_vel"],
                'torques': sub_data["outputs"][0]["fb_joint_tor"],
            },
            'right': {
                'joints': sub_data["outputs"][1]["fb_joint_pos"],
                'velocities': sub_data["outputs"][1]["fb_joint_vel"],
                'torques': sub_data["outputs"][1]["fb_joint_tor"],
            }
        }

    def get_arm_states_only(self):
        """Lightweight state read: return (left_cur_state, right_cur_state).

        Used by service callbacks for hardware-driven state checks
        (e.g. to confirm the arm is actually in the target state rather
        than trusting a Python flag). One SDK subscribe, no servo-error
        read — about 5 ms. Much lighter than get_arm_status().
        """
        dcss = DCSS()
        sub_data = self.robot.subscribe(dcss)
        return (sub_data["states"][0]["cur_state"], sub_data["states"][1]["cur_state"])

    def get_arm_status(self):
        """Dual-arm state codes, servo error codes, and trajectory-realtime metrics.

        Returns:
            dict: {'left': {'state': int, 'err_code': int,
                            'servo_errors': list[str],          # 7 hex strings
                            'servo_descriptions': list[str],    # 7 EN strings (empty=no error)
                            'realtime': dict},                  # frame_miss_cnt etc.
                   'right': ...}
        """
        dcss = DCSS()
        sub_data = self.robot.subscribe(dcss)
        result = {}
        for idx, (arm_id, side) in enumerate([('A', 'left'), ('B', 'right')]):
            servo_errors = self._get_servo_errors_raw(arm_id)
            inputs = sub_data.get('inputs') or []
            rt_in = inputs[idx] if idx < len(inputs) else {}
            frame_miss_cnt = int(rt_in.get('frame_miss_cnt', 0))
            result[side] = {
                'state': sub_data["states"][idx]["cur_state"],
                'err_code': sub_data["states"][idx]["err_code"],
                'servo_errors': servo_errors,
                'servo_descriptions': describe_fault_codes(servo_errors, empty=''),
                'realtime': {
                    'frame_miss_cnt': frame_miss_cnt,
                    'max_frame_miss_cnt': int(rt_in.get('max_frame_miss_cnt', 0)),
                    'in_frame_serial': int(rt_in.get('in_frame_serial', 0)),
                    'sys_cyc_miss_cnt': int(rt_in.get('sys_cyc_miss_cnt', 0)),
                    'max_sys_cyc_miss_cnt': int(rt_in.get('max_sys_cyc_miss_cnt', 0)),
                    'quality': 'good' if frame_miss_cnt < 20 else 'poor',
                },
            }
        return result

    def _get_servo_errors_raw(self, arm: str):
        """Return 7 servo fault codes for the given arm (hex string list)."""
        return read_servo_fault_codes(self.robot.robot, arm)

    # ==================== Impedance mode ====================

    def _send_cmd_wait_response(self, context, timeout_ms=100):
        """Dispatch the SDK batch and require a positive API result."""
        result = int(self.robot.send_cmd_wait_response(timeout_ms))
        if result <= 0:
            raise RuntimeError(
                f"{context}: controller command dispatch failed (result={result})")
        return result

    def _require_impedance_mode(self, max_cache_age=0.1, arms=None,
                                standby_arms=None):
        """Require commanded arms in state=3 and selected inactive arms in 0."""
        arms = self._normalize_arms(arms)
        standby_arms = (
            [] if not standby_arms
            else self._normalize_arms(standby_arms)
        )
        overlap = set(arms) & set(standby_arms)
        if overlap:
            raise ValueError(
                f"Arms cannot be active and standby simultaneously: {overlap}")
        arm_key = (tuple(arms), tuple(standby_arms))
        now = time.monotonic()
        if (
            arm_key == getattr(self, '_last_impedance_check_arms', ())
            and now - getattr(self, '_last_impedance_check_at', 0.0)
            <= max_cache_age
        ):
            return
        left_state, right_state = self.get_arm_states_only()
        states = {'A': left_state, 'B': right_state}
        invalid = [
            f"{self.ARM_SIDE[arm]} state={states[arm]}"
            for arm in arms
            if states[arm] != 3
        ]
        if invalid:
            raise RuntimeError(
                "Motion command rejected: impedance mode is not active "
                f"for the commanded arm(s) ({', '.join(invalid)})")
        unsafe_standby = [
            f"{self.ARM_SIDE[arm]} state={states[arm]}"
            for arm in standby_arms
            if states[arm] != 0
        ]
        if unsafe_standby:
            raise RuntimeError(
                "Motion command rejected: inactive arm is not in standby "
                f"({', '.join(unsafe_standby)})")
        self._last_impedance_check_at = now
        self._last_impedance_check_arms = arm_key

    def _require_position_mode(self, max_cache_age=0.1, arms=None,
                               standby_arms=None):
        """Require replay arms in state=1 and inactive arms in standby."""
        arms = self._normalize_arms(arms)
        standby_arms = (
            [] if not standby_arms
            else self._normalize_arms(standby_arms)
        )
        overlap = set(arms) & set(standby_arms)
        if overlap:
            raise ValueError(
                f"Arms cannot be active and standby simultaneously: {overlap}")
        arm_key = (tuple(arms), tuple(standby_arms))
        now = time.monotonic()
        if (
            arm_key == getattr(self, '_last_position_check_arms', ())
            and now - getattr(self, '_last_position_check_at', 0.0)
            <= max_cache_age
        ):
            return
        left_state, right_state = self.get_arm_states_only()
        states = {'A': left_state, 'B': right_state}
        invalid = [
            f"{self.ARM_SIDE[arm]} state={states[arm]}"
            for arm in arms
            if states[arm] != 1
        ]
        if invalid:
            raise RuntimeError(
                "Motion command rejected: position mode is not active "
                f"for the replay arm(s) ({', '.join(invalid)})")
        unsafe_standby = [
            f"{self.ARM_SIDE[arm]} state={states[arm]}"
            for arm in standby_arms
            if states[arm] != 0
        ]
        if unsafe_standby:
            raise RuntimeError(
                "Motion command rejected: inactive arm is not in standby "
                f"({', '.join(unsafe_standby)})")
        self._last_position_check_at = now
        self._last_position_check_arms = arm_key

    def set_impedance_mode(self, mode='joint', K=None, D=None,
                           velRatio=15, AccRatio=15, arms=None):
        """
        Set dual-arm impedance mode.

        Order: set impedance params and type first, then servo-on
        (state=3). Servoing on with stale params is what causes the
        first-frame jitter we hit in production.

        Args:
            mode: 'joint' or 'cart'.
            K: stiffness list (7 elements).
            D: damping list (7 elements).
        """
        arms = self._normalize_arms(arms)
        if mode not in ('cart', 'joint'):
            raise ValueError(f"Unsupported impedance mode: {mode!r}")

        if mode == 'cart':
            K = K or [8000, 8000, 8000, 100, 100, 100, 20]
            D = D or [0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4]
            impedance_type = 2
        else:
            K = K or [2, 2, 2, 1.6, 1, 1, 1]
            D = D or [0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2]
            impedance_type = 1

        # Match the field-tested reference implementation: load K/D and the
        # impedance type while the arm is still at its recovery target, then
        # switch state last. Entering state=3 before loading these parameters
        # caused an immediate asymmetric drift during the 2026-07-15 test.
        self.robot.clear_set()
        for arm in arms:
            if mode == 'cart':
                self.robot.set_cart_kd_params(arm=arm, K=K, D=D, type=2)
            else:
                self.robot.set_joint_kd_params(arm=arm, K=K, D=D)
            self.robot.set_impedance_type(arm=arm, type=impedance_type)
        self._send_cmd_wait_response(f"preload {mode} impedance parameters")
        time.sleep(0.5)

        self.robot.clear_set()
        for arm in arms:
            self.robot.set_state(arm=arm, state=3)
            self.robot.set_vel_acc(
                arm=arm, velRatio=velRatio, AccRatio=AccRatio)
        self._send_cmd_wait_response("enter impedance mode")

        # Poll until the selected arms reach state=3. SDK state=100 is a
        # terminal fault that requires an explicit clear; only 101-104/109 are
        # transition states.
        ok, diag = self._poll_arm_state_after_switch(
            target_state=3, timeout=5.0, arms=arms)
        if not ok:
            raise RuntimeError(f"set_impedance_mode: {diag}")

        # Seed torque mode from fresh measured joints before the controller
        # starts accepting tracker targets. Without this hold command the
        # low-stiffness arm sags under gravity immediately after state=3.
        current_left, current_right = self.get_current_joints()
        hold_targets = {'A': current_left, 'B': current_right}
        self.robot.clear_set()
        for arm in arms:
            self.robot.set_joint_cmd_pose(
                arm=arm, joints=list(hold_targets[arm]))
        self._send_cmd_wait_response("seed impedance hold from measured joints")

        # Parameter writes are not readable through this SDK, but cur_state
        # must remain in impedance after applying them.
        self._last_impedance_check_at = 0.0
        self._last_impedance_check_arms = ()
        self._require_impedance_mode(max_cache_age=0.0, arms=arms)
        label = ','.join(self.ARM_SIDE[arm] for arm in arms)
        self.logger.info(
            f"{label} {mode} impedance mode active (cur_state=3, K={K})")

    def _poll_arm_state_after_switch(self, target_state, timeout=5.0,
                                     poll_interval=0.05, arms=None):
        """Poll until selected arms reach target_state, fault, or timeout.

        The local 100343 SDK header defines state=100 as ARM_STATE_ERROR.
        It must fail immediately and be cleared explicitly by the operator;
        101/102/103/104/109 are the actual transition states.

        Returns: (ok: bool, diagnostic: str).
        """
        arms = self._normalize_arms(arms)
        indices = [self.ARM_INDEX[arm] for arm in arms]
        dcss = DCSS()
        deadline = time.monotonic() + timeout
        a_state, b_state, a_err, b_err = -1, -1, -1, -1
        while time.monotonic() < deadline:
            sub_data = self.robot.subscribe(dcss)
            a_state = sub_data["states"][0]["cur_state"]
            b_state = sub_data["states"][1]["cur_state"]
            a_err = sub_data["states"][0]["err_code"]
            b_err = sub_data["states"][1]["err_code"]
            states = [a_state, b_state]
            errors = [a_err, b_err]
            faults = []
            for arm, index in zip(arms, indices):
                state = int(states[index])
                error = int(errors[index])
                if state == self.ARM_STATE_ERROR or error != 0:
                    description = self.ARM_ERROR_DESCRIPTIONS.get(
                        error, 'unknown arm error')
                    faults.append(
                        f"{self.ARM_SIDE[arm]}(state={state}, "
                        f"err_code={error}: {description})")
            if faults:
                return False, (
                    "state-switch fault; explicit clear_error is required: "
                    + ", ".join(faults)
                )
            if all(states[index] == target_state for index in indices):
                return True, ""
            time.sleep(poll_interval)
        diag = (
            f"state-switch timed out after {timeout}s target={target_state} "
            f"left(state={a_state}, err_code={a_err}); "
            f"right(state={b_state}, err_code={b_err})"
        )
        return False, diag

    # ==================== Cartesian-space control ====================

    def move_to_pose_direct(self, left_pose=None, right_pose=None, unit='mm', send=True):
        """
        Cartesian-space control: IK-solve both arms and dispatch joint commands.
        Non-blocking, suitable for realtime tracking.

        Args:
            left_pose: [X, Y, Z, RX, RY, RZ] left target pose; None to skip the arm.
            right_pose: [X, Y, Z, RX, RY, RZ] right target pose; None to skip.
            unit: 'mm' or 'm'.
            send: if False, only run IK and return the solved joints without
                pushing them to the controller. Used by the handoff ramp in
                tianji_arm_node, which needs to blend the IK target with a
                start snapshot before dispatching via move_to_joints_direct.

        Returns:
            tuple: (left_success, right_success, left_joints, right_joints).
        """
        call_started_ns = time.perf_counter_ns()
        left_mm = None
        right_mm = None
        if left_pose is not None:
            left_mm = list(left_pose)
            if unit == 'm':
                for i in range(3):
                    left_mm[i] *= 1000
        if right_pose is not None:
            right_mm = list(right_pose)
            if unit == 'm':
                for i in range(3):
                    right_mm[i] *= 1000

        read_started_ns = time.perf_counter_ns()
        ref_left, ref_right = self.get_current_joints()
        read_finished_ns = time.perf_counter_ns()

        left_success = False
        right_success = False
        left_joints = None
        right_joints = None

        if left_mm is not None:
            try:
                left_mat = self.kine_left.xyzabc_to_mat4x4(left_mm)
                left_ik = self.kine_left.ik(
                    robot_serial=0,
                    pose_mat=left_mat,
                    ref_joints=ref_left,
                    zsp_type=self.zsp_type,
                    zsp_para=self.left_zsp_para,
                    zsp_angle=self.zsp_angle,
                    dgr=self.dgr,
                )
                if left_ik is not False:
                    if not left_ik.m_Output_IsOutRange and not left_ik.m_Output_IsJntExd:
                        left_joints = left_ik.m_Output_RetJoint.to_list()
                        left_success = True
            except Exception as e:
                self.logger.debug(f"left IK exception: {e}")

        if right_mm is not None:
            try:
                right_mat = self.kine_right.xyzabc_to_mat4x4(right_mm)
                right_ik = self.kine_right.ik(
                    robot_serial=1,
                    pose_mat=right_mat,
                    ref_joints=ref_right,
                    zsp_type=self.zsp_type,
                    zsp_para=self.right_zsp_para,
                    zsp_angle=self.zsp_angle,
                    dgr=self.dgr,
                )
                if right_ik is not False:
                    if not right_ik.m_Output_IsOutRange and not right_ik.m_Output_IsJntExd:
                        right_joints = right_ik.m_Output_RetJoint.to_list()
                        right_success = True
            except Exception as e:
                self.logger.debug(f"right IK exception: {e}")

        ik_finished_ns = time.perf_counter_ns()
        self._last_pose_timing = {
            'total_ms': (ik_finished_ns - call_started_ns) / 1e6,
            'sdk_reference_read_ms': (
                read_finished_ns - read_started_ns
            ) / 1e6,
            'ik_compute_ms': (ik_finished_ns - read_finished_ns) / 1e6,
        }

        if left_joints is not None:
            left_joints_str = ', '.join([f'{j:7.2f}' for j in left_joints])
            self.logger.debug(f"[LEFT_IK]  joints: [{left_joints_str}]")
        else:
            self.logger.debug("[LEFT_IK]  FAILED!")

        if right_joints is not None:
            right_joints_str = ', '.join([f'{j:7.2f}' for j in right_joints])
            self.logger.debug(f"[RIGHT_IK] joints: [{right_joints_str}]")
        else:
            self.logger.debug("[RIGHT_IK] FAILED!")

        if send:
            command_arms = []
            if left_joints is not None:
                command_arms.append('A')
            if right_joints is not None:
                command_arms.append('B')
            if not command_arms:
                return left_success, right_success, left_joints, right_joints
            self._require_impedance_mode(arms=command_arms)
            self.robot.clear_set()
            if left_joints is not None:
                self.robot.set_joint_cmd_pose(arm='A', joints=left_joints)
            if right_joints is not None:
                self.robot.set_joint_cmd_pose(arm='B', joints=right_joints)
            self.robot.send_cmd()

        return left_success, right_success, left_joints, right_joints

    # ==================== Joint-space control ====================

    @staticmethod
    def _joint_vector(values, label):
        vector = np.asarray(values, dtype=float)
        if vector.shape != (7,) or not np.all(np.isfinite(vector)):
            raise RuntimeError(f"{label} must contain 7 finite joint values")
        return vector

    def _validate_joint_limits(self, arm, joints, margin_deg=0.0):
        arm = self._normalize_arms([arm])[0]
        vector = self._joint_vector(joints, f"{self.ARM_SIDE[arm]} joints")
        limits = (
            self._joint_limits_left if arm == 'A'
            else self._joint_limits_right
        )
        for index, (value, (lower, upper)) in enumerate(zip(vector, limits)):
            if value < lower + margin_deg or value > upper - margin_deg:
                raise RuntimeError(
                    f"{self.ARM_SIDE[arm]} joint {index + 1}={value:.2f} deg "
                    f"outside recovery limit [{lower + margin_deg:.2f}, "
                    f"{upper - margin_deg:.2f}]")
        return vector

    def _validated_recovery_feedback(
        self,
        arm,
        snapshot,
        encoder_agreement_deg,
    ):
        """Validate motor/external encoder feedback for one recovery frame."""
        arm = self._normalize_arms([arm])[0]
        index = self.ARM_INDEX[arm]
        side = self.ARM_SIDE[arm]
        primary = self._validate_joint_limits(
            arm, snapshot['joints'][index])
        try:
            external_values = snapshot['external_joints'][index]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"{side} recovery requires external-encoder feedback"
            ) from exc
        external = self._validate_joint_limits(arm, external_values)
        disagreement = float(np.max(np.abs(primary - external)))
        if disagreement > encoder_agreement_deg:
            joint = int(np.argmax(np.abs(primary - external))) + 1
            raise RuntimeError(
                f"{side} motor/external encoder disagreement "
                f"{disagreement:.2f} deg at joint {joint} exceeds "
                f"{encoder_agreement_deg:.2f} deg; recovery is refused")
        return primary, external

    def get_verified_init_feedback(
        self,
        encoder_agreement_deg=1.0,
        arms=None,
        active_arm='both',
    ):
        """Return init errors, dual-encoder verified for selected arms."""
        if encoder_agreement_deg <= 0.0:
            raise ValueError("encoder_agreement_deg must be positive")
        verified_arms = set(self._normalize_arms(arms))
        snapshot = self.get_arm_snapshot()
        init_left, init_right = self.get_init_joints(active_arm)
        references = (init_left, init_right)
        feedback = []
        errors = []
        for arm, reference in zip(('A', 'B'), references):
            index = self.ARM_INDEX[arm]
            primary = self._validate_joint_limits(
                arm, snapshot['joints'][index])
            external = primary
            if arm in verified_arms:
                primary, external = self._validated_recovery_feedback(
                    arm, snapshot, encoder_agreement_deg)
            target = self._validate_joint_limits(
                arm, reference, margin_deg=2.0)
            feedback.append(primary.tolist())
            errors.append(max(
                float(np.max(np.abs(target - primary))),
                float(np.max(np.abs(target - external))),
            ))
        return feedback, errors

    def seed_joint_target(self, arm, joints):
        """Cache a measured hold target while the selected arm is in standby."""
        arm = self._normalize_arms([arm])[0]
        vector = self._validate_joint_limits(arm, joints)
        self.robot.clear_set()
        self.robot.set_joint_cmd_pose(arm=arm, joints=vector.tolist())
        self._send_cmd_wait_response(
            f"seed {self.ARM_SIDE[arm]} recovery hold target", timeout_ms=50)

    def recover_arm_to_init(
        self,
        arm,
        *,
        target_joints=None,
        target_label='init',
        vel_ratio=10,
        acc_ratio=10,
        max_speed_deg_s=1.0,
        max_accel_deg_s2=2.0,
        dt=0.02,
        tracking_error_deg=2.0,
        command_lead_deg=0.5,
        reverse_motion_deg=0.2,
        arrival_tolerance_deg=0.5,
        encoder_agreement_deg=1.0,
        hold_sec=2.0,
        stall_timeout_sec=5.0,
        cancel_event=None,
    ):
        """Recover one arm to a guarded joint target in state=1.

        The non-selected arm receives no position-mode or joint commands. The
        command always stays at most one rate-limited step ahead of measured
        feedback, so a stalled arm cannot accumulate a distant open-loop
        target. By default the target is the configured init pose; an explicit
        target is used for inactive-arm parking.
        """
        if not getattr(self, '_has_send_wait_response', False):
            raise RuntimeError(
                "Guarded recovery requires Marvin SDK support for "
                "OnSetSendWaitResponse; legacy asynchronous OnSetSend is not "
                "accepted for position-mode motion")
        arm = self._normalize_arms([arm])[0]
        index = self.ARM_INDEX[arm]
        side = self.ARM_SIDE[arm]
        max_feedback_speed = max_speed_deg_s * 2.0
        if min(
            max_speed_deg_s,
            max_accel_deg_s2,
            dt,
            tracking_error_deg,
            command_lead_deg,
            reverse_motion_deg,
            arrival_tolerance_deg,
            encoder_agreement_deg,
            stall_timeout_sec,
        ) <= 0.0 or hold_sec < 0.0:
            raise ValueError("Recovery motion limits must be positive")
        if command_lead_deg >= tracking_error_deg:
            raise ValueError(
                "Recovery command lead must be smaller than tracking-error limit")

        snapshot = self.get_arm_snapshot()
        if snapshot['states'][index] != 0:
            raise RuntimeError(
                f"{side} recovery requires standby state=0, got "
                f"state={snapshot['states'][index]}")
        if snapshot['errors'][index] != 0:
            raise RuntimeError(
                f"{side} recovery rejected: arm error={snapshot['errors'][index]}")

        start, external_start = self._validated_recovery_feedback(
            arm, snapshot, encoder_agreement_deg)
        init_left, init_right = self.get_init_joints()
        reference = (
            init_left if arm == 'A' else init_right
        ) if target_joints is None else target_joints
        target_label = str(target_label).strip() or 'target'
        target = self._validate_joint_limits(
            arm, reference, margin_deg=2.0)
        initial_error = max(
            float(np.max(np.abs(target - start))),
            float(np.max(np.abs(target - external_start))),
        )
        started_at_target = initial_error <= arrival_tolerance_deg
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("recovery cancelled by stop request")
        self.seed_joint_target(arm, start)
        if started_at_target:
            self.logger.info(
                f"{side} already within {arrival_tolerance_deg:.2f} deg of "
                f"{target_label}; entering guarded state=1 hold to verify hardware "
                "feedback (no physical motion expected)")

        self.set_position_mode(
            velRatio=vel_ratio,
            AccRatio=acc_ratio,
            arms=[arm],
            seed_joints={arm: start.tolist()},
            recovery=True,
        )

        last_command = start.copy()
        hold_start = start.copy()
        feedback = start.copy()
        external_feedback = external_start.copy()
        hold_deadline = time.monotonic() + hold_sec
        while time.monotonic() < hold_deadline:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("recovery cancelled by stop request")
            snapshot = self.get_arm_snapshot()
            if snapshot['states'][index] != 1 or snapshot['errors'][index] != 0:
                raise RuntimeError(
                    f"{side} recovery hold lost state=1 or reported an error: "
                    f"state={snapshot['states'][index]}, "
                    f"error={snapshot['errors'][index]}")
            feedback_speed = float(np.max(np.abs(
                np.asarray(snapshot['velocities'][index], dtype=float))))
            if feedback_speed > max_feedback_speed:
                raise RuntimeError(
                    f"{side} recovery feedback speed {feedback_speed:.2f} deg/s "
                    f"exceeds {max_feedback_speed:.2f} deg/s")
            feedback, external_feedback = self._validated_recovery_feedback(
                arm, snapshot, encoder_agreement_deg)
            drift = max(
                float(np.max(np.abs(feedback - hold_start))),
                float(np.max(np.abs(external_feedback - external_start))),
            )
            if drift > 0.5:
                raise RuntimeError(
                    f"{side} moved {drift:.2f} deg while holding the seeded "
                    "recovery start pose")
            self.robot.clear_set()
            self.robot.set_joint_cmd_pose(arm=arm, joints=last_command.tolist())
            self._send_cmd_wait_response(
                f"hold {side} recovery start pose", timeout_ms=50)
            if cancel_event is not None:
                if cancel_event.wait(dt):
                    raise RuntimeError("recovery cancelled by stop request")
            else:
                time.sleep(dt)

        if started_at_target:
            final_error = max(
                float(np.max(np.abs(target - feedback))),
                float(np.max(np.abs(target - external_feedback))),
            )
            if final_error > arrival_tolerance_deg:
                raise RuntimeError(
                    f"{side} recovery verification drifted to "
                    f"{final_error:.2f} deg from {target_label}")
            self.logger.info(
                f"{side} recovery verified at {target_label} in guarded state=1; "
                f"max dual-encoder error={final_error:.2f} deg")
            return final_error

        # Advance an independent, rate-limited reference. The previous
        # implementation rebuilt every command as feedback + one 20 ms step;
        # that left too little following error for the position controller and
        # produced only ~0.15 deg/s from a requested 0.5 deg/s. Keep a bounded
        # look-ahead instead: enough for the controller to track, but always
        # below the tracking-error trip threshold.
        planned = start.copy()
        velocity = np.zeros(7, dtype=float)
        previous_feedback = hold_start.copy()
        reverse_accum = np.zeros(7, dtype=float)
        best_error = initial_error
        last_progress_at = time.monotonic()
        report_at = last_progress_at
        report_error = initial_error

        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("recovery cancelled by stop request")
            now = time.monotonic()

            snapshot = self.get_arm_snapshot()
            state = snapshot['states'][index]
            error_code = snapshot['errors'][index]
            if state != 1 or error_code != 0:
                raise RuntimeError(
                    f"{side} recovery lost state=1 or reported an error: "
                    f"state={state}, error={error_code}")
            feedback_speed = float(np.max(np.abs(
                np.asarray(snapshot['velocities'][index], dtype=float))))
            if feedback_speed > max_feedback_speed:
                raise RuntimeError(
                    f"{side} recovery feedback speed {feedback_speed:.2f} deg/s "
                    f"exceeds {max_feedback_speed:.2f} deg/s")
            feedback, external_feedback = self._validated_recovery_feedback(
                arm, snapshot, encoder_agreement_deg)
            tracking_error = float(np.max(np.abs(feedback - last_command)))
            if tracking_error > tracking_error_deg:
                raise RuntimeError(
                    f"{side} recovery tracking error {tracking_error:.2f} deg "
                    f"exceeds {tracking_error_deg:.2f} deg")

            remaining = target - feedback
            max_error = max(
                float(np.max(np.abs(remaining))),
                float(np.max(np.abs(target - external_feedback))),
            )
            if max_error <= arrival_tolerance_deg:
                break

            desired_sign = np.sign(target - previous_feedback)
            feedback_delta = feedback - previous_feedback
            wrong = (
                (desired_sign * feedback_delta < -1e-4)
                & (np.abs(remaining) > arrival_tolerance_deg)
            )
            reverse_accum[wrong] += np.abs(feedback_delta[wrong])
            reverse_accum[~wrong] = 0.0
            if float(np.max(reverse_accum)) > reverse_motion_deg:
                joint = int(np.argmax(reverse_accum)) + 1
                raise RuntimeError(
                    f"{side} joint {joint} moved in the opposite direction by "
                    f"{float(np.max(reverse_accum)):.2f} deg")

            if max_error < best_error - 0.1:
                best_error = max_error
                last_progress_at = now
            elif now - last_progress_at > stall_timeout_sec:
                raise RuntimeError(
                    f"{side} recovery stalled for {stall_timeout_sec:.1f}s "
                    f"with {max_error:.2f} deg remaining")

            if now - report_at >= 2.0:
                interval = now - report_at
                progress_rate = max(0.0, report_error - max_error) / interval
                eta = (
                    f"{max_error / progress_rate:.1f}s"
                    if progress_rate > 1e-3 else "unknown"
                )
                dominant_joint = int(np.argmax(np.abs(remaining))) + 1
                self.logger.info(
                    f"{side} recovery progress: remaining={max_error:.2f} deg, "
                    f"joint={dominant_joint}, feedback_speed="
                    f"{feedback_speed:.2f} deg/s, rate={progress_rate:.2f} "
                    f"deg/s, ETA={eta}")
                report_at = now
                report_error = max_error

            planned_remaining = target - planned
            desired_velocity = np.clip(
                planned_remaining / dt, -max_speed_deg_s, max_speed_deg_s)
            velocity = np.clip(
                desired_velocity,
                velocity - max_accel_deg_s2 * dt,
                velocity + max_accel_deg_s2 * dt,
            )
            planned_step = velocity * dt
            planned_step = np.sign(planned_remaining) * np.minimum(
                np.abs(planned_step), np.abs(planned_remaining))
            planned = self._validate_joint_limits(
                arm, planned + planned_step)

            bounded_target = np.clip(
                planned,
                feedback - command_lead_deg,
                feedback + command_lead_deg,
            )
            max_command_step = max_speed_deg_s * dt
            last_command = self._validate_joint_limits(
                arm,
                last_command + np.clip(
                    bounded_target - last_command,
                    -max_command_step,
                    max_command_step,
                ),
            )

            self.robot.clear_set()
            self.robot.set_joint_cmd_pose(
                arm=arm, joints=last_command.tolist())
            self._send_cmd_wait_response(
                f"{side} feedback-limited recovery command", timeout_ms=50)
            previous_feedback = feedback
            if cancel_event is not None:
                if cancel_event.wait(dt):
                    raise RuntimeError("recovery cancelled by stop request")
            else:
                time.sleep(dt)

        final_error = max_error
        settle_deadline = time.monotonic() + hold_sec
        while time.monotonic() < settle_deadline:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("recovery cancelled by stop request")
            snapshot = self.get_arm_snapshot()
            feedback_speed = float(np.max(np.abs(
                np.asarray(snapshot['velocities'][index], dtype=float))))
            if feedback_speed > max_feedback_speed:
                raise RuntimeError(
                    f"{side} recovery feedback speed {feedback_speed:.2f} deg/s "
                    f"exceeds {max_feedback_speed:.2f} deg/s")
            feedback, external_feedback = self._validated_recovery_feedback(
                arm, snapshot, encoder_agreement_deg)
            final_error = max(
                float(np.max(np.abs(target - feedback))),
                float(np.max(np.abs(target - external_feedback))),
            )
            if snapshot['states'][index] != 1 or snapshot['errors'][index] != 0:
                raise RuntimeError(
                    f"{side} recovery settle lost state=1 or reported an error")
            if final_error > arrival_tolerance_deg:
                raise RuntimeError(
                    f"{side} recovery settle drifted to {final_error:.2f} deg")
            if cancel_event is not None:
                if cancel_event.wait(dt):
                    raise RuntimeError("recovery cancelled by stop request")
            else:
                time.sleep(dt)

        self.logger.info(
            f"{side} recovery reached {target_label}; "
            f"max error={final_error:.2f} deg")
        return final_error

    def move_to_joints_direct(self, left_joints=None, right_joints=None,
                              active_arms=None, standby_arms=None,
                              command_state=3):
        """
        Joint-space control: dispatch joint angle commands directly.
        Non-blocking, suitable for realtime tracking.

        Args:
            left_joints: [j1..j7] left target angles (degrees); None to skip.
            right_joints: [j1..j7] right target angles (degrees); None to skip.
            active_arms: SDK arms required in ``command_state``. None
                preserves the legacy dual-arm requirement.
            standby_arms: Optional SDK arm names that must remain in state=0.
            command_state: Marvin state required before dispatch. Normal
                teleoperation uses impedance state=3; explicit external replay
                may use position state=1.

        Returns:
            tuple: (left_success, right_success).
        """
        call_started_ns = time.perf_counter_ns()
        left_success = left_joints is not None
        right_success = right_joints is not None
        command_arms = []
        if left_success:
            command_arms.append('A')
        if right_success:
            command_arms.append('B')
        if not command_arms:
            self._last_joint_command_timing = {
                'total_ms': 0.0,
                'state_guard_ms': 0.0,
                'prepare_ms': 0.0,
                'sdk_send_ms': 0.0,
            }
            return left_success, right_success
        required_active_arms = self._normalize_arms(active_arms)
        if not set(command_arms).issubset(required_active_arms):
            raise RuntimeError(
                "Joint target includes an arm outside active_arms: "
                f"commanded={command_arms}, active={required_active_arms}")
        guard_started_ns = time.perf_counter_ns()
        if int(command_state) == 3:
            self._require_impedance_mode(
                arms=required_active_arms,
                standby_arms=standby_arms,
            )
        elif int(command_state) == 1:
            self._require_position_mode(
                arms=required_active_arms,
                standby_arms=standby_arms,
            )
        else:
            raise ValueError(
                f"command_state must be Marvin state 1 or 3, got "
                f"{command_state!r}")
        guard_finished_ns = time.perf_counter_ns()

        self.robot.clear_set()
        if left_joints is not None:
            self.robot.set_joint_cmd_pose(arm='A', joints=list(left_joints))
            left_joints_str = ', '.join([f'{j:7.2f}' for j in left_joints])
            self.logger.debug(f"[LEFT]  joints: [{left_joints_str}]")
        if right_joints is not None:
            self.robot.set_joint_cmd_pose(arm='B', joints=list(right_joints))
            right_joints_str = ', '.join([f'{j:7.2f}' for j in right_joints])
            self.logger.debug(f"[RIGHT] joints: [{right_joints_str}]")
        send_started_ns = time.perf_counter_ns()
        try:
            self.robot.send_cmd()
        finally:
            send_finished_ns = time.perf_counter_ns()
            self._last_joint_command_timing = {
                'total_ms': (send_finished_ns - call_started_ns) / 1e6,
                'state_guard_ms': (
                    guard_finished_ns - guard_started_ns
                ) / 1e6,
                'prepare_ms': (
                    send_started_ns - guard_finished_ns
                ) / 1e6,
                'sdk_send_ms': (
                    send_finished_ns - send_started_ns
                ) / 1e6,
            }

        return left_success, right_success

    def move_to_joints_smooth(self, left_target=None, right_target=None,
                              duration=25.0, dt=0.02,
                              max_tracking_error_deg=15.0,
                              cancel_event=None):
        """
        Smoothly move both arms to the target joint angles using a quintic blend.

        Args:
            left_target: [j1..j7] left target angles; None keeps current.
            right_target: [j1..j7] right target angles; None keeps current.
            duration: trajectory duration (s); larger = slower / smoother.
            dt: interpolation step (s).

        Returns:
            bool: success.
        """
        if duration <= 0.0 or dt <= 0.0:
            raise ValueError("duration and dt must be positive")
        self._require_impedance_mode(max_cache_age=0.0)

        left_joints, right_joints = self.get_current_joints()
        start_left = list(left_joints)
        start_right = list(right_joints)

        if left_target is None:
            left_target = start_left
        if right_target is None:
            right_target = start_right

        num_points = max(1, int(duration / dt))

        self.logger.debug(f"Smooth move to target ({duration}s quintic blend)...")

        for i in range(num_points + 1):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("smooth motion cancelled by stop request")
            t = i / num_points
            s = 10 * (t ** 3) - 15 * (t ** 4) + 6 * (t ** 5)

            target_left = [
                start_left[j] + s * (left_target[j] - start_left[j])
                for j in range(7)
            ]
            target_right = [
                start_right[j] + s * (right_target[j] - start_right[j])
                for j in range(7)
            ]

            self.robot.clear_set()
            self.robot.set_joint_cmd_pose(arm='A', joints=target_left)
            self.robot.set_joint_cmd_pose(arm='B', joints=target_right)
            self._send_cmd_wait_response("smooth joint command", timeout_ms=50)

            # At 5 Hz, verify both the active mode and feedback tracking. A
            # large residual indicates an unexpected obstruction, wrong
            # direction, or ineffective impedance command; continuing would
            # only accumulate more motion error.
            if i % max(1, int(0.2 / dt)) == 0 or i == num_points:
                self._require_impedance_mode(max_cache_age=0.0)
                feedback_left, feedback_right = self.get_current_joints()
                left_error = max(abs(feedback_left[j] - target_left[j]) for j in range(7))
                right_error = max(abs(feedback_right[j] - target_right[j]) for j in range(7))
                if max(left_error, right_error) > max_tracking_error_deg:
                    raise RuntimeError(
                        "smooth motion tracking error exceeded safety limit: "
                        f"left={left_error:.1f} deg, right={right_error:.1f} deg, "
                        f"limit={max_tracking_error_deg:.1f} deg")

            if cancel_event is not None:
                if cancel_event.wait(dt):
                    raise RuntimeError("smooth motion cancelled by stop request")
            else:
                time.sleep(dt)

        return True

    # ==================== Home pose / release ====================

    def move_to_init(self, wait=True, timeout=3.0, duration=25.0, dt=0.02,
                     cancel_event=None):
        """Reject the legacy impedance-mode init trajectory.

        Init recovery must run through recover_arm_to_init(), one arm at a
        time, from state=0. Keep this method as a fail-closed compatibility
        shim so older callers cannot reintroduce the 2026-07-15 failure path.
        """
        del wait, timeout, duration, dt, cancel_event
        raise RuntimeError(
            "Legacy move_to_init is disabled; use guarded recover_arm_to_init")

    def set_standby(self, arms=None):
        """Servo-off (state=0), keep connection, ready to resume. State-driven (waits cur_state=0)."""
        arms = self._normalize_arms(arms)
        label = ','.join(self.ARM_SIDE[arm] for arm in arms)
        self.logger.info(f"{label} STANDBY (state=0, connection preserved)")
        self.robot.clear_set()
        for arm in arms:
            self.robot.set_state(arm=arm, state=0)
        self._send_cmd_wait_response("enter standby")

        ok, diag = self._poll_arm_state_after_switch(
            target_state=0, timeout=5.0, arms=arms)
        if not ok:
            raise RuntimeError(f"set_standby: {diag}")
        self._last_impedance_check_at = 0.0
        self._last_impedance_check_arms = ()
        self._last_position_check_at = 0.0
        self._last_position_check_arms = ()
        self.logger.info(f"{label} STANDBY reached (cur_state=0)")

    def set_active(self, mode='joint', K=None, D=None,
                   velRatio=15, AccRatio=15, arms=None):
        """Servo-on (state=3), apply impedance params. Safe to call any time."""
        arms = self._normalize_arms(arms)
        try:
            label = ','.join(self.ARM_SIDE[arm] for arm in arms)
            self.logger.info(f"{label} ACTIVE (impedance mode)")
            status = self.get_arm_status()
            unsafe = []
            for arm in arms:
                side = self.ARM_SIDE[arm]
                arm_status = status[side]
                servo_errors = arm_status.get('servo_errors') or []
                servo_faults = [
                    code for code in servo_errors
                    if str(code).strip().lower() not in ('', '0', '0x0', '0x0000')
                ]
                if (int(arm_status['state']) != 0
                        or int(arm_status['err_code']) != 0
                        or servo_faults):
                    unsafe.append(
                        f"{side}(state={arm_status['state']}, "
                        f"err_code={arm_status['err_code']}, "
                        f"servo_errors={servo_faults})")
            if unsafe:
                raise RuntimeError(
                    "impedance enable requires clean standby feedback; "
                    "clear errors explicitly before retrying: " + ', '.join(unsafe))
            self.logger.info(
                "clean standby feedback confirmed; skipping clear_error "
                "before impedance enable")
            self.set_impedance_mode(
                mode=mode, K=K, D=D,
                velRatio=velRatio, AccRatio=AccRatio, arms=arms)
        except Exception as exc:
            try:
                self.set_standby(arms=arms)
            except Exception as standby_exc:
                raise RuntimeError(
                    f"impedance enable failed ({exc}); CRITICAL: standby "
                    f"recovery also failed ({standby_exc})") from exc
            raise

    def set_position_mode(self, velRatio=10, AccRatio=10, arms=None,
                          seed_joints=None, recovery=False, replay=False):
        """Enter state=1 for guarded recovery or explicit external replay.

        ``replay`` is deliberately separate from the normal impedance enable
        path. The ROS controller permits it only for an explicit external
        source, so Tracker teleoperation cannot accidentally enter rigid
        position mode.
        """
        if bool(recovery) == bool(replay):
            raise RuntimeError(self.RIGID_MODE_DISABLED)
        if not 1 <= int(velRatio) <= self.MAX_RECOVERY_VEL_RATIO:
            raise ValueError(
                f"Recovery velRatio must be in [1, {self.MAX_RECOVERY_VEL_RATIO}]")
        if not 1 <= int(AccRatio) <= self.MAX_RECOVERY_ACC_RATIO:
            raise ValueError(
                f"Recovery AccRatio must be in [1, {self.MAX_RECOVERY_ACC_RATIO}]")
        arms = self._normalize_arms(arms)
        snapshot = self.get_arm_snapshot()
        for arm in arms:
            index = self.ARM_INDEX[arm]
            if snapshot['states'][index] != 0:
                raise RuntimeError(
                    f"{self.ARM_SIDE[arm]} position recovery requires state=0, "
                    f"got {snapshot['states'][index]}")

        seeds = seed_joints or {}
        for arm in arms:
            index = self.ARM_INDEX[arm]
            seed = seeds.get(arm, snapshot['joints'][index])
            self.seed_joint_target(arm, seed)

        self.robot.clear_set()
        for arm in arms:
            self.robot.set_vel_acc(
                arm=arm, velRatio=int(velRatio), AccRatio=int(AccRatio))
            self.robot.set_state(arm=arm, state=1)
        context = "replay" if replay else "guarded recovery"
        self._send_cmd_wait_response(f"enter {context} position mode")
        ok, diag = self._poll_arm_state_after_switch(
            target_state=1, timeout=5.0, arms=arms)
        if not ok:
            raise RuntimeError(f"set_position_mode {context}: {diag}")
        self._last_position_check_at = 0.0
        self._last_position_check_arms = ()

    def release_brake(self, arm: str = 'A'):
        """Release the brake. Direct SDK call, not via clear_set/send_cmd batching."""
        param = 'BRAK0' if arm == 'A' else 'BRAK1'
        self.robot.set_param('int', param, 2)
        self.logger.info(f"{'left' if arm == 'A' else 'right'} arm brake released")

    def hold_brake(self, arm: str = 'A'):
        """Engage the brake. Direct SDK call."""
        param = 'BRAK0' if arm == 'A' else 'BRAK1'
        self.robot.set_param('int', param, 1)
        self.logger.info(f"{'left' if arm == 'A' else 'right'} arm brake held")

    def clear_arm_error(self, arm: str = 'A'):
        """Clear error state on a single arm."""
        side = 'left' if arm == 'A' else 'right'
        self.robot.clear_set()
        self.robot.clear_error(arm)
        self.robot.send_cmd()
        time.sleep(0.5)
        self.logger.info(f"{side} arm error cleared (clear_error {arm})")

    def disable_and_release(self, poll_timeout=3.0):
        """Shutdown path: clear errors, request set_state(0), poll until the
        firmware confirms cur_state==0 (or `poll_timeout` elapses), then
        release the TCP session.

        We do NOT call self.set_standby() here because its poll-and-raise
        loop is built for the runtime enable path (where a failed state
        switch is actionable). On shutdown a raise would skip release_robot()
        and leak the Marvin TCP session — which is exactly what users hit
        as "Stop won't power down". So we poll, log on timeout, and still
        release. A fixed sleep was tried and proved unreliable: the firmware
        sometimes takes longer than 2s to acknowledge state=0, especially
        when transitioning out of impedance (state=3).
        """
        self.logger.info("Disabling arms (set_state=0)...")
        try:
            # Empirically, calling clear_error() BEFORE set_state(0) prevents
            # the firmware from honoring the state transition (the arms stay
            # at cur_state=3 indefinitely; cf. the smoke test in this commit
            # message). set_standby() works because it never clear_errors —
            # mirror that. If the user really needs errors cleared on
            # shutdown, do it AFTER cur_state reaches 0.
            self.robot.clear_set()
            self.robot.set_state(arm='A', state=0)
            self.robot.set_state(arm='B', state=0)
            ack = int(self.robot.send_cmd_wait_response(100))
            if ack <= 0:
                self.logger.warning(
                    "set_state(0) shutdown acknowledgement failed: result=%d", ack)
        except Exception as exc:
            self.logger.warning("set_state(0) on shutdown failed (ignored): %s", exc)

        a, b = -1, -1
        try:
            dcss = DCSS()
            deadline = time.monotonic() + max(0.5, float(poll_timeout))
            while time.monotonic() < deadline:
                sub = self.robot.subscribe(dcss)
                a = sub["states"][0]["cur_state"]
                b = sub["states"][1]["cur_state"]
                if a == 0 and b == 0:
                    self.logger.info("Both arms at cur_state=0")
                    break
                time.sleep(0.05)
            else:
                self.logger.warning(
                    "Servos may still be on (cur_state left=%d, right=%d); "
                    "releasing TCP anyway", a, b)
        except Exception as exc:
            self.logger.warning("cur_state poll failed (ignored): %s", exc)

        self.logger.debug("Releasing connection...")
        try:
            self.robot.release_robot()
        except Exception as exc:
            self.logger.warning("release_robot failed (ignored): %s", exc)
        self.logger.info("Safely exited")
