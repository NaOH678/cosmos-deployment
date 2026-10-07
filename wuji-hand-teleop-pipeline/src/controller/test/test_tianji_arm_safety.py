from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from sensor_msgs.msg import JointState

import controller.tianji_arm_node as tianji_arm_node
from controller.common import ArmState
from controller.tianji_arm_node import (
    EXTERNAL_HANDOFF_ACTIVE,
    EXTERNAL_HANDOFF_COMPLETE,
    EXTERNAL_HANDOFF_WAITING_TARGET,
    LIFECYCLE_CLUTCH_DISCONNECTED,
    LIFECYCLE_DISABLED,
    LIFECYCLE_ENABLE_FAILED,
    LIFECYCLE_RECOVERING,
    LIFECYCLE_RECOVERY_FAILED,
    LIFECYCLE_RECOVERY_PARTIAL,
    LIFECYCLE_RECOVERY_READY,
    LIFECYCLE_SDK_ERROR,
    LIFECYCLE_TARGET_HOLD,
    TianjiArmControllerNode,
)


class FakeLogger:
    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


class FakeController:
    RIGID_MODE_DISABLED = 'rigid available only during guarded recovery'

    def __init__(self, current=None, reference=None, active_error=None,
                 recovery_error=None):
        self.current = [
            list(values) for values in (
                current or ([0.0] * 7, [0.0] * 7))
        ]
        self.reference = [
            list(values) for values in (
                reference or ([0.0] * 7, [0.0] * 7))
        ]
        self.states = [0, 0]
        self.active_error = active_error
        self.recovery_error = recovery_error
        self.calls = []

    @staticmethod
    def _normalize_arms(arms=None):
        return ['A', 'B'] if arms is None else list(arms)

    def get_arm_states_only(self):
        self.calls.append('get_arm_states_only')
        return tuple(self.states)

    def set_active(self, **kwargs):
        self.calls.append(('set_active', kwargs))
        if self.active_error is not None:
            raise self.active_error
        for arm in self._normalize_arms(kwargs.get('arms')):
            self.states[0 if arm == 'A' else 1] = 3

    def set_position_mode(self, **kwargs):
        self.calls.append(('set_position_mode', kwargs))
        if self.active_error is not None:
            raise self.active_error
        for arm in self._normalize_arms(kwargs.get('arms')):
            self.states[0 if arm == 'A' else 1] = 1

    def recover_arm_to_init(self, arm, **kwargs):
        self.calls.append(('recover_arm_to_init', arm, kwargs))
        if self.recovery_error is not None:
            raise self.recovery_error
        index = 0 if arm == 'A' else 1
        self.states[index] = 1
        target = kwargs.get('target_joints', self.reference[index])
        self.current[index] = list(target)
        return 0.0

    def get_current_joints(self):
        self.calls.append('get_current_joints')
        return tuple(list(values) for values in self.current)

    def get_init_joints(self, active_arm='both'):
        self.calls.append(('get_init_joints', active_arm))
        return tuple(list(values) for values in self.reference)

    def get_park_joints(self, arm):
        self.calls.append(('get_park_joints', arm))
        if arm != 'A':
            raise ValueError(f'no parking pose for {arm}')
        return [90.0, -90.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def get_verified_init_feedback(
        self,
        encoder_agreement_deg=1.0,
        arms=None,
        active_arm='both',
    ):
        self.calls.append(
            (
                'get_verified_init_feedback', encoder_agreement_deg, arms,
                active_arm,
            ))
        errors = [
            max(abs(value - target) for value, target in zip(values, reference))
            for values, reference in zip(self.current, self.reference)
        ]
        return (
            [list(values) for values in self.current],
            errors,
        )

    def set_standby(self, arms=None):
        normalized = self._normalize_arms(arms)
        self.calls.append(('set_standby', tuple(normalized)))
        for arm in normalized:
            self.states[0 if arm == 'A' else 1] = 0


def test_external_handoff_freezes_first_valid_ik_target(monkeypatch):
    current_time = [10.0]
    monkeypatch.setattr(
        tianji_arm_node.time, 'monotonic', lambda: current_time[0])
    published_states = []
    publisher = SimpleNamespace(
        publish=lambda message: published_states.append(message.data))
    node = SimpleNamespace(
        _external_handoff_state=EXTERNAL_HANDOFF_WAITING_TARGET,
        _external_handoff_target_left=None,
        _external_handoff_target_right=None,
        _external_handoff_completed_at=0.0,
        _handoff_start_at=None,
        _handoff_start_left=None,
        _handoff_start_right=[0.0] * 7,
        _handoff_hold_sec=0.2,
        _handoff_ramp_sec=1.0,
        _external_lock=threading.Lock(),
        _external_stream_started=True,
        _control_source='external',
        _external_handoff_gate_enabled=True,
        _handoff_state_pub=publisher,
        _control_sides=lambda: ('right',),
        get_logger=lambda: FakeLogger(),
    )
    node._external_handoff_payload = lambda: (
        TianjiArmControllerNode._external_handoff_payload(node))
    node._publish_external_handoff_state = lambda state=None: (
        TianjiArmControllerNode._publish_external_handoff_state(node, state))
    node._blend_joints = lambda start, target, s: (
        TianjiArmControllerNode._blend_joints(start, target, s))

    first = TianjiArmControllerNode._external_handoff_joint_targets(
        node, None, [10.0] * 7)
    assert first == (None, [0.0] * 7)
    assert node._external_handoff_state == EXTERNAL_HANDOFF_ACTIVE

    # A moving policy target cannot change the ramp endpoint once latched.
    current_time[0] = 10.7
    _, halfway = TianjiArmControllerNode._external_handoff_joint_targets(
        node, None, [99.0] * 7)
    assert 0.0 < halfway[0] < 10.0

    current_time[0] = 11.3
    completed = TianjiArmControllerNode._external_handoff_joint_targets(
        node, None, [99.0] * 7)
    assert completed == (None, [10.0] * 7)
    assert node._external_handoff_state == EXTERNAL_HANDOFF_COMPLETE
    assert len(published_states) == 2


def make_node(controller, active_arm='both'):
    lifecycle = []
    active_config = {
        'both': (('left', 'right'), ('A', 'B')),
        'left': (('left',), ('A',)),
        'right': (('right',), ('B',)),
    }
    configured_sides, configured_sdk_arms = active_config[active_arm]
    node = SimpleNamespace(
        controller=controller,
        _active_arm_mode=active_arm,
        _configured_sides=configured_sides,
        _configured_sdk_arms=configured_sdk_arms,
        _enable_lock=threading.Lock(),
        _enable_cancel_event=threading.Event(),
        _enable_in_progress=threading.Event(),
        _enable_thread=None,
        _recovery_cancel_event=threading.Event(),
        _recovery_in_progress=threading.Event(),
        _recovery_thread=None,
        _recovery_complete=False,
        _arm_enabled=False,
        _arm_state=ArmState.IMPEDANCE,
        left_pose=None,
        right_pose=None,
        _impedance_velocity_ratio=15,
        _impedance_acceleration_ratio=15,
        _impedance_k=np.array([14.0, 14.0, 14.0, 10.5, 5.6, 5.6, 5.6]),
        _impedance_d=np.array([0.3] * 7),
        _enable_init_tolerance_deg=3.0,
        _recovery_velocity_ratio=10,
        _recovery_acceleration_ratio=10,
        _recovery_max_speed_deg_s=1.0,
        _recovery_max_accel_deg_s2=2.0,
        _recovery_control_period_sec=0.02,
        _recovery_tracking_error_deg=2.0,
        _recovery_command_lead_deg=0.5,
        _recovery_reverse_motion_deg=0.2,
        _recovery_arrival_tolerance_deg=0.5,
        _recovery_encoder_agreement_deg=1.0,
        _recovery_hold_sec=2.0,
        _recovery_stall_timeout_sec=5.0,
        _impedance_stability_sec=3.0,
        _impedance_max_drift_deg=1.0,
        _replay_first_joint_targets={},
        _publish_lifecycle=lifecycle.append,
        _publish_teleop_status=lambda _status: None,
        _arm_handoff_snapshot=lambda: None,
        _capture_teleop_neutral=lambda: None,
        _capture_external_neutral=lambda: None,
        _reset_control_targets=lambda: None,
        get_logger=lambda: FakeLogger(),
        lifecycle=lifecycle,
    )
    node._control_sides = lambda: TianjiArmControllerNode._control_sides(node)
    node._control_sdk_arms = lambda: TianjiArmControllerNode._control_sdk_arms(node)
    node._control_indices = lambda: TianjiArmControllerNode._control_indices(node)
    node._init_pose_errors = lambda: TianjiArmControllerNode._init_pose_errors(node)
    node._monitor_impedance_stability = lambda: None
    node._move_to_replay_first_qpos = lambda: (
        TianjiArmControllerNode._move_to_replay_first_qpos(node)
    )
    node._motion_in_progress = lambda: (
        node._enable_in_progress.is_set()
        or node._recovery_in_progress.is_set()
    )
    node._do_disable = lambda: TianjiArmControllerNode._do_disable(node)
    return node


def test_enable_service_refuses_until_recovery_completes():
    node = make_node(FakeController())
    node._start_enable_worker = lambda: pytest.fail(
        'enable worker must not start before recovery')
    response = SimpleNamespace(success=None, message='')

    result = TianjiArmControllerNode._set_enabled_callback(
        node, SimpleNamespace(data=True), response)

    assert result.success is False
    assert 'recover_to_init' in result.message


def test_recovery_runs_selected_arms_serially_and_returns_to_standby():
    controller = FakeController(
        current=([40.0] * 7, [-30.0] * 7),
        reference=([1.0] * 7, [-1.0] * 7),
    )
    node = make_node(controller)

    errors, all_ready = TianjiArmControllerNode._do_recover_to_init(
        node, ['A', 'B'])

    recover_calls = [call for call in controller.calls if call[0] == 'recover_arm_to_init']
    assert [call[1] for call in recover_calls] == ['A', 'B']
    assert controller.states == [0, 0]
    assert node._recovery_complete is True
    assert errors == [0.0, 0.0]
    assert all_ready is True
    assert LIFECYCLE_RECOVERING in node.lifecycle
    assert LIFECYCLE_RECOVERY_READY in node.lifecycle


def test_recovery_failure_requests_dual_arm_standby():
    controller = FakeController(recovery_error=RuntimeError('tracking reversed'))
    node = make_node(controller)

    with pytest.raises(RuntimeError, match='tracking reversed'):
        TianjiArmControllerNode._do_recover_to_init(node, ['A'])

    assert controller.states == [0, 0]
    assert node._recovery_complete is False
    assert LIFECYCLE_RECOVERY_FAILED in node.lifecycle


def test_single_arm_recovery_can_finish_partial_without_enabling():
    controller = FakeController(
        current=([30.0] * 7, [20.0] * 7),
        reference=([0.0] * 7, [0.0] * 7),
    )
    node = make_node(controller)

    errors, all_ready = TianjiArmControllerNode._do_recover_to_init(
        node, ['A'])

    assert errors == [0.0, 20.0]
    assert all_ready is False
    assert node._recovery_complete is False
    assert LIFECYCLE_RECOVERY_PARTIAL in node.lifecycle


@pytest.mark.parametrize(
    ('active_arm', 'sdk_arm', 'active_index', 'inactive_index'),
    [('left', 'A', 0, 1), ('right', 'B', 1, 0)],
)
def test_single_arm_mode_recovers_and_enables_only_selected_arm(
    active_arm, sdk_arm, active_index, inactive_index
):
    current = ([25.0] * 7, [-30.0] * 7)
    reference = ([1.0] * 7, [-1.0] * 7)
    controller = FakeController(current=current, reference=reference)
    node = make_node(controller, active_arm=active_arm)

    errors, ready = TianjiArmControllerNode._do_recover_to_init(
        node, [sdk_arm])

    recover_calls = [
        call for call in controller.calls
        if isinstance(call, tuple) and call[0] == 'recover_arm_to_init'
    ]
    if active_arm == 'right':
        assert [call[1] for call in recover_calls] == ['A', 'B']
        assert recover_calls[0][2]['target_joints'] == [
            90.0, -90.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]
        assert recover_calls[0][2]['target_label'] == 'vertical park'
    else:
        assert [call[1] for call in recover_calls] == ['A']
    assert ready is True
    assert errors[active_index] == 0.0
    assert abs(errors[inactive_index]) > node._enable_init_tolerance_deg
    assert controller.states == [0, 0]
    assert node._recovery_complete is True

    TianjiArmControllerNode._do_enable(node)

    active_call = next(
        call for call in controller.calls
        if isinstance(call, tuple) and call[0] == 'set_active'
    )
    assert active_call[1]['arms'] == [sdk_arm]
    assert controller.states[active_index] == 3
    assert controller.states[inactive_index] == 0


def test_single_arm_controller_rejects_recovery_of_inactive_arm():
    controller = FakeController()
    node = make_node(controller, active_arm='left')
    node._start_recovery_worker = lambda _arms: pytest.fail(
        'inactive-arm recovery must not start')
    response = SimpleNamespace(success=None, message='')

    result = TianjiArmControllerNode._recover_callback(
        node, ['B'], response)

    assert result.success is False
    assert 'only recover_left_to_init' in result.message


@pytest.mark.parametrize(
    ('active_side', 'active_arm', 'inactive_arm'),
    [('left', 'A', 'B'), ('right', 'B', 'A')],
)
def test_single_arm_teleop_reads_and_commands_only_selected_side(
    active_side, active_arm, inactive_arm
):
    class _TeleopController:
        def __init__(self):
            self.left_zsp_para = None
            self.right_zsp_para = None
            self.pose_calls = []
            self.joint_calls = []

        def move_to_pose_direct(self, **kwargs):
            self.pose_calls.append(kwargs)
            left_target = [42.0] * 7 if kwargs['left_pose'] is not None else None
            right_target = [-42.0] * 7 if kwargs['right_pose'] is not None else None
            return left_target is not None, right_target is not None, left_target, right_target

        def move_to_joints_direct(self, **kwargs):
            self.joint_calls.append(kwargs)

    controller = _TeleopController()
    lookups = []
    node = SimpleNamespace(
        controller=controller,
        _control_source='tracker',
        left_pose=None,
        right_pose=None,
        left_y_axis=None,
        right_y_axis=None,
        _log_counter=0,
        _control_sides=lambda: (active_side,),
        _control_sdk_arms=lambda: (active_arm,),
        _lookup_transform=lambda target, source: (
            lookups.append((target, source)) or np.eye(4)
        ),
        _map_tracker_target=lambda _side, matrix: matrix,
        _matrix_to_pose=lambda _matrix: np.zeros(6),
        _map_upper_arm_direction=lambda _side, _matrix: np.array([0.0, 1.0, 0.0]),
        _handoff_blend_factor=lambda: None,
        _publish_alignment_target=lambda _left, _right: None,
        _publish_command=lambda _left, _right: None,
        _clear_target_hold=lambda: None,
        _publish_zsp_para_and_pose=lambda: None,
        _enter_target_hold=lambda reason: pytest.fail(reason),
        _on_sdk_error=lambda context, error: pytest.fail(f'{context}: {error}'),
        get_logger=lambda: FakeLogger(),
    )

    TianjiArmControllerNode._teleop_control(node)

    assert lookups == [
        (f'{active_side}_chest', f'tianji_{active_side}'),
        (f'{active_side}_chest', f'{active_side}_arm'),
    ]
    assert len(controller.joint_calls) == 1
    call = controller.joint_calls[0]
    assert call['active_arms'] == [active_arm]
    assert call['standby_arms'] == [inactive_arm]
    assert (call['left_joints'] is not None) == (active_side == 'left')
    assert (call['right_joints'] is not None) == (active_side == 'right')
    expected = [42.0] * 7 if active_side == 'left' else [-42.0] * 7
    assert call[f'{active_side}_joints'] == expected


def test_ik_failure_skips_only_failed_arm_for_that_frame():
    class _TeleopController:
        left_zsp_para = None
        right_zsp_para = None

        def __init__(self):
            self.joint_calls = []

        @staticmethod
        def move_to_pose_direct(**_kwargs):
            return False, True, None, [12.0] * 7

        def move_to_joints_direct(self, **kwargs):
            self.joint_calls.append(kwargs)

    controller = _TeleopController()
    node = SimpleNamespace(
        controller=controller,
        _control_source='tracker',
        left_pose=None,
        right_pose=None,
        left_y_axis=None,
        right_y_axis=None,
        _log_counter=0,
        _control_sides=lambda: ('left', 'right'),
        _control_sdk_arms=lambda: ('A', 'B'),
        _lookup_transform=lambda _target, _source: np.eye(4),
        _map_tracker_target=lambda _side, matrix: matrix,
        _matrix_to_pose=lambda _matrix: np.zeros(6),
        _map_upper_arm_direction=lambda _side, _matrix: np.array([0.0, 1.0, 0.0]),
        _handoff_blend_factor=lambda: None,
        _publish_alignment_target=lambda _left, _right: None,
        _publish_command=lambda _left, _right: None,
        _clear_target_hold=lambda: None,
        _publish_zsp_para_and_pose=lambda: None,
        _enter_target_hold=lambda reason: pytest.fail(reason),
        _on_sdk_error=lambda context, error: pytest.fail(f'{context}: {error}'),
        get_logger=lambda: FakeLogger(),
    )

    TianjiArmControllerNode._teleop_control(node)

    assert len(controller.joint_calls) == 1
    assert controller.joint_calls[0]['left_joints'] is None
    assert controller.joint_calls[0]['right_joints'] == [12.0] * 7


def test_enable_after_recovery_has_no_move_to_init_trajectory():
    controller = FakeController()
    node = make_node(controller)
    node._recovery_complete = True

    TianjiArmControllerNode._do_enable(node)

    assert node._arm_enabled is True
    assert any(
        isinstance(call, tuple) and call[0] == 'set_active'
        for call in controller.calls
    )
    assert not any(
        isinstance(call, tuple) and call[0] == 'move_to_init'
        for call in controller.calls
    )


def test_external_eef_replay_enables_position_mode_only():
    controller = FakeController()
    node = make_node(controller, active_arm='right')
    node._control_source = 'external'
    node._external_command_mode = 'eef'
    node._arm_hardware_mode = 'position'
    node._recovery_complete = True

    TianjiArmControllerNode._do_enable(node)

    position_calls = [
        call for call in controller.calls
        if isinstance(call, tuple) and call[0] == 'set_position_mode'
    ]
    assert position_calls == [(
        'set_position_mode',
        {
            'velRatio': node._recovery_velocity_ratio,
            'AccRatio': node._recovery_acceleration_ratio,
            'arms': ['B'],
            'replay': True,
        },
    )]
    assert controller.states == [0, 1]
    assert node._arm_state is ArmState.POSITION
    assert node._arm_enabled is True
    assert not any(
        isinstance(call, tuple) and call[0] == 'set_active'
        for call in controller.calls
    )


def test_absolute_replay_enable_moves_to_first_qpos_after_recovery():
    controller = FakeController(
        current=([0.0] * 7, [-1.0] * 7),
        reference=([0.0] * 7, [-1.0] * 7),
    )
    node = make_node(controller, active_arm='right')
    node._control_source = 'external'
    node._external_command_mode = 'eef'
    node._arm_hardware_mode = 'position'
    node._recovery_complete = True
    node._replay_first_joint_targets = {'right': [12.0] * 7}

    TianjiArmControllerNode._do_enable(node)

    first_qpos_call = next(
        call for call in controller.calls
        if isinstance(call, tuple)
        and call[0] == 'recover_arm_to_init'
    )
    assert first_qpos_call[1] == 'B'
    assert first_qpos_call[2]['target_joints'] == [12.0] * 7
    assert first_qpos_call[2]['target_label'] == 'episode first qpos'
    position_index = next(
        index for index, call in enumerate(controller.calls)
        if isinstance(call, tuple) and call[0] == 'set_position_mode'
    )
    first_qpos_index = controller.calls.index(first_qpos_call)
    assert first_qpos_index < position_index
    assert controller.current[1] == [12.0] * 7
    assert node._arm_enabled is True


def test_external_eef_replay_dispatches_ik_in_position_mode():
    class _ReplayController:
        left_zsp_para = None
        right_zsp_para = None

        def __init__(self):
            self.joint_calls = []

        @staticmethod
        def move_to_pose_direct(**_kwargs):
            return False, True, None, [8.0] * 7

        def move_to_joints_direct(self, **kwargs):
            self.joint_calls.append(kwargs)

    controller = _ReplayController()
    node = SimpleNamespace(
        controller=controller,
        _control_source='external',
        _external_command_mode='eef',
        _arm_hardware_mode='position',
        _external_handoff_gate_enabled=False,
        left_pose=None,
        right_pose=np.zeros(6),
        left_y_axis=None,
        right_y_axis=None,
        _log_counter=0,
        _control_sides=lambda: ('right',),
        _control_sdk_arms=lambda: ('B',),
        _refresh_external_targets=lambda: True,
        _handoff_blend_factor=lambda: None,
        _publish_alignment_target=lambda _left, _right: None,
        _publish_command=lambda _left, _right: None,
        _clear_target_hold=lambda: None,
        _publish_zsp_para_and_pose=lambda: None,
        _enter_target_hold=lambda reason: pytest.fail(reason),
        _on_sdk_error=lambda context, error: pytest.fail(f'{context}: {error}'),
        get_logger=lambda: FakeLogger(),
    )

    TianjiArmControllerNode._teleop_control(node)

    assert len(controller.joint_calls) == 1
    assert controller.joint_calls[0]['right_joints'] == [8.0] * 7
    assert controller.joint_calls[0]['active_arms'] == ['B']
    assert controller.joint_calls[0]['standby_arms'] == ['A']
    assert controller.joint_calls[0]['command_state'] == 1


def test_external_joint_handoff_gate_dispatches_directly_without_ik():
    class _JointController:
        def __init__(self):
            self.joint_calls = []
            self.pose_calls = 0

        def move_to_pose_direct(self, **_kwargs):
            self.pose_calls += 1
            raise AssertionError("joint deployment must not call IK")

        def move_to_joints_direct(self, **kwargs):
            self.joint_calls.append(kwargs)

    controller = _JointController()
    handoff_calls = []
    node = SimpleNamespace(
        controller=controller,
        _control_source='external',
        _external_command_mode='joint',
        _arm_hardware_mode='impedance',
        _external_handoff_gate_enabled=True,
        _external_handoff_state=EXTERNAL_HANDOFF_WAITING_TARGET,
        _control_sides=lambda: ('right',),
        _control_sdk_arms=lambda: ('B',),
        _refresh_external_joint_targets=lambda: (None, [5.0] * 7),
        _external_handoff_joint_targets=lambda left, right: (
            handoff_calls.append((left, right)) or (left, right)
        ),
        _handoff_blend_factor=lambda: pytest.fail(
            "gated joint deployment must use the fixed-target handoff"
        ),
        _publish_command=lambda _left, _right: None,
        _clear_target_hold=lambda: None,
        _enter_target_hold=lambda reason: pytest.fail(reason),
        _on_sdk_error=lambda context, error: pytest.fail(
            f'{context}: {error}'
        ),
        get_logger=lambda: FakeLogger(),
    )

    TianjiArmControllerNode._teleop_control(node)

    assert handoff_calls == [(None, [5.0] * 7)]
    assert controller.pose_calls == 0
    assert len(controller.joint_calls) == 1
    assert controller.joint_calls[0]['left_joints'] is None
    assert controller.joint_calls[0]['right_joints'] == [5.0] * 7
    assert controller.joint_calls[0]['active_arms'] == ['B']
    assert controller.joint_calls[0]['standby_arms'] == ['A']
    assert controller.joint_calls[0]['command_state'] == 3


def test_impedance_transition_failure_requests_standby_and_invalidates_recovery():
    controller = FakeController(active_error=RuntimeError('mode switch failed'))
    node = make_node(controller)
    node._recovery_complete = True

    with pytest.raises(RuntimeError, match='mode switch failed'):
        TianjiArmControllerNode._do_enable(node)

    assert controller.states == [0, 0]
    assert node._arm_enabled is False
    assert node._recovery_complete is False
    assert LIFECYCLE_ENABLE_FAILED in node.lifecycle


def test_impedance_stability_check_rejects_startup_drift():
    controller = FakeController()
    controller.states = [3, 3]
    samples = iter([
        ([0.0] * 7, [0.0] * 7),
        ([1.1] * 7, [0.0] * 7),
    ])
    controller.get_current_joints = lambda: next(samples)
    node = make_node(controller)
    node._impedance_stability_sec = 1.0
    node._impedance_max_drift_deg = 1.0

    with pytest.raises(RuntimeError, match='startup drift'):
        TianjiArmControllerNode._monitor_impedance_stability(node)


def test_stop_cancels_enable_and_recovery_and_invalidates_ready_state():
    controller = FakeController()
    node = make_node(controller)
    node._recovery_complete = True

    TianjiArmControllerNode._do_disable(node)

    assert node._enable_cancel_event.is_set()
    assert node._recovery_cancel_event.is_set()
    assert node._recovery_complete is False
    assert controller.states == [0, 0]
    assert LIFECYCLE_DISABLED in node.lifecycle


def test_sdk_error_cancels_motion_and_invalidates_recovery():
    controller = FakeController()
    node = make_node(controller)
    node._arm_enabled = True
    node._recovery_complete = True

    TianjiArmControllerNode._on_sdk_error(
        node, "state loop", RuntimeError("link lost"))

    assert node._enable_cancel_event.is_set()
    assert node._recovery_cancel_event.is_set()
    assert node._arm_enabled is False
    assert node._recovery_complete is False
    assert controller.states == [0, 0]
    assert LIFECYCLE_SDK_ERROR in node.lifecycle


def test_direct_rigid_mode_service_is_rejected_without_position_command():
    controller = FakeController()
    node = make_node(controller)
    request = SimpleNamespace(data=True)
    response = SimpleNamespace(success=None, message='')

    result = TianjiArmControllerNode._set_arm_state_callback(
        node, request, response)

    assert result.success is False
    assert 'guarded recovery' in result.message
    assert not any(
        isinstance(call, tuple) and call[0] == 'set_position_mode'
        for call in controller.calls
    )


def test_rejected_target_enters_hold_without_disabling_hardware():
    lifecycle = []
    statuses = []
    node = SimpleNamespace(
        _target_hold_reason=None,
        _target_hold_log_at=0.0,
        _publish_lifecycle=lifecycle.append,
        _publish_teleop_status=statuses.append,
        get_logger=lambda: FakeLogger(),
    )

    TianjiArmControllerNode._enter_target_hold(
        node, 'external command stale for 0.3s')

    assert node._target_hold_reason == 'external command stale for 0.3s'
    assert lifecycle[-1] == LIFECYCLE_TARGET_HOLD
    assert statuses[-1].startswith('HOLD:')


def test_valid_target_resumes_automatically_from_hold():
    lifecycle = []
    statuses = []
    node = SimpleNamespace(
        _target_hold_reason='left target invalid',
        _target_hold_log_at=1.0,
        _publish_lifecycle=lifecycle.append,
        _publish_teleop_status=statuses.append,
        get_logger=lambda: FakeLogger(),
    )

    TianjiArmControllerNode._clear_target_hold(node)

    assert node._target_hold_reason is None
    assert lifecycle[-1] == 2
    assert statuses[-1].startswith('READY:')


def test_pose6_matrix_round_trip_preserves_tianji_orientation():
    pose = np.array([0.45, 0.2, 0.55, 15.0, -20.0, 35.0])

    matrix = TianjiArmControllerNode._pose6_to_matrix(pose)
    restored = TianjiArmControllerNode._matrix_to_pose(matrix)

    assert np.allclose(restored, pose)


def test_external_stream_timeout_requests_standby():
    now = __import__('time').monotonic()
    node = SimpleNamespace(
        _external_lock=threading.Lock(),
        _external_pose={'left': np.eye(4), 'right': np.eye(4)},
        _external_pose_at={'left': now - 2.0, 'right': now - 2.0},
        _external_zsp={'left': None, 'right': None},
        _external_stream_started=True,
        _external_hold_pose={'left': np.eye(4), 'right': np.eye(4)},
        _external_command_timeout_sec=0.25,
        _external_standby_timeout_sec=1.0,
        _control_sides=lambda: ('left', 'right'),
        get_logger=lambda: FakeLogger(),
        disabled=False,
    )
    node._do_disable = lambda: setattr(node, 'disabled', True)

    result = TianjiArmControllerNode._refresh_external_targets(node)

    assert result is False
    assert node.disabled is True


def test_external_joint_callback_converts_ros_radians_to_sdk_degrees():
    node = SimpleNamespace(
        _control_source='external',
        _external_command_mode='joint',
        _control_sides=lambda: ('right',),
        _external_lock=threading.Lock(),
        _external_joint={'left': None, 'right': None},
        _external_joint_at={'left': 0.0, 'right': 0.0},
        get_logger=lambda: FakeLogger(),
    )
    message = JointState()
    message.position = np.radians([0, 10, 20, 30, 40, 50, 60]).tolist()

    TianjiArmControllerNode._external_joint_callback(
        node, 'right', message
    )

    assert np.allclose(
        node._external_joint['right'], [0, 10, 20, 30, 40, 50, 60]
    )
    assert node._external_joint_at['right'] > 0.0


def test_external_joint_stream_uses_fresh_selected_side_only():
    now = time.monotonic()
    node = SimpleNamespace(
        _external_lock=threading.Lock(),
        _external_joint={'left': None, 'right': [1.0] * 7},
        _external_joint_at={'left': 0.0, 'right': now},
        _external_stream_started=False,
        _external_hold_joint={'left': None, 'right': [0.0] * 7},
        _external_command_timeout_sec=0.25,
        _external_standby_timeout_sec=1.0,
        _control_sides=lambda: ('right',),
        get_logger=lambda: FakeLogger(),
    )

    left, right = TianjiArmControllerNode._refresh_external_joint_targets(
        node
    )

    assert left is None
    assert right == [1.0] * 7
    assert node._external_stream_started is True


def make_neutral_mapper():
    left_robot = np.eye(4)
    left_robot[:3, 3] = [0.4, 0.2, 0.6]
    right_robot = np.eye(4)
    right_robot[:3, 3] = [0.4, -0.2, 0.6]
    return SimpleNamespace(
        _neutral_tracker_left=np.eye(4),
        _neutral_tracker_right=np.eye(4),
        _neutral_arm_left=np.eye(4),
        _neutral_arm_right=np.eye(4),
        _neutral_robot_left=left_robot,
        _neutral_robot_right=right_robot,
        _teleop_position_scale=1.0,
    )


def test_neutral_tracker_maps_exactly_to_robot_snapshot():
    node = make_neutral_mapper()

    left = TianjiArmControllerNode._map_tracker_target(
        node, 'left', np.eye(4))
    right = TianjiArmControllerNode._map_tracker_target(
        node, 'right', np.eye(4))

    assert np.allclose(left, node._neutral_robot_left)
    assert np.allclose(right, node._neutral_robot_right)


def test_tracker_translation_is_aligned_as_delta_around_robot_snapshot():
    node = make_neutral_mapper()
    current = np.eye(4)
    current[:3, 3] = [0.1, -0.2, 0.05]

    target = TianjiArmControllerNode._map_tracker_target(
        node, 'left', current)

    assert np.allclose(target[:3, 3], [0.5, 0.0, 0.65])


def test_tracker_rotation_uses_same_side_frame_math_for_both_arms():
    node = make_neutral_mapper()
    current = np.eye(4)
    current[:3, :3] = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    left = TianjiArmControllerNode._map_tracker_target(
        node, 'left', current)
    right = TianjiArmControllerNode._map_tracker_target(
        node, 'right', current)

    assert np.allclose(left[:3, :3], current[:3, :3])
    assert np.allclose(right[:3, :3], current[:3, :3])


def test_tracker_delta_is_not_rejected_by_a_software_workspace_guard():
    node = make_neutral_mapper()
    current = np.eye(4)
    current[0, 3] = 0.5

    target = TianjiArmControllerNode._map_tracker_target(
        node, 'right', current)

    assert np.allclose(target[:3, 3], [0.9, -0.2, 0.6])


def test_upper_arm_direction_has_stable_neutral():
    node = make_neutral_mapper()

    left = TianjiArmControllerNode._map_upper_arm_direction(
        node, 'left', np.eye(4))
    right = TianjiArmControllerNode._map_upper_arm_direction(
        node, 'right', np.eye(4))

    assert np.allclose(left, np.array([0.0, -1.0, -0.5]) / np.sqrt(1.25))
    assert np.allclose(right, np.array([0.0, 1.0, -0.5]) / np.sqrt(1.25))


def test_upper_arm_rotation_is_relative_to_enable_neutral():
    node = make_neutral_mapper()
    current = np.eye(4)
    current[:3, :3] = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    right = TianjiArmControllerNode._map_upper_arm_direction(
        node, 'right', current)
    expected = current[:3, :3] @ np.array([0.0, 1.0, -0.5])

    assert np.allclose(right, expected / np.linalg.norm(expected))


def test_state_publisher_does_not_touch_sdk_during_recovery():
    node = make_node(FakeController())
    calls = []
    node._publish_state = lambda: calls.append(True)
    node._recovery_in_progress.set()

    TianjiArmControllerNode._state_publish_loop(node)

    assert calls == []


def test_tracker_clutch_disconnects_without_requesting_standby_then_reanchors():
    controller = FakeController()
    node = make_node(controller)
    node._control_source = "tracker"
    node._arm_enabled = True
    node._tracker_connected = True
    statuses = []
    calls = []
    node._publish_teleop_status = statuses.append
    node._reset_control_targets = lambda: calls.append("reset")
    node._arm_handoff_snapshot = lambda: calls.append("snapshot")
    node._capture_teleop_neutral = lambda: calls.append("neutral")
    response = SimpleNamespace(success=None, message="")

    TianjiArmControllerNode._toggle_tracker_clutch_callback(
        node, object(), response
    )

    assert response.success is True
    assert node._tracker_connected is False
    assert node.lifecycle[-1] == LIFECYCLE_CLUTCH_DISCONNECTED
    assert not any(
        isinstance(call, tuple) and call[0] == "set_standby"
        for call in controller.calls
    )

    TianjiArmControllerNode._toggle_tracker_clutch_callback(
        node, object(), response
    )

    assert response.success is True
    assert node._tracker_connected is True
    assert calls == ["reset", "snapshot", "neutral"]
    assert node.lifecycle[-1] == 2
    assert statuses[-1].startswith("READY:")


def test_tracker_clutch_is_rejected_until_teleop_is_enabled():
    node = make_node(FakeController())
    node._control_source = "tracker"
    node._tracker_connected = True
    response = SimpleNamespace(success=None, message="")

    TianjiArmControllerNode._toggle_tracker_clutch_callback(
        node, object(), response
    )

    assert response.success is False
    assert "Enable teleoperation" in response.message


def test_control_loop_sends_no_tracker_command_while_clutch_is_disconnected():
    calls = []
    node = SimpleNamespace(
        _arm_enabled=True,
        _control_source="tracker",
        _tracker_connected=False,
        _teleop_control=lambda: calls.append("control"),
    )

    TianjiArmControllerNode._control_loop(node)

    assert calls == []


def test_status_snapshot_uses_only_cached_feedback_and_controller_state():
    class ExplodingController:
        def __getattr__(self, name):
            raise AssertionError(f"status snapshot touched SDK attribute {name}")

    node = SimpleNamespace(
        controller=ExplodingController(),
        _last_state_feedback_at=time.monotonic() - 0.02,
        _last_arm_state_codes=(3, 0),
        _last_arm_error_codes=(0, 0),
        _last_left_state_joints=[1.0] * 7,
        _last_right_state_joints=[-1.0] * 7,
        _last_detailed_arm_status=None,
        _last_detailed_arm_status_at=0.0,
        _last_lifecycle_state=2,
        _last_teleop_status="READY: target valid",
        _status_snapshot_sequence=4,
        _arm_enabled=True,
        _arm_state=ArmState.IMPEDANCE,
        _active_arm_mode="both",
        _control_source="tracker",
        _tracker_connected=True,
        _target_hold_reason=None,
        _motion_in_progress=lambda: False,
    )

    snapshot = TianjiArmControllerNode._build_status_snapshot(node)

    assert snapshot["sequence"] == 4
    assert snapshot["lifecycle_state"] == 2
    assert snapshot["arms"]["left"]["state"] == 3
    assert snapshot["arms"]["right"]["state"] == 0
    assert snapshot["feedback_age_s"] < 0.2
    assert snapshot["detailed_hardware_status"]["available"] is False


def test_full_arm_status_service_remains_hardware_backed_and_refreshes_cache():
    expected = {
        "left": {"state": 3, "err_code": 0},
        "right": {"state": 0, "err_code": 2},
    }
    calls = []
    node = SimpleNamespace(
        controller=SimpleNamespace(
            get_arm_status=lambda: calls.append("get_arm_status") or expected
        ),
        _performance_metrics=None,
        _motion_in_progress=lambda: False,
        _last_detailed_arm_status=None,
        _last_detailed_arm_status_at=0.0,
        _last_arm_state_codes=(None, None),
        _last_arm_error_codes=(None, None),
    )
    response = SimpleNamespace(success=None, message="")

    TianjiArmControllerNode._arm_status_cb(node, object(), response)

    assert response.success is True
    assert calls == ["get_arm_status"]
    assert node._last_detailed_arm_status == expected
    assert node._last_arm_state_codes == (3, 0)
    assert node._last_arm_error_codes == (0, 2)


def test_console_process_exit_happens_only_after_normal_cleanup(monkeypatch):
    events = []
    monkeypatch.setattr(
        tianji_arm_node,
        "main",
        lambda _argv=None: events.append("cleanup_complete"),
    )

    def hard_exit(code):
        events.append(("hard_exit", code))
        raise SystemExit(code)

    monkeypatch.setattr(tianji_arm_node.os, "_exit", hard_exit)

    with pytest.raises(SystemExit) as exited:
        tianji_arm_node.process_main([])

    assert exited.value.code == 0
    assert events == ["cleanup_complete", ("hard_exit", 0)]


def test_console_entrypoint_uses_deterministic_post_sdk_exit():
    setup_source = (
        Path(__file__).resolve().parents[1] / "setup.py"
    ).read_text()
    assert (
        "tianji_arm_controller = controller.tianji_arm_node:process_main"
        in setup_source
    )
