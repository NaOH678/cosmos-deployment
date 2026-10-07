import logging
import threading

import pytest

from tianji_output.tianji_chest_driver import TianjiChestDriver
from tianji_output._internal.fx_robot import Marvin_Robot


class FakeRobot:
    def __init__(self, ack=1, states=(0, 0), joints=None,
                 command_response=1.0):
        self.ack = ack
        self.command_response = command_response
        self.external_command_response = command_response
        self.calls = []
        joints = joints or ([1.0] * 7, [-1.0] * 7)
        self.snapshot = {
            'states': [
                {'cur_state': states[0], 'err_code': 0},
                {'cur_state': states[1], 'err_code': 0},
            ],
            'outputs': [
                {
                    'fb_joint_pos': list(joints[0]),
                    'fb_joint_posE': list(joints[0]),
                    'fb_joint_cmd': list(joints[0]),
                    'fb_joint_vel': [0.0] * 7,
                },
                {
                    'fb_joint_pos': list(joints[1]),
                    'fb_joint_posE': list(joints[1]),
                    'fb_joint_cmd': list(joints[1]),
                    'fb_joint_vel': [0.0] * 7,
                },
            ],
        }

    def subscribe(self, _dcss):
        self.calls.append(('subscribe', (), {}))
        return self.snapshot

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name == 'send_cmd_wait_response':
                return self.ack
            if name == 'set_state':
                index = 0 if kwargs['arm'] == 'A' else 1
                self.snapshot['states'][index]['cur_state'] = kwargs['state']
            if name == 'set_joint_cmd_pose':
                index = 0 if kwargs['arm'] == 'A' else 1
                if self.snapshot['states'][index]['cur_state'] == 1:
                    current = self.snapshot['outputs'][index]['fb_joint_pos']
                    external = self.snapshot['outputs'][index]['fb_joint_posE']
                    target = kwargs['joints']
                    self.snapshot['outputs'][index]['fb_joint_pos'] = [
                        actual + self.command_response * (command - actual)
                        for actual, command in zip(current, target)
                    ]
                    self.snapshot['outputs'][index]['fb_joint_posE'] = [
                        actual + self.external_command_response
                        * (command - actual)
                        for actual, command in zip(external, target)
                    ]
                    self.snapshot['outputs'][index]['fb_joint_cmd'] = list(target)
            return 1
        return record


class LegacySdkLibrary:
    def __init__(self, send_result=1):
        self.send_result = send_result

    def OnSetSend(self):
        return self.send_result


def make_driver(states=(3, 3), ack=1, joints=None):
    driver = object.__new__(TianjiChestDriver)
    driver.robot = FakeRobot(ack=ack, states=states, joints=joints)
    driver.logger = logging.getLogger('test_tianji_safety')
    driver._has_send_wait_response = True
    driver._last_impedance_check_at = 0.0
    driver._last_impedance_check_arms = ()
    driver._init_joints_left = [5.0] * 7
    driver._init_joints_right = [-5.0] * 7
    driver._single_arm_init_joints_right = [-6.0] * 7
    driver._park_joints_left = [
        90.0, -90.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]
    driver._joint_limits_left = [(-170.0, 170.0)] * 7
    driver._joint_limits_right = [(-170.0, 170.0)] * 7
    driver.get_current_joints = lambda: ([1.0] * 7, [-1.0] * 7)
    driver.get_arm_states_only = lambda: states
    driver.get_arm_status = lambda: {
        'left': {
            'state': states[0], 'err_code': 0,
            'servo_errors': ['0x0000'] * 7,
        },
        'right': {
            'state': states[1], 'err_code': 0,
            'servo_errors': ['0x0000'] * 7,
        },
    }
    driver._poll_arm_state_after_switch = lambda **kwargs: (True, '')
    return driver


def test_left_inactive_park_pose_matches_vertical_down_configuration():
    driver = make_driver()

    assert driver.get_park_joints('A') == [
        90.0, -90.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]
    with pytest.raises(ValueError, match='No inactive parking pose'):
        driver.get_park_joints('B')


def test_right_single_arm_mode_has_a_separate_recovery_reference():
    driver = make_driver()

    assert driver.get_init_joints('both') == ([5.0] * 7, [-5.0] * 7)
    assert driver.get_init_joints('left') == ([5.0] * 7, [-5.0] * 7)
    assert driver.get_init_joints('right') == ([5.0] * 7, [-6.0] * 7)


def call_names(driver):
    return [call[0] for call in driver.robot.calls]


def test_tool_parameters_use_identified_right_hand_payload(monkeypatch):
    driver = make_driver(states=(0, 0))
    monkeypatch.setattr(
        'tianji_output.tianji_chest_driver.time.sleep', lambda _seconds: None)

    TianjiChestDriver._set_tool_params(driver)

    tool_calls = [
        call for call in driver.robot.calls if call[0] == 'set_tool'
    ]
    assert [call[2] for call in tool_calls] == [
        {
            'arm': 'A',
            'kineParams': [0, 0, 0, 0, 0, 0],
            'dynamicParams': [0.95, 0, 0, 90, 0, 0, 0, 0, 0, 0],
        },
        {
            'arm': 'B',
            'kineParams': [0, 0, 0, 0, 0, 0],
            'dynamicParams': [
                1.044, 6.089, -21.289, 101.794,
                0.012, 0.0, 0.0, 0.008, 0.0, 0.002,
            ],
        },
    ]
    assert call_names(driver)[-1] == 'send_cmd'


def test_joint_feedback_read_caches_state_and_error_codes_from_same_sdk_frame():
    driver = object.__new__(TianjiChestDriver)
    driver.robot = FakeRobot(states=(3, 0))
    driver.robot.snapshot["states"][0]["err_code"] = 2
    driver.robot.snapshot["states"][1]["err_code"] = 0

    left, right = TianjiChestDriver.get_current_joints(driver)

    assert left == [1.0] * 7
    assert right == [-1.0] * 7
    assert driver.latest_arm_state_codes == (3, 0)
    assert driver.latest_arm_error_codes == (2, 0)
    assert call_names(driver) == ["subscribe"]


def test_direct_rigid_position_mode_is_hard_disabled():
    driver = make_driver()

    with pytest.raises(RuntimeError, match='state=1'):
        driver.set_position_mode(velRatio=1, AccRatio=1)

    assert driver.robot.calls == []


def test_legacy_sdk_falls_back_when_wait_response_symbol_is_missing():
    robot = object.__new__(Marvin_Robot)
    robot.robot = LegacySdkLibrary(send_result=1)

    assert robot.supports_send_cmd_wait_response() is False
    assert robot.send_cmd_wait_response(100) == 1


def test_guarded_position_mode_is_rate_capped_and_seeds_before_state_one():
    driver = make_driver(states=(0, 0))

    driver.set_position_mode(
        velRatio=10,
        AccRatio=10,
        arms=['A'],
        seed_joints={'A': [1.0] * 7},
        recovery=True,
    )

    state_calls = [call for call in driver.robot.calls if call[0] == 'set_state']
    pose_calls = [call for call in driver.robot.calls if call[0] == 'set_joint_cmd_pose']
    assert [call[2] for call in state_calls] == [{'arm': 'A', 'state': 1}]
    assert pose_calls[0][2] == {'arm': 'A', 'joints': [1.0] * 7}
    assert driver.robot.calls.index(pose_calls[0]) < driver.robot.calls.index(state_calls[0])


def test_explicit_replay_position_mode_seeds_before_state_one():
    driver = make_driver(states=(0, 0))

    driver.set_position_mode(
        velRatio=10,
        AccRatio=10,
        arms=['B'],
        replay=True,
    )

    state_calls = [call for call in driver.robot.calls if call[0] == 'set_state']
    pose_calls = [
        call for call in driver.robot.calls
        if call[0] == 'set_joint_cmd_pose'
    ]
    assert [call[2] for call in state_calls] == [{'arm': 'B', 'state': 1}]
    assert pose_calls[0][2] == {'arm': 'B', 'joints': [-1.0] * 7}
    assert driver.robot.calls.index(pose_calls[0]) < driver.robot.calls.index(
        state_calls[0])


def test_guarded_position_mode_rejects_ratio_above_ten():
    driver = make_driver(states=(0, 0))

    with pytest.raises(ValueError, match='velRatio'):
        driver.set_position_mode(
            velRatio=11, AccRatio=10, arms=['A'], recovery=True)


def test_impedance_enable_preloads_params_before_state_three(monkeypatch):
    driver = make_driver(states=(3, 3))
    monkeypatch.setattr(
        'tianji_output.tianji_chest_driver.time.sleep', lambda _seconds: None)

    driver.set_impedance_mode(mode='joint', velRatio=15, AccRatio=15)

    state_calls = [call for call in driver.robot.calls if call[0] == 'set_state']
    assert [call[2]['state'] for call in state_calls] == [3, 3]
    kd_calls = [call for call in driver.robot.calls if call[0] == 'set_joint_kd_params']
    assert driver.robot.calls.index(kd_calls[-1]) < driver.robot.calls.index(state_calls[0])
    pose_calls = [call for call in driver.robot.calls if call[0] == 'set_joint_cmd_pose']
    assert [call[2] for call in pose_calls] == [
        {'arm': 'A', 'joints': [1.0] * 7},
        {'arm': 'B', 'joints': [-1.0] * 7},
    ]
    assert driver.robot.calls.index(state_calls[-1]) < driver.robot.calls.index(
        pose_calls[0])
    assert call_names(driver).count('send_cmd_wait_response') == 3
    assert 'send_cmd' not in call_names(driver)


def test_impedance_enable_rejects_missing_ack(monkeypatch):
    driver = make_driver(states=(0, 0), ack=0)
    monkeypatch.setattr(
        'tianji_output.tianji_chest_driver.time.sleep', lambda _seconds: None)

    with pytest.raises(RuntimeError, match='command dispatch failed'):
        driver.set_impedance_mode(mode='joint')


def test_set_active_skips_clear_error_when_standby_feedback_is_clean(monkeypatch):
    driver = make_driver(states=(0, 0))
    driver.get_arm_states_only = lambda: tuple(
        item['cur_state'] for item in driver.robot.snapshot['states'])
    monkeypatch.setattr(
        'tianji_output.tianji_chest_driver.time.sleep', lambda _seconds: None)

    driver.set_active(mode='joint')

    assert 'clear_error' not in call_names(driver)
    state_calls = [call for call in driver.robot.calls if call[0] == 'set_state']
    assert [call[2]['state'] for call in state_calls] == [3, 3]


@pytest.mark.parametrize(
    ('arms', 'expected_arm'),
    [(['A'], 'A'), (['B'], 'B')],
)
def test_set_active_can_enable_only_selected_arm(monkeypatch, arms, expected_arm):
    driver = make_driver(states=(0, 0))
    driver.get_arm_states_only = lambda: tuple(
        item['cur_state'] for item in driver.robot.snapshot['states'])
    driver.get_arm_status = lambda: {
        side: {
            'state': item['cur_state'],
            'err_code': 0,
            'servo_errors': ['0x0000'] * 7,
        }
        for side, item in zip(
            ('left', 'right'), driver.robot.snapshot['states'])
    }
    monkeypatch.setattr(
        'tianji_output.tianji_chest_driver.time.sleep', lambda _seconds: None)

    driver.set_active(mode='joint', arms=arms)

    state_calls = [call for call in driver.robot.calls if call[0] == 'set_state']
    assert [call[2] for call in state_calls] == [
        {'arm': expected_arm, 'state': 3}
    ]
    inactive_index = 1 if expected_arm == 'A' else 0
    assert driver.robot.snapshot['states'][inactive_index]['cur_state'] == 0


@pytest.mark.parametrize(
    ('states', 'left_target', 'right_target', 'expected_arm'),
    [((3, 0), [1.0] * 7, None, 'A'), ((0, 3), None, [-1.0] * 7, 'B')],
)
def test_direct_joint_command_requires_only_the_commanded_arm_in_impedance(
    states, left_target, right_target, expected_arm
):
    driver = make_driver(states=states)

    driver.move_to_joints_direct(
        left_joints=left_target,
        right_joints=right_target,
        active_arms=[expected_arm],
        standby_arms=['B' if expected_arm == 'A' else 'A'],
    )

    pose_calls = [
        call for call in driver.robot.calls if call[0] == 'set_joint_cmd_pose'
    ]
    assert len(pose_calls) == 1
    assert pose_calls[0][2]['arm'] == expected_arm
    assert set(driver._last_joint_command_timing) == {
        'total_ms',
        'state_guard_ms',
        'prepare_ms',
        'sdk_send_ms',
    }
    assert all(
        value >= 0.0
        for value in driver._last_joint_command_timing.values()
    )


def test_pose_solver_exposes_passive_timing_without_sending_a_command():
    driver = make_driver()

    result = driver.move_to_pose_direct(send=False)

    assert result == (False, False, None, None)
    assert set(driver._last_pose_timing) == {
        'total_ms',
        'sdk_reference_read_ms',
        'ik_compute_ms',
    }
    assert all(value >= 0.0 for value in driver._last_pose_timing.values())
    assert 'send_cmd' not in call_names(driver)


def test_direct_joint_command_rejects_a_command_to_standby_arm():
    driver = make_driver(states=(3, 0))

    with pytest.raises(RuntimeError, match='right state=0'):
        driver.move_to_joints_direct(right_joints=[-1.0] * 7)

    assert 'set_joint_cmd_pose' not in call_names(driver)


def test_direct_joint_command_default_preserves_dual_arm_requirement():
    driver = make_driver(states=(3, 0))

    with pytest.raises(RuntimeError, match='right state=0'):
        driver.move_to_joints_direct(left_joints=[1.0] * 7)

    assert 'set_joint_cmd_pose' not in call_names(driver)


def test_single_arm_command_rejects_inactive_arm_out_of_standby():
    driver = make_driver(states=(3, 3))

    with pytest.raises(RuntimeError, match='inactive arm is not in standby'):
        driver.move_to_joints_direct(
            left_joints=[1.0] * 7,
            active_arms=['A'],
            standby_arms=['B'],
        )

    assert 'set_joint_cmd_pose' not in call_names(driver)


def test_direct_joint_replay_command_accepts_position_mode():
    driver = make_driver(states=(0, 1))

    driver.move_to_joints_direct(
        right_joints=[-2.0] * 7,
        active_arms=['B'],
        standby_arms=['A'],
        command_state=1,
    )

    pose_calls = [
        call for call in driver.robot.calls
        if call[0] == 'set_joint_cmd_pose'
    ]
    assert pose_calls[-1][2] == {'arm': 'B', 'joints': [-2.0] * 7}


def test_direct_joint_replay_command_rejects_impedance_state():
    driver = make_driver(states=(0, 3))

    with pytest.raises(RuntimeError, match='position mode is not active'):
        driver.move_to_joints_direct(
            right_joints=[-2.0] * 7,
            active_arms=['B'],
            standby_arms=['A'],
            command_state=1,
        )


def test_set_active_rejects_nonzero_error_without_clearing(monkeypatch):
    driver = make_driver(states=(0, 0))
    driver.get_arm_status = lambda: {
        'left': {
            'state': 0, 'err_code': 13,
            'servo_errors': ['0x0000'] * 7,
        },
        'right': {
            'state': 0, 'err_code': 0,
            'servo_errors': ['0x0000'] * 7,
        },
    }
    monkeypatch.setattr(
        'tianji_output.tianji_chest_driver.time.sleep', lambda _seconds: None)

    with pytest.raises(RuntimeError, match='clean standby feedback'):
        driver.set_active(mode='joint')

    assert 'clear_error' not in call_names(driver)


def test_smooth_motion_requires_impedance_mode():
    driver = make_driver(states=(1, 1))

    with pytest.raises(RuntimeError, match='impedance mode is not active'):
        driver.move_to_joints_smooth(duration=0.04, dt=0.02)

    assert 'set_joint_cmd_pose' not in call_names(driver)


def test_standby_requires_state_confirmation():
    driver = make_driver(states=(3, 3))
    driver._poll_arm_state_after_switch = lambda **kwargs: (
        False, 'state-switch timed out')

    with pytest.raises(RuntimeError, match='state-switch timed out'):
        driver.set_standby()

    state_calls = [call for call in driver.robot.calls if call[0] == 'set_state']
    assert [call[2]['state'] for call in state_calls] == [0, 0]
    assert 'send_cmd_wait_response' in call_names(driver)


def test_state_switch_fails_immediately_on_sdk_error_state():
    driver = make_driver(states=(100, 0))
    driver._poll_arm_state_after_switch = (
        TianjiChestDriver._poll_arm_state_after_switch.__get__(driver))
    driver.robot.snapshot['states'][0]['err_code'] = 6

    ok, diagnostic = driver._poll_arm_state_after_switch(
        target_state=3, timeout=1.0, arms=['A'])

    assert ok is False
    assert 'state-switch fault' in diagnostic
    assert 'err_code=6: request-enter-torque failed' in diagnostic
    assert call_names(driver).count('subscribe') == 1


def test_state_switch_does_not_accept_target_state_with_nonzero_error():
    driver = make_driver(states=(3, 0))
    driver._poll_arm_state_after_switch = (
        TianjiChestDriver._poll_arm_state_after_switch.__get__(driver))
    driver.robot.snapshot['states'][0]['err_code'] = 7

    ok, diagnostic = driver._poll_arm_state_after_switch(
        target_state=3, timeout=1.0, arms=['A'])

    assert ok is False
    assert 'err_code=7: enter-torque failed' in diagnostic


def test_smooth_motion_honors_stop_before_next_command():
    driver = make_driver(states=(3, 3))
    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(RuntimeError, match='cancelled by stop request'):
        driver.move_to_joints_smooth(
            duration=0.04, dt=0.02, cancel_event=cancel_event)

    assert 'set_joint_cmd_pose' not in call_names(driver)


def test_recovery_honors_preexisting_stop_before_state_one():
    driver = make_driver(states=(0, 0))
    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(RuntimeError, match='cancelled by stop request'):
        driver.recover_arm_to_init('A', cancel_event=cancel_event)

    assert 'set_state' not in call_names(driver)


def test_recovery_refuses_legacy_sdk_without_synchronous_ack():
    driver = make_driver(states=(0, 0))
    driver._has_send_wait_response = False

    with pytest.raises(RuntimeError, match='OnSetSendWaitResponse'):
        driver.recover_arm_to_init('A')

    assert driver.robot.calls == []


def test_recovery_at_init_still_enters_guarded_state_one(monkeypatch):
    driver = make_driver(
        states=(0, 0), joints=([5.0] * 7, [-5.0] * 7))
    monkeypatch.setattr(
        'tianji_output.tianji_chest_driver.time.sleep', lambda _seconds: None)

    final_error = driver.recover_arm_to_init('A', hold_sec=0.0)

    state_calls = [
        call for call in driver.robot.calls if call[0] == 'set_state'
    ]
    assert final_error == pytest.approx(0.0)
    assert [call[2] for call in state_calls] == [{'arm': 'A', 'state': 1}]


def test_recovery_rejects_initial_encoder_disagreement():
    driver = make_driver(
        states=(0, 0), joints=([1.0] * 7, [-1.0] * 7))
    driver.robot.snapshot['outputs'][0]['fb_joint_posE'][3] = 3.0

    with pytest.raises(RuntimeError, match='encoder disagreement'):
        driver.recover_arm_to_init('A', encoder_agreement_deg=1.0)

    assert 'set_state' not in call_names(driver)


def test_recovery_rejects_external_encoder_that_does_not_follow(monkeypatch):
    driver = make_driver(
        states=(0, 0), joints=([1.0] * 7, [-1.0] * 7))
    driver._init_joints_left = [1.5] * 7
    driver.robot.external_command_response = 0.0
    monkeypatch.setattr(
        'tianji_output.tianji_chest_driver.time.sleep', lambda _seconds: None)

    with pytest.raises(RuntimeError, match='encoder disagreement'):
        driver.recover_arm_to_init(
            'A',
            max_speed_deg_s=1.0,
            max_accel_deg_s2=100.0,
            dt=0.1,
            arrival_tolerance_deg=0.01,
            encoder_agreement_deg=0.05,
            hold_sec=0.0,
        )


def test_recovery_commands_selected_arm_monotonically_toward_init(monkeypatch):
    driver = make_driver(
        states=(0, 0), joints=([1.0] * 7, [-1.0] * 7))
    driver._init_joints_left = [1.3] * 7
    monkeypatch.setattr(
        'tianji_output.tianji_chest_driver.time.sleep', lambda _seconds: None)

    final_error = driver.recover_arm_to_init(
        'A',
        max_speed_deg_s=1.0,
        max_accel_deg_s2=100.0,
        dt=0.1,
        arrival_tolerance_deg=0.01,
        hold_sec=0.0,
    )

    commands = [
        call[2]['joints'][0]
        for call in driver.robot.calls
        if call[0] == 'set_joint_cmd_pose' and call[2]['arm'] == 'A'
    ]
    moving_commands = [value for value in commands if value > 1.0]
    assert final_error <= 0.01
    assert moving_commands == sorted(moving_commands)
    assert moving_commands[-1] == pytest.approx(1.3)
    assert max(
        current - previous
        for previous, current in zip(moving_commands, moving_commands[1:])
    ) <= 0.1 + 1e-9
    assert not any(
        call[0] == 'set_joint_cmd_pose' and call[2]['arm'] == 'B'
        for call in driver.robot.calls
    )


def test_recovery_accepts_explicit_inactive_arm_park_target(monkeypatch):
    driver = make_driver(
        states=(0, 0), joints=([1.0] * 7, [-1.0] * 7))
    monkeypatch.setattr(
        'tianji_output.tianji_chest_driver.time.sleep', lambda _seconds: None)

    final_error = driver.recover_arm_to_init(
        'A',
        target_joints=[1.2] * 7,
        target_label='vertical park',
        max_speed_deg_s=1.0,
        max_accel_deg_s2=100.0,
        dt=0.1,
        arrival_tolerance_deg=0.01,
        hold_sec=0.0,
    )

    assert final_error <= 0.01
    assert driver.robot.snapshot['outputs'][0]['fb_joint_pos'] == pytest.approx(
        [1.2] * 7
    )


def test_recovery_keeps_bounded_lookahead_with_slow_feedback(monkeypatch):
    driver = make_driver(
        states=(0, 0), joints=([1.0] * 7, [-1.0] * 7))
    driver.robot.command_response = 0.25
    driver._init_joints_left = [1.5] * 7
    monkeypatch.setattr(
        'tianji_output.tianji_chest_driver.time.sleep', lambda _seconds: None)

    final_error = driver.recover_arm_to_init(
        'A',
        max_speed_deg_s=1.0,
        max_accel_deg_s2=100.0,
        dt=0.1,
        tracking_error_deg=2.0,
        command_lead_deg=0.5,
        arrival_tolerance_deg=0.01,
        hold_sec=0.0,
    )

    moving_commands = [
        call[2]['joints'][0]
        for call in driver.robot.calls
        if call[0] == 'set_joint_cmd_pose'
        and call[2]['arm'] == 'A'
        and call[2]['joints'][0] > 1.0
    ]
    assert final_error <= 0.01
    assert moving_commands[:3] == pytest.approx([1.1, 1.2, 1.3])
    assert max(value - 1.0 for value in moving_commands) <= 0.5 + 1e-9


def test_recovery_aborts_when_feedback_moves_opposite_direction(monkeypatch):
    driver = make_driver(
        states=(0, 0), joints=([1.0] * 7, [-1.0] * 7))
    driver.robot.command_response = -1.0
    driver._init_joints_left = [5.0] * 7
    monkeypatch.setattr(
        'tianji_output.tianji_chest_driver.time.sleep', lambda _seconds: None)

    with pytest.raises(RuntimeError, match='opposite direction'):
        driver.recover_arm_to_init(
            'A',
            max_speed_deg_s=1.0,
            max_accel_deg_s2=100.0,
            dt=0.1,
            reverse_motion_deg=0.15,
            hold_sec=0.0,
        )


def test_legacy_impedance_move_to_init_is_disabled():
    driver = make_driver(states=(3, 3))

    with pytest.raises(RuntimeError, match='guarded recover_arm_to_init'):
        driver.move_to_init()
