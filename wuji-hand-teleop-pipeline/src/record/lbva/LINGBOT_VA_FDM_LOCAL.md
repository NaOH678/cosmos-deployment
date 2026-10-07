# LingBot-VA FDM robot-side mode

This is the robot-side implementation contract. The handoff directory is a
read-only source snapshot; none of its cloud servers is assumed to implement
this protocol.

## Mode isolation

- `config/pi05_protocol_v2.yaml` selects `deployment.protocol_mode: pi_v2` and
  keeps Pi's own horizon, observation request, time-alignment, and boundary
  behavior. FDM tests deliberately do not pin those independently tuned Pi
  parameters.
- `config/lingbot_va_fdm.yaml` selects `fdm_async`, 48/48 execution, no
  observation-time skip, no boundary blend, and an explicit `action_mode`.
- Both profiles configure checkpoint identity once at
  `deployment.policy_http_expected_model_id`. The FDM cloud handoff profile
  repeats the same value at `deployment.fdm_async.model_id`; startup rejects
  the configuration if the two values differ.
- To switch the LingBot-VA checkpoint, edit only those two values in
  `config/lingbot_va_fdm.yaml` and keep them identical. No script, Python,
  launch-file, or test change is required.
- Protocol and timeline logic lives in `wuji_data_pipeline/fdm_async.py`.
  Hardware publishers, limits, lifecycle, E-stop/Standby behavior, authenticated
  transport, logging, and replay remain in their existing modules.

Use the dedicated supervised entrypoint. It fixes the active hardware profile
to right arm + right WujiHand and always starts both required cameras:

```bash
export LINGBOT_VA_API_KEY='<provided out of band>'
./src/scripts/run_lingbot_va_fdm.sh \
  --server https://SERVICE_BASE_URL
```

Before the first deployment, the container/package preflight can be run
without a server address or API key:

```bash
./src/scripts/run_lingbot_va_fdm.sh --check
```

There is intentionally no `--no-camera` mode: FDM execution feedback requires
a complete post-action `head` + `right_wrist` image snapshot every four model
actions.

The `~/ready` service stays false until `hello` capability validation and
`bootstrap` have produced and validated W0. Only then should the operator
Enable the hardware.

## Implemented timeline

The action connection performs:

```text
hello -> hello_ack -> bootstrap -> W0
                              robot Enable
execute active Wn <-------> request/prefetch Wn+1
```

`wire_chunk_id`, `global_action_start`, and reported `native_spans` are checked
independently. The native timeline is N0=48 executable actions and N1+=64;
the continuous stream is cut into fixed 48-step wire chunks. For example W2
contains N1[48:64] followed by N2[0:32]. A wire response never advances the
executed frontier.

At each successfully published 30 Hz waypoint, the local executed frontier
advances once. Every four waypoints, the control callback only enqueues their
final locally processed 54D mappings. A separate feedback worker waits for a
complete `head` and `right_wrist` snapshot timestamped after the fourth
waypoint, JPEG-encodes it, sends `execution_feedback`, and retries the exact
same immutable request until `feedback_ack` or the configured failure limit.
The action and feedback workers own different persistent transports.

`deployment.fdm_async.state_history.enabled` controls an optional,
backward-compatible measured-state extension. When `true`, the same 7.5 Hz
feedback request additionally contains `qpos_history` with shape `(4, 54)` and
the matching four `qpos_timestamps`. Each sample is the latest measured robot
state at its corresponding 30 Hz action tick, ordered as left arm 7, left hand
20, right arm 7, right hand 20, all in radians. An inactive hand retains the
existing 20D zero-filled protocol slot. The current per-side state snapshot
after the fourth action and the post-action camera keyframe remain unchanged.
When `false` (also the default if the mapping is absent), qpos is not sampled
on each action and neither history field is added, preserving the original
feedback payload for older models.

The current dropper checkpoint uses `action_mode: joint`, for which state
history is mandatory. Its wire action is 54D in
`left arm 7 + left hand 20 + right arm 7 + right hand 20` order, with both arm
and hand values in radians. Each arm mapping contains `joint_pos`; no EEF
position, quaternion, ZSP, or degree-valued hand action is accepted in this
mode. `pchip_joint` interpolates only joint-space arm/hand targets; the EEF
SLERP path is not entered. The existing EEF and replay formats remain
unchanged. The deployment session reads this profile value and starts both the
Tianji controller and the policy client in joint mode. The controller performs
the only radians-to-SDK degrees conversion at its hardware boundary.

At a wire boundary with no pending chunk, `pending_miss_policy: hold_last`
keeps publishing the last validated target on the 120 Hz controller timer.
These publications only keep the controller watchdog alive: they do not pop a
model waypoint, increment the per-wire/global action indices, or create an
`execution_feedback` entry. When Wn arrives, execution resumes at Wn[0]. The
default `pending_miss_timeout_s: 5.0` is a final safety ceiling, not a normal
GPU-latency budget. Once it expires, the session is invalidated, old queues are
cleared, Standby is requested, and a new hello/bootstrap is required. Fatal
authentication, session, protocol, or feedback failures retain their immediate
reset/Standby behavior rather than waiting for this ceiling.

Delivery-time safety validation preserves lookahead semantics: a prefetched
Wn is checked from Wn-1[-1] to Wn[0], then through every step inside Wn. It is
not compared with the robot's current target near the beginning of Wn-1,
because that would incorrectly treat almost 48 valid future steps as one jump.
At activation, the splice is checked again against the target that actually
reached the preceding wire boundary.

## Protocol-review items

The following values are explicit in `deployment.fdm_async`; changing them
does not require editing controller code:

1. Formal protocol version (local baseline `3`).
2. Feedback HTTP path (local baseline is a second connection to
   `/v1/robot-policy`).
3. Actual action definition (`final_30hz_waypoint`).
4. First-phase boundary blend (disabled).
5. Keyframe choice (`first_complete_snapshot_after_stride`).
6. Feedback queue capacity, retry count/backoff, and keyframe timeout.
7. Pending-miss policy/final safety timeout and action retry backoff.
8. Accepted-range and required `feedback_ack` field names.
9. Single active robot session (required in the initial profile).

The cloud team still needs to confirm these field names and values. The
non-negotiable invariants are continuous global action indexing, actual-action
feedback every four steps, post-action real images, native selective grounding,
48-wire re-chunking without truncating 64-native output, and old-session
isolation.
