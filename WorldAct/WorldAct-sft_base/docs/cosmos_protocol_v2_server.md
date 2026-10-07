# Cosmos protocol-v2 robot policy service

This service adapts the 27D single-right-hand Cosmos WAM checkpoint to the
Tianji/Wuji protocol-v2 HTTP contract. It is separate from the existing JSON
and WebSocket action-policy servers because the robot contract uses an
authenticated pickle mapping over HTTP/1.1.

## Deployment manifest

Start from
[`examples/deployment/cosmos_singlerighthand_protocol_v2.yaml`](../examples/deployment/cosmos_singlerighthand_protocol_v2.yaml).
Change `deployment.model_id` and `model.checkpoint_path` together whenever the
artifact changes. The checked-in `robot_layout` is the complete mapping emitted
by the current Wuji `RobotLayout().metadata()` implementation. If the robot
layout changes, update both profiles together; the server rejects any mismatch.

The checked-in profile records the following contract:

| Item | Value |
| --- | --- |
| Mode | right side only |
| Required cameras, wire order | `head`, `right_wrist` |
| Model view | right-wrist image above head image |
| Image conversion | JPEG decode as BGR, explicit BGR to RGB, aspect-preserving shared-width composition, reflection padding to the closest 480-tier size |
| Model normalization | model-native `uint8 -> [-1, 1]`; no duplicate server normalization |
| State | right EEF `xyz + xyzw` in `right_chest`, then 20 hand joints in radians: 27D |
| State/history | one current state row and one current composed image; no cross-request image history |
| Native action | 32 absolute 27D targets at 15 Hz; `right_chest`; hand radians |
| Wire action | 32 complete bilateral targets at 15 Hz; hands in degrees |
| Inactive side | current measured left EEF and hand held for the entire chunk |
| Transform | right EEF identity (`right_chest -> right_chest`), quaternion normalized as `xyzw` |
| Action normalizer | none in this training recipe; output externalization only removes model padding |
| Checkpoint weights | training EMA (`net_ema.*`) loaded into the inference model |
| Sampling | seed 0, UniPC, guidance 3.0, four steps, shift 5.0 |
| Startup handoff | disabled by default; the robot profile must use the same value |

The manifest intentionally leaves the final checkpoint hash, runtime hardware,
and measured latency dependent on the selected artifact. These must be filled
and measured before full-task robot rollout; do not treat the debug `model_id`
as a production artifact identifier.

When `checkpoint_sha256` is set, startup verifies it before loading weights. A
checkpoint file uses its normal SHA-256. A checkpoint directory uses SHA-256
over each file's UTF-8 relative path, a NUL separator, and file contents, in
sorted relative-path order. Leaving the field `null` skips this potentially
expensive scan and is appropriate only during artifact debugging.

## Start and progress through integration modes

The pickle endpoint is safe only on an authenticated, trusted network. Set the
key through the environment; never write its value to the YAML or command
line.

For the single-right-hand Edge deployment on the current cluster, use the
convenience launcher. It accepts the platform's `H_API_KEY` or
`COSMOS_POLICY_API_KEY`, and derives both the DCP directory and `model_id` from
one checkpoint-step setting:

```bash
COSMOS_CKPT_STEP=50000 bash script/start_cosmos_policy_server.sh
```

The default run name is `singlerighthand-edge-droid-50k-retrain-v1` and the
default step is 50000, so the normal platform command can simply execute the
script. `COSMOS_RUN_NAME` switches the run directory and derived model ID from
one setting. `COSMOS_MODEL_ID`, `CUDA_VISIBLE_DEVICES`,
`COSMOS_SERVICE_MODE`, `COSMOS_SERVICE_HOST`, and `SERVICE_PORT` remain
optional overrides. The launcher defaults to `full`; set
`COSMOS_SERVICE_MODE=small_motion` explicitly for bounded robot diagnostics.
Sampling can be changed in one place with
`COSMOS_GUIDANCE`, `COSMOS_NUM_STEPS`, and `COSMOS_SHIFT` (defaults: 3, 4, 5).
`COSMOS_TRAJECTORY_SMOOTHING` selects `binomial5` (default) or `none`. The
five-tap symmetric filter removes alternating 15 Hz waypoint noise while
preserving the observed starting state and the model's final EEF/hand target.

```bash
cd /path/to/WorldAct-cosmos3-edge-droid-sft
source .venv/bin/activate
export LD_LIBRARY_PATH=''
export COSMOS_POLICY_API_KEY='<secret>'

python -m cosmos_framework.scripts.action_policy_server_protocol_v2 \
  --config examples/deployment/cosmos_singlerighthand_protocol_v2.yaml
```

The example starts in `hold` mode and does not load a checkpoint. Proceed to a
bounded action check only after hold integration succeeds:

```bash
python -m cosmos_framework.scripts.action_policy_server_protocol_v2 \
  --config examples/deployment/cosmos_singlerighthand_protocol_v2.yaml \
  --service-mode small_motion \
  --model-id '<artifact-id>-small-motion' \
  --checkpoint-path /path/to/checkpoint
```

Use `--service-mode full` only after small-motion validation. The command also
accepts `--model-id`, `--checkpoint-path`, `--model-config-file`, `--host`,
`--port`, `--guidance`, `--num-steps`, `--shift`, `--trajectory-smoothing`, and
`--no-warmup`; CLI overrides are validated again as a complete manifest. For a
training-time DCP, pass the matching frozen `config.yaml` from the run directory
with `--model-config-file`.

The current model loader is single-process/single-GPU. The server initializes a
one-rank process group because Cosmos model loading uses FSDP/DTensor utilities.
It rejects `WORLD_SIZE > 1` rather than starting multiple HTTP listeners with
an invalid collective request flow.

### Dropper Edge 4B checkpoint

The dropper deployment uses a dedicated joint-action manifest and adapter. Its
27D model state is seven right-arm joints followed by 20 right-hand joints, all
in radians; its 27D output has the same absolute-joint layout. The wire response
uses `arm_joint_action_{left,right}` in radians and `hand_action_{left,right}` in
degrees. The left side holds its measured joints for the entire chunk. EEF
fields remain observation diagnostics and are never used to build this model's
state or interpret its output.

The inference prompt exactly matches the training cache:
`draw liquid from the beaker with a dropper and dispense it into the test tube`.
The manifest also requires joint `arm_command_mode`/`action_space`, the exact
joint `robot_layout`, and enables `startup_handoff`. Its launcher defaults to
`iter_000030000` from the
`singlerighthand-dropper-edge-droid-50k-aot` run:

```bash
bash script/start_cosmos_dropper_policy_server.sh
```

The normal platform command is:

```bash
/bin/bash -c 'cd /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/WorldAct-cosmos3-edge-droid-sft && exec /bin/bash script/start_cosmos_dropper_policy_server.sh'
```

Change only `COSMOS_CKPT_STEP` when testing another checkpoint. The shared Edge
launcher zero-pads the step and derives the checkpoint directory and model ID:

```bash
COSMOS_CKPT_STEP=32500 bash script/start_cosmos_dropper_policy_server.sh
```

If the run directory name changes, set `COSMOS_RUN_NAME` once. The dedicated
manifest is
`examples/deployment/cosmos_singlerighthand_dropper_edge_protocol_v2.yaml`.

### Nano 16B checkpoint

Nano 16B uses the same protocol-v2 Python service with the original EEF adapter,
plus a separate deployment manifest and launcher. The launcher defaults to the
single-right-hand Nano run's `iter_000027500` checkpoint:

```bash
bash script/start_cosmos_nano_policy_server.sh
```

The normal platform command is:

```bash
/bin/bash -c 'cd /mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/WorldAct-cosmos3-edge-droid-sft && exec /bin/bash script/start_cosmos_nano_policy_server.sh'
```

Change only `COSMOS_CKPT_STEP` to select another checkpoint. The launcher
zero-pads it and derives the checkpoint directory and model ID together:

```bash
COSMOS_CKPT_STEP=25000 bash script/start_cosmos_nano_policy_server.sh
```

Nano's frozen training config remains under
`runs/cosmos/singlerighthand-nano-policy-droid/cosmos3_action/action_sft/action_policy_singlerighthand_nano/config.yaml`.
Do not point the Nano launcher at an Edge config or checkpoint: their model
architectures are incompatible. This protocol server currently loads Nano on
one visible GPU, so the selected GPU must have enough memory for the 16B BF16
inference model and runtime activations.

## Health and operation

```bash
curl -fsS http://127.0.0.1:8000/healthz
curl -fsS http://127.0.0.1:8000/readyz
```

`/healthz` reports process liveness. `/readyz` returns 200 only after model
load and optional warmup finish. The formal endpoint is exactly
`POST /v1/robot-policy`; it uses HTTP/1.1 persistent connections,
`application/octet-stream`, and `pickle.HIGHEST_PROTOCOL`.

Every hello starts a new generation for that session and HTTP connection. A
reconnect, duplicate hello, session eviction, or TTL expiry makes an older
in-flight result stale. The stale action is discarded and cannot be returned
as an `action_chunk`. Session storage is bounded by `max_sessions` and
`session_ttl_s`; GPU work is bounded by `max_inflight_inferences`.

Python logging deliberately omits request bodies, image bytes, and auth
headers. The model output directory contains the standard Cosmos console/debug
logs configured during model load.

## Stable errors

Model and protocol failures use HTTP 200 with the protocol pickle error
envelope. Authentication failures use HTTP 401 with an empty body.

| Code | Meaning | Retry/session behavior |
| --- | --- | --- |
| `COSMOS_BAD_PICKLE` | body is not a pickle mapping | fix request; connection may remain open |
| `COSMOS_PROFILE_MISMATCH` | robot layout/cameras/side profile differs | fix profile |
| `COSMOS_SESSION_REQUIRED` | no current hello on this connection | fatal; hello again |
| `COSMOS_BUSY` | configured inference slots are occupied | retryable |
| `COSMOS_INFERENCE_TIMEOUT` | inference exceeded `inference_timeout_s`; late output is discarded | retryable; may be busy until old GPU work exits |
| `COSMOS_STALE_SESSION` | hello/reconnect/eviction replaced an in-flight generation | fatal for old connection |
| `COSMOS_GPU_OOM` | CUDA allocation failed | fatal; restart/recover worker |
| `COSMOS_UNSAFE_ACTION` | a returned action violates step limits | do not execute |
| `COSMOS_RESPONSE_TOO_LARGE` | serialized response exceeds the configured limit | raise response limit only after review |
| `COSMOS_INFERENCE_FAILED` | sanitized unexpected model failure | fatal; inspect server logs |

## Verification and latency report

CPU-only protocol, conversion, safety, reconnection, and persistent-connection
tests live next to the implementation:

```bash
pytest --capture=no cosmos_framework/inference/robot_policy/robot_policy_test.py
```

Before robot rollout, run cold-start and steady-state tests using real JPEG
sizes and the final checkpoint/hardware. Report model-only `inference_ms` plus
client-observed HTTP RTT mean/P95/P99. Use those measurements to set
`inference_timeout_s` and the robot profile's horizon/prefetch values. No
latency numbers are claimed in this document because they have not been
measured against the final artifact and GPU.
