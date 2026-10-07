WAM vLLM-Omni isolated experiment (2026-10-07)

Scope: ordinary video+action WAM, 4w EMA, single RTX5090. No PointFlow/FK,
no RTC, no ROS or robot command path. Existing port 18005 is not modified.

Current status: see build_status.json. This is NOT a robot-ready backend.
A successful conversion/CPU test does not establish inference correctness or speed.

Source: pinned revision in upstream.json. Local changes recorded in wam_upstream.patch.
Run benchmark_wam.py to ensure the patched SOURCE is used, not the original installed wheel.

Artifacts:
  artifacts/4w-ema-hf: native EMA export using the original exporter.
  artifacts/4w-ema-omni: lossless name conversion; all 549 tensors checked.
  ema_verification.json: direct DCP EMA checks of action heads and first/last GEN MLP.
  fixtures/observation_01.npz: native camera preprocessing, state, prompt, image_size.

The legacy YAML metadata adapter only changes export metadata loading. No training
configuration or model mathematics is replaced. Standard Diffusers conversion was
tried and refused by the existing Diffusers version (missing Edge constructor
fields). convert_wam.py instead reuses the native converter's key mapping and
checks full tensor-name coverage and all shapes against the local Edge component
schema. It never substitutes the base transformer weights. VAE, tokenizer and
scheduler are linked from the existing local base package; verify VAE parity during
numerical comparison.

WAM adaptation:
  33 action rows, first row observed 27D state, 64D zero padding, domain 26;
  condition index [0], temporal offset 0, absolute actions; only row 0 is removed.
  Native wrist-over-head composition, reflected canvas and formatted prompt.
  NumPy RandomState reset per modality; action noise BF16 rounding matches native.
  UniPC 4 steps, guidance 3, shift 5, seed 0, 15Hz.
  No video decoding in the action response path.

CPU checks (existing Cosmos environment):
  PYTHONPATH=../WorldAct-sft ../WorldAct-sft-pointflow-fk/.venv/bin/python -m unittest discover -s . -p test_wam_adapter.py -v

Offline benchmark (after runtime construction; consumes GPU, never controls robot):
  .venv/bin/python benchmark_wam.py --model artifacts/4w-ema-omni --packet fixtures/observation_01.npz --output results/bf16-eager --eager --warmup 2 --repeat 10

Do not compare latency while another inference request is running on the GPU.
Do not claim numerical equivalence based on shape or seed alone. Compare condition
latents, text tokens, first-step velocity and final raw actions before robot use.

Completed runtime check:
  Isolated environment installed; CUDA accessible; 4w EMA loaded on RTX5090.
  WAM generated 32x27 finite actions. Two warmups plus six measured eager calls:
  P50 707.95 ms, P95 709.81 ms. All eight action arrays identical.
  Fresh native model-time samples: 620.61 and 626.80 ms. Timing boundaries differ;
  native service stayed resident, so these are preliminary non-exclusive timings.
  Native vs Omni: mean position difference 2.54 cm, max 4.82 cm; this is NOT
  numerically validated for robot use. Text tokens match exactly (140/19).
  Numerical equivalence remains open before robot integration.
  Existing production service was not changed.

Speed experiments completed (2026-10-07; speed prioritized):
  Same saved observation, 4w EMA, 4 steps / CFG 3 / shift 5 / 33 frames.
  Each measured run: 5 warmup + 20 requests, OMP_NUM_THREADS=4.
  Warm offline complete-call P50 / P95 ms:
    Native deployed backend        621.93 / 630.04
    Omni regional BF16 fixed       630.98 / 633.64
    Omni full static BF16          639.58 / 640.86
    Omni BF16 first-frame input    590.21 / 592.11
    Omni FP8 first-frame input     440.54 / 444.14
    Omni FP8 repeat + audit        441.63 / 446.84
  Actual runtime: 336 FP8 linear layers, 28 compiled GEN layer forwards.
  First-frame optimization retained bitwise identical actions for all 25 calls.
  It avoids CPU allocation/normalization of unused future input frames, NOT
  shortening the generated video/action sequence. Native/Omni parity still open.
  Regional compile enum graph break fixed in isolated CUDNN backend only.
  Full static compile still breaks on Tensor.item and adds no speed benefit.
  Detailed conditions and limitations: results/speed_summary.json.

Reproduce fastest OFFLINE candidate from this directory:
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark_wam.py --model artifacts/4w-ema-omni --packet fixtures/observation_01.npz --output results/fp8-retest --first-frame-only --quantization fp8 --warmup 5 --repeat 20
  Omit --quantization fp8 for BF16. Add --eager to disable compilation.
  --compile-granularity full --static selects full static compile experiment.
  No production launcher, original model source, ROS, or robot controls changed.

Live protocol-v2 integration (2026-10-07):
  Optional backend: WorldAct-sft/cosmos_framework/inference/robot_policy/omni_http.py.
  Existing start_local_cosmos_deployment.sh now accepts --backend omni,
  --omni-model, --omni-quantization fp8|none and --record-video. Native is default.
  Real observations travel in memory; the fixture is never used by the live service.
  Native image/state/prompt construction, wire actions and smoothing are reused.
  Export checkpoint and frozen-config hash must match requested 4w run.
  Results with recording enabled: native HTTP RTT P50/P95 698/707 ms;
  Omni FP8 HTTP RTT 510/519 ms (same JPEG bytes, 20 measured requests each).
  Actual kernel is vLLM CutlassFP8ScaledMMLinearKernel; CUDNN_ATTN attention.
  Different state/image changes output; original input repeats exactly.
  29 image/state/action records and 29 finite video latents matched by session/id,
  fully drained at shutdown, no drops/errors. No VAE decode on serving path.
  Detailed evidence: results/http-integration/integration_summary.json.
  Native-vs-Omni numerical equivalence and FP8 task quality remain unverified.
  Test service 18016 is stopped. Existing native service 18005 was not stopped.

Operator command from /home/pjlab/ros2_ws/worktrees/wuji-hand-teleop-pipeline:
  COSMOS_DEPLOYMENT_CONFIG=../WorldAct/WorldAct-sft-pointflow-fk/examples/deployment/cosmos_singlerighthand_50k_retrain_v1_edge_protocol_v2.yaml ./src/scripts/start_local_cosmos_deployment.sh --backend omni --port 18006 --record-video --checkpoint-dir ../WorldAct/model_ch/real_ckpt/singlerighthand-edge-droid-50k-retrain-v1-0831/iter_000040000 --model-config-file ../WorldAct/model_ch/real_ckpt/singlerighthand-edge-droid-50k-retrain-v1-0831/config.deploy.yaml --config /home/wuji/ros2_ws/src/wuji_data_pipeline/config/cosmos_protocol_v2_50k_retrain_single24_sequential_blend8_hand2.yaml
  Add --check-only for file/provenance validation without GPU/HTTP/ROS.
  This preserves the latest sequential24 arm8/hand2 control profile.
  Normal launcher r/a/q lifecycle applies; only the operator starts the robot.
