Combined CPU first-frame packet + CPU domain validation, offline HTTP ABBA
2026-10-07; no ROS, recovery, actuator publish or real robot commands.

Changes isolated here:
first_frame_packet.py: reusable CPU packet helper using original camera compose,
reflection pad and ActionPromptJsonFormatter. No native function source rewriting.
Shape-only expanded temporal view supplies formatter duration; return only real
first frame. Action futures remain zero. Explicit 27D/32-step guard.
serve_candidate.py: opt-in process-local Omni adapter _make_packet replacement.
candidate_lab/upstream.json points to profiling/domain_sync_candidate isolated
source, retaining CPU request domain check and generic fallback validation.
Original production files and existing 18005 service unchanged.

run_http_abba.py launches ONE temporary server at a time on loopback18026:
A baseline, B combined, B combined, A baseline. Each server starts fresh, then
5 HTTP warmups +20 measured requests: five recent real observations repeated
in identical order. Each stage resets session and model/postprocess state.
Same 4w EMA, FP8, 4 diffusion steps, CFG3, shift5, resolution and binomial5.
Captured YAML service_mode=hold was CLI-overridden in original deployment;
all test configs explicitly use full. Original checkpoint/config paths retained.
Recording and video latent capture disabled in ALL stages; not a record-video test.
JPEG quality95 bytes encoded once per observation and identical across stages.
RTT covers client pickle, HTTP request, server JPEG decode/preprocess/model/
postprocess, HTTP response/unpickle; excludes pre-encoding JPEG and disk save.
Original 18005 occupies ~9904MiB but idle, desktop remains; GPU not fully exclusive.

Results, 40 measured requests per implementation:
Baseline P50=504.27ms P95=513.09ms
Combined P50=496.98ms P95=505.45ms
P50 improvement7.29ms (~1.45%), a small improvement.
Per-stage P50 A1=503.29, B1=496.98, B2=496.92, A2=504.61ms.
Server reported model interval medians: baseline436.75ms, combined435.89ms.
Whole server adapter wall medians: baseline459.30ms, combined453.31ms.
Don't compare this HTTP RTT directly with prepared packet or historical robot RTT.

60/60 matched cross-stage output pairs (20 each vs first baseline) are bitwise equal.
These are returned32x27 wire actions after common postprocessing, not task accuracy.
Independent CPU helper parity: all5 saved images/states match native first_frame,
action,image_size and fullmetadata including prompt. SHA256 in packet_parity.json.

All owned servers exited normally; port18026 closed, original18005 alive.
Results report.json, abba.log, stage config/serverlog/runtime audit/output arrays
retained. Domain isolated overlay symlinks unchanged files; don't edit symlink files.
