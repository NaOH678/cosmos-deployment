5090 Omni FP8 profiling, 2026-10-07

Scope: offline only, no ROS or actuator commands, original port 18005 server kept resident.
Fixed 4w EMA, 4 steps, CFG 3, shift 5, first-frame preprocessing enabled.
5 warmups, 20 unprofiled measurements, then 2 torch.profiler CPU/CUDA requests.
profile_wam.py measures prepared observation packet -> CPU actions, not full HTTP RTT.
Baseline P50 441.71ms, P95 443.83ms. Profiled requests 507.21/500.46ms excluded.

Artifacts:
fp8_baseline/report.json: unprofiled and profiled per-request timing
fp8_baseline/traces/20261007-163351_warm_fp8/trace_rank0.json.gz: Chrome/Perfetto trace
fp8_baseline/traces/20261007-163351_warm_fp8/profiler_out_0.txt: operator table
fp8_baseline/traces/20261007-163351_warm_fp8/memory_snapshot_rank0.pickle: allocator snapshot
profile_summary.json: raw CUDA-event aggregation and limitations
analyze_trace.py: reproducible aggregation

Measured profiled GPU work per request: 374.16ms kernel total, including
208.65ms SM120 CUTLASS FP8 GEMM, 120.65ms cuDNN SDPA, ~13.49ms VAE convolution.
CPU synchronization is not additive with GPU compute; do not subtract waiting time.
Actual H2D 0.22ms/request, so PCIe transfer bandwidth is not dominant.
Zero cudaGraphLaunch/cuGraphLaunch events: regional compile != graph replay.

Candidate domain_sync_candidate is an isolated source overlay. All unchanged files
are symlinks; transformer_cosmos3.py and pipeline_cosmos3.py are independent files.
It validates integer domain range once at CPU ingress and marks that generated tensor,
retaining generic fallback GPU range validation, batch and rank validation.
Do not edit symlinked files.
profile_domain_sync.py benchmarks this candidate. test_domain_validation.py tests
4 domain ids, two tensor ranks, invalid ids/types, batch and rank constraints.
Do not deploy before warm end-to-end and GPU output parity checks.

Domain sync candidate measured after baseline:
P50 437.73ms / P95 441.40ms (5 warm + 20 measurements, profiler off).
20/20 raw output pairs bitwise equal, max absolute difference 0.
Profiler confirms all 32 repeated domain bool syncs across 2 requests removed.
Net speed difference only ~4ms; waiting moves to later synchronization, not saved wholesale.
This is a small candidate improvement, not a large breakthrough. Generic checks retained.
See domain_sync_comparison.json; GPU released to backend optimization agent.

CPU first-frame packet prototype:
benchmark_packet_cpu.py, cpu_packet_comparison.json, cpu_packet.log.
CUDA_VISIBLE_DEVICES empty, no model/GPU context. 5 recent robot observations,
40 interleaved baseline/candidate timing pairs each, 5 warmups, 4 torch threads.
Baseline 6.33ms P50 / 8.09ms P95; first-frame 2.54ms / 4.27ms.
All 5 packets equal: first_frame, action, image_size and entire metadata/prompt.
Both variants exclude native image_size GPU roundtrip; actual savings there unmeasured.
Prototype derives original native function and changes only full-video allocation
into expand shape-view, with image_size remaining CPU. This is not production design:
production should factor common compose/pad/prompt logic into shared first-frame helper.
Expanded video must remain formatter-only and must not be provided as future frames
for model inference. Formatter currently reads video shape only.
