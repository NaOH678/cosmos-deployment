Offline backend experiments, RTX 5090 / SM120, 2026-10-07.
No production source or robot control changed.

benchmark_backend.py is an independent copy of benchmark_wam.py with explicit attention,
compile mode, optional layer-output clone, corpus rotation and separate profiler calls.
Only Cosmos3GenDecoderLayer.forward receives experimental torch.compile mode override;
all other torch.compile calls retain their original options. FA4 dependencies are isolated
under fa4_deps and added to Python import path only for explicit FLASH_ATTN runs.

Official flash-attn-4==4.0.0b34 installed --no-deps to fa4_deps; its SM120 kernel is used.
Sources: https://github.com/Dao-AILab/flash-attention and pinned upstream.json in parent.

Strict backend results (5 warmup, 20 measured, same original observation):
  CUDNN_ATTN dynamic regional FP8: P50 440.18 ms, P95 442.69 ms.
  FLASH_ATTN FA4 dynamic regional FP8: P50 448.44 ms, P95 451.67 ms.
  CUDNN_ATTN static regional FP8: P50 455.62 ms, P95 463.40 ms.
These are single-session preliminary comparisons, not architecture-wide rankings.

CUDA Graph experiment history:
  cudnn_reduce_overhead: INVALID experiment injection did not reach worker. Ignore.
  cudnn_reduce_overhead_verified: startup failed on compile decorator options conflict.
  cudnn_graph_profile: mode correctly applied to 28 GEN layers; actual capture began,
    then warmup failed with CUDAGraph output overwritten by subsequent invocation.
  cudnn_graph_clone: explicit clone outside each compiled GEN layer made calls run.
    TORCH_LOGS=cudagraphs produced excessive log output; timing is not a clean comparison.
    27 requests completed and trace exported; expensive key_averages postprocessing was
    terminated. graph_trace_summary.json counts 448 cudaGraphLaunch across 2 requests,
    i.e. 224 per request (28 layers x 4 steps x 2 CFG). This is per-layer graph replay.
  corpus_graph_clone: clean run without debug logging/profile; 5 real observations rotated.
    P50 442.25 ms, P95 452.28 ms. Requires corresponding no-graph output comparison.

Model and sampler semantics unchanged: 4w EMA, FP8 linear layers, BF16 attention,
4 UniPC steps, CFG 3, 33 frames / 32 returned absolute actions, first-frame preprocessing.
All timings packet->CPU raw actions; no HTTP or production recording in this runner.
Original native service stayed resident but was not queried; desktop GPU not exclusive.
GPU memory utilization in this Omni revision only sizes paged-scheduler KV cache;
ordinary WAM diffusion does not benefit from raising it from .90 to .95.

Final five-observation corpus comparison (results in summary.json):
  cuDNN dynamic: 441.19 / 445.84 ms P50/P95.
  FA4 dynamic: 447.94 / 453.13 ms.
  cuDNN static + reduce-overhead + layer-output clone: 442.25 / 452.28 ms.
No meaningful acceleration from these alternatives; retain existing cuDNN dynamic.

Numerical caveat: graph candidate vs dynamic baseline has mean/max position difference
3.18/6.54 cm; FA4 vs baseline 3.49/9.01 cm. These are prediction differences, not measured
task errors. Graph candidate changes static compile and execution mode together; difference
is not isolated to graph replay. No candidate is recommended for production.
All backends repeat exactly after intervening four other observations (A->B->C->D->E->A),
and different observations do change outputs. No observed stale-output reuse.

Legacy corpus reports preserve initialization_packet and explicitly note input_metadata
came from the initial fixture; each samples entry identifies the actual new corpus packet,
and all per-request metadata/frame/state/prompt were reloaded from that packet. Runner fixed
for future runs to use corpus first packet as top-level initialization metadata.

The graph debug .log is gzip-compressed; .evidence.txt retains compact key lines.
Only this experiment process was terminated during slow profiler aggregation, after trace
export; original native production service remained untouched. All GPU experiments ended.
