Complete GEN-stack CUDA Graph experiment, RTX5090, 2026-10-07

Authoritative results: summary.json. GPU experiments ended; production was not changed.
No ROS, recovery, enabling or robot commands were used.

Five real observations, 5 warmups +20 measurements:
  Existing serial CFG, unchanged dynamic compile: 440.30ms P50 /443.20ms P95.
  Complete GEN-stack graphs:                    432.48ms P50 /435.86ms P95.
  P50 reduction7.83ms (1.78%). Modest benefit; needs matched HTTP confirmation.
All25 raw32x27 action-array pairs are bitwise equal. Same observation after four others
repeats exactly; different observations produce different outputs.

Two profiled requests contain16 actual cudaGraphLaunch calls =8/request:
4 denoising steps x2 serial CFG branches. Each graph captures all28 GEN layers plus
final normalization. Inner Inductor cudagraphs are explicitly disabled; this is not
224 per-layer graph launches. Profiling requests are excluded from timing statistics.

Design:
  Independent static hidden/KV/RoPE buffers and graph for140/19 real text lengths.
  Every invocation copies current hidden state, all28 KV pairs and branch RoPE.
  UND, GEN preprocess, original postprocess, CFG arithmetic and UniPC stay unchanged.
  Graph output goes immediately into graph-external projection, producing independent
  video/action tensors before reuse. No per-layer clones.
  Compilation remains original default dynamic regional, unlike the prior failed
  reduce-overhead/clone experiment.
Cold graph preparation (including3 side-stream warmups):166ms cond,156ms uncond.
Peak allocated5.50GB versus baseline5.42GB; maxima include initialization and warmup.

Runs:
  smoke: failed because GenPrepared is NamedTuple, not dataclass; failure retained.
  smoke_namedtuple: fixed _replace, graph capture works; two old-fixture warm calls
    were421.88/418.34ms and matched previous reference. Do not use this small preliminary
    run instead of the formal multi-observation results.
  baseline_corpus: same isolated source/runner, graph off; formal comparison baseline.
  graph_corpus: graph on,5-observation rotation,20 samples plus2 excluded profile calls.

CPU test test_buffer_lifetime_cpu.py uses NamedTuple plus simulated graph execution to
verify fixed-buffer updates, separate branch slots and graph-external output lifetime.
It passed; actual GPU action equality above supplies real-kernel evidence.

Reproduce from omni-wam-lab:
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 .venv/bin/python full_gen_graph_experiments/benchmark_full_gen.py --model artifacts/4w-ema-omni --packet fixtures/observation_01.npz --corpus optimization_20261007/corpus.json --output full_gen_graph_experiments/results/retest --first-frame-only --quantization fp8 --warmup 5 --repeat 20
Add --full-gen-graph to enable candidate, --profile for2 excluded final profile requests.
No production environment flags are changed by these commands.

Integration contract for another isolated experiment:
  Candidate source: full_gen_graph_experiments/source.
  Add full_gen_graph_experiments to PYTHONPATH, set WAM_FULL_GEN_GRAPH=1.
  Optional WAM_FULL_GEN_AUDIT path records capture snapshots.
  Dynamic default compile only; no inner compiler graphs.
  Graph cache lacks eviction/model-hot-swap/concurrent execution support; prototype is
  limited to single-GPU B1 video/action, not a general serving implementation.
This experiment does not establish a speed limit for5090.
