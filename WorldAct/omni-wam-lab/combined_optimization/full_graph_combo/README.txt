Three-optimization isolated HTTP benchmark, 2026-10-07

Candidate combines CPU first-frame packet, CPU domain validation, full GEN CUDA Graph.
source/vllm_omni is a FULL INDEPENDENT copy; no symlinks. Merged transformer contains
both domain trusted fastpath and full GEN graph dispatch; pipeline contains domain
CPU ingress check. merge_audit.json records input/copied file hashes and checks.
Production sources/configs untouched. No ROS or robot operations.

Reproduction: parent prepare_full_graph_combo.py refreshes final candidate files.
run_http_abba_graph.py launches one temporary loopback18026 server at a time,
original18005 kept resident idle. A/B/B/A with5realHTTP warmups+20 measured each,
5 saved observations in identical order. Same encodedJPEG bytes (quality95),
4wEMA/FP8/4steps/CFG3/shift5/binomial5. No record/video capture in either arm.
RTT includes pickle/HTTP, JPEG decode, preprocess/model/postprocess and response;
excludes client JPEG encode and disk arrays written after timer.

40 timed requests per implementation:
Baseline P50=501.21ms P95=513.16ms max523.95ms.
Combined P50=494.50ms P95=506.75ms max524.93ms.
Observed median benefit6.71ms(~1.34%). Worst sample did NOT improve.
All60 matched cross-stage32x27wire pairs bitwise equal, maxabs0.
Previous TWO-optimization run496.98ms is a different run: cannot establish extra
fullgraph gain from comparing that result to494.50ms. Individual speedups cannot
be added to promise15ms. Need same-run two-vs-three comparison if required later.

Cold shapes: service syntheticzero warmup creates3138GEN-token graphs140/19text;
realinput3399GEN tokens captures two more graphs (~150-160ms capture each).
Four shape-specialized graphs observed; timed requests exclude these captures
via5realHTTP warmups. Any production rollout must warm ACTUAL input shape;
current syntheticwarmup alone does not guarantee first request latency.

All four owned servers exited rc0. Temporary server closed, GPU released.
No production rollout performed. summary.json/report.json and stage logs/configs/
audit/output arrays retain full evidence. Graph correctness/actual launch trace
also independently verified by full_gen_graph_experiments prior to this run.
