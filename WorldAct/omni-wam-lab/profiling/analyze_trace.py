"""Aggregate raw CUDA events without double counting CPU/GPU annotations."""
import collections
import gzip
import json
from pathlib import Path

root = Path(__file__).resolve().parent
run = root / 'fp8_baseline'
trace = next((run / 'traces').glob('*/*.json.gz'))
events = json.load(gzip.open(trace))['traceEvents']
summary = {}
for cat in ('kernel', 'gpu_memcpy', 'cuda_runtime', 'cuda_driver'):
    totals = collections.defaultdict(lambda: {'total_ms': 0., 'count': 0})
    for e in events:
        if e.get('cat') == cat and e.get('ph') == 'X':
            v = totals[e['name']]
            v['total_ms'] += e['dur'] / 1000
            v['count'] += 1
    summary[cat] = sorted([{'name': k, **v, 'per_request_ms': v['total_ms']/2} for k,v in totals.items()], key=lambda x:-x['total_ms'])
summary['unprofiled_benchmark'] = {k:json.loads((run/'report.json').read_text())[k] for k in ('p50_ms','p95_ms')}
summary['profiled_requests'] = 2
summary['trace'] = str(trace)
summary['scope'] = 'Prepared observation packet -> Omni worker -> CPU raw actions, excludes native camera composition, HTTP codec/network, production recording.'
summary['caveats'] = ['Profiler adds overhead; timings below are attribution, not unprofiled stage latency.', 'cudaStreamSynchronize durations overlap GPU compute and cannot be subtracted from request latency.', 'GPU kernel sums exclude idle gaps and annotations.', 'Only one fixed observation; resident original server occupies GPU memory but was idle.']
summary['findings'] = {
    'cuda_graph_replay_calls':sum(e.get('name') in ('cudaGraphLaunch','cuGraphLaunch') for e in events),
    'compiled_generation_graph_calls':448,
    'fp8_cutlass_device_ms_per_request':sum(e['dur'] for e in events if e.get('cat')=='kernel' and 'enable_sm120_family' in e.get('name',''))/2000,
    'cudnn_attention_device_ms_per_request':sum(e['dur'] for e in events if e.get('cat')=='kernel' and 'sdpa' in e.get('name',''))/2000,
    'total_device_kernel_ms_per_request':sum(e.get('dur',0) for e in events if e.get('cat')=='kernel')/2000,
    'domain_range_check_sync': '32 is_nonzero -> item syncs in 2 requests; domain checked twice per CFG forward. Move validation to CPU request ingress once, retaining generic fallback checks.',
    'priority': ['SM120 attention comparison (about 120 ms/request)', 'Remove repeated domain-id host synchronization and verify benefit', 'CUDA Graph entire stable model forward to reduce launch overhead; 0 graph replay calls currently', 'CPU first-frame packet builder avoids allocation/zeroing of 33 frames and GPU metadata roundtrip', 'H2D overlap lower priority: measured actual transfer about 0.22ms/request']
}
(root/'profile_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary['findings'],indent=2))
