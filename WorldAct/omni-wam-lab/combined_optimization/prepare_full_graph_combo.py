"""Prepare independent merged source; CPU filesystem work only, no benchmark."""
import hashlib,json,shutil
from pathlib import Path
root=Path(__file__).resolve().parent;lab=root.parent
base=Path(json.loads((lab/'upstream.json').read_text())['source'])
graph=lab/'full_gen_graph_experiments'
out=root/'full_graph_combo';out.mkdir(exist_ok=True)
source=out/'source'
if not (source/'vllm_omni').exists():
 shutil.copytree(base/'vllm_omni',source/'vllm_omni',symlinks=False,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
rel=Path('vllm_omni/diffusion/models/cosmos3')
# Base is same pinned source; start with graph-modified transformer then add
# validated-domain fast path with exact single-match guards.
s=(graph/'source'/rel/'transformer_cosmos3.py').read_text()
old='        if domain_id.ndim == 0:\n';assert s.count(old)==1
s=s.replace(old,'        trusted_bounds = getattr(domain_id, "_wam_validated_domain_bounds", None)\n'+old)
old='        if torch.any((domain_id < 0) | (domain_id >= self.num_domains)):';assert s.count(old)==1
s=s.replace(old,'        if trusted_bounds != self.num_domains and torch.any((domain_id < 0) | (domain_id >= self.num_domains)):')
(source/rel/'transformer_cosmos3.py').write_text(s)
# Domain candidate changes pipeline only in CPU ingress + validated tensor ctor.
shutil.copy2(lab/'profiling/domain_sync_candidate'/rel/'pipeline_cosmos3.py',source/rel/'pipeline_cosmos3.py')
candidate=out/'candidate_lab';candidate.mkdir(exist_ok=True)
for origin in (lab/'wam_adapter.py',graph/'full_gen_graph_impl.py'):
 shutil.copy2(origin,candidate/origin.name)
upstream=json.loads((lab/'upstream.json').read_text());upstream['source']=str(source)
(candidate/'upstream.json').write_text(json.dumps(upstream,indent=2))
audit={'no_gpu_used':True,'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (graph/'source'/rel/'transformer_cosmos3.py',lab/'profiling/domain_sync_candidate'/rel/'pipeline_cosmos3.py',graph/'full_gen_graph_impl.py')},'merged':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source/rel/'transformer_cosmos3.py',source/rel/'pipeline_cosmos3.py',candidate/'full_gen_graph_impl.py')},'assertions':{'domain_fastpath': 'trusted_bounds != self.num_domains' in s,'full_graph_dispatch':'return run_full_gen_graph(self, prep, self._run_gen_stack_original)' in s,'copied_not_symlink':all(not p.is_symlink() for p in source.rglob('*'))}}
assert all(audit['assertions'].values());(out/'merge_audit.json').write_text(json.dumps(audit,indent=2))
# Derive same proven HTTP runner: preserve baseline, corpus, transport, seeds,
# recording settings; only candidate source and graph flag differ.
s=(root/'run_http_abba.py').read_text()
s=s.replace("root=Path(__file__).resolve().parent;lab=root.parent;worktrees=lab.parent.parent", "root=Path(__file__).resolve().parent;lab=root.parents[1];worktrees=lab.parent.parent")
s=s.replace("str(root/'serve_candidate.py')", "str(root.parent/'serve_candidate.py')")
s=s.replace("WAM_COMBINED_CANDIDATE='1' if label=='combined' else '0',", "WAM_COMBINED_CANDIDATE='1' if label=='combined' else '0',WAM_FULL_GEN_GRAPH='1' if label=='combined' else '0',WAM_FULL_GEN_AUDIT=str(directory/'full_gen_audit.json'),")
s=s.replace('ABBA to mitigate drift.', 'ABBA to mitigate drift. Candidate combines first-frame helper, CPU domain validation, full GEN CUDA Graph.')
(out/'run_http_abba_graph.py').write_text(s)
print('Prepared merged source and HTTP runner only. No GPU/server launched:',out)
