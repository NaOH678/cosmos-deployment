"""CPU-only production scope, disable switch, exception recovery and cap checks."""
import ast
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from wam_gen_graph import run_full_gen_graph
source=Path(json.loads((root/'upstream.json').read_text())['source'])
tree=ast.parse((source/'vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py').read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and any(getattr(c,'name','')=='_forward_robolab_policy' for c in n.body))
method=next(n for n in cls.body if getattr(n,'name','')=='_forward_robolab_policy')
scope={}
exec(compile(ast.Module(body=[method],type_ignores=[]),'<actual production WAM wrapper>','exec'),scope)

class ScopeTest(unittest.TestCase):
    def test_wam_only_and_disable(self):
        for flag,wam,expect in [('1',True,True),('1',False,False),('0',True,False),('',True,False)]:
            model=SimpleNamespace(_wam_gen_graph_enabled=False)
            node=SimpleNamespace(transformer=model,_forward_robolab_policy_impl=lambda *a:model._wam_gen_graph_enabled)
            with patch.dict(os.environ,{'WAM_FULL_GEN_GRAPH':flag}):
                self.assertEqual(scope['_forward_robolab_policy'](node,None,SimpleNamespace(observation={'_wam_native':wam}),0),expect)
            self.assertFalse(model._wam_gen_graph_enabled)
    def test_exception_restores_flag(self):
        def fail(*args):raise RuntimeError('intentional')
        model=SimpleNamespace(_wam_gen_graph_enabled=False)
        with patch.dict(os.environ,{'WAM_FULL_GEN_GRAPH':'1'}):
            with self.assertRaises(RuntimeError):scope['_forward_robolab_policy'](SimpleNamespace(transformer=model,_forward_robolab_policy_impl=fail),None,SimpleNamespace(observation={'_wam_native':True}),0)
        self.assertFalse(model._wam_gen_graph_enabled)
    def test_cache_capacity_falls_back_without_cuda(self):
        model=SimpleNamespace(cached_kv=[(torch.zeros(1,3,1,1),torch.zeros(1,3,1,1))],cached_freqs_gen=(torch.zeros(1,2,1,1),torch.zeros(1,2,1,1)),_wam_gen_graph_states={i:{} for i in range(4)})
        prep=SimpleNamespace(ulysses_size=1,has_control=False,has_sound=False,hidden_gen=torch.zeros(1,2,2),s_video=2,s_action=0,has_action=True,use_multi_control_attention=False)
        self.assertEqual(run_full_gen_graph(model,prep,lambda _:123),123)
        self.assertEqual(len(model._wam_gen_graph_states),4)
    def test_unsupported_uses_original(self):
        prep=SimpleNamespace(ulysses_size=2,has_control=False,has_sound=False,hidden_gen=torch.zeros(1,2,2))
        self.assertEqual(run_full_gen_graph(SimpleNamespace(),prep,lambda _:321),321)

if __name__=='__main__':unittest.main()
