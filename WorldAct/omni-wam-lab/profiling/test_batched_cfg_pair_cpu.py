"""Run actual predict_pair orchestration with CPU fake model, no CUDA calls."""
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'batched_cfg_experiments'))
from batched_cfg_impl import predict_pair
class Layer:
 def __init__(self):self.cross_attention=SimpleNamespace();self.ropes=[]
 def __call__(self,hidden,**kw):
  assert hidden.shape[0]==2
  self.ropes.append(kw['freqs_cos'].clone())
  return hidden+kw['freqs_cos'].reshape(2,1,1)
class Model:
 def __init__(self):self.cached_kv=None;self.cached_freqs_gen=None;self.gen_layers=[Layer()];self.und_calls=0
 def _gen_preprocess(self,text_ids,text_mask,hidden_states,**kw):
  assert hidden_states.dtype==torch.bfloat16
  n=text_ids.shape[1]
  if self.cached_kv is None:
   self.und_calls+=1
   self.cached_kv=[(torch.full((1,n,1,2),float(n)),torch.full((1,n,1,2),float(n+1)))]
   self.cached_freqs_gen=(torch.tensor([float(n)]).reshape(1,1,1,1),torch.tensor([float(n+10)]).reshape(1,1,1,1))
  return SimpleNamespace(hidden_gen=hidden_states,ulysses_size=1,has_control=False,has_sound=False)
 def norm_moe_gen(self,x):return x
 def _gen_postprocess(self,x,prep):return (x,x+1)
m=Model();p=SimpleNamespace(transformer=m,dtype=torch.bfloat16,sampling_dtype=torch.float32)
for lens in ((3,1),(5,2)):
 cache={};ids=[torch.ones((1,n),dtype=torch.long) for n in lens]
 for _ in range(2):
  out=predict_pair(p,{'hidden_states':torch.zeros(1,2,2)},ids[0],ids[0],ids[1],ids[1],cache)
  assert cache['lengths']==lens
  assert [float(x[0][0,0,0]) for x in out]==list(lens)
  assert all(x.dtype==torch.float32 for branch in out for x in branch)
 assert m.gen_layers[0].cross_attention._cfg_text_lengths==lens
assert m.und_calls==4 # once per branch per request, not once per step
print('PASS actual predict_pair: independent branch RoPE rows, per-request cache rebuild, step cache reuse, float32 output tuples')
