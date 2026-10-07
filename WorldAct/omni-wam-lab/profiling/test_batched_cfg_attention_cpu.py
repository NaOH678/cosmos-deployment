"""CPU test of actual isolated candidate _forward_local implementation."""
import ast
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
root=Path(__file__).resolve().parents[1]
p=root/'batched_cfg_experiments/source/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py'
tree=ast.parse(p.read_text());cl=next(n for n in tree.body if getattr(n,'name',None)=='Cosmos3CrossAttention');fn=next(n for n in cl.body if getattr(n,'name',None)=='_forward_local');scope={'torch':torch,'AttentionMetadata':SimpleNamespace};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),scope);run=scope['_forward_local']
def attention(q,k,v,metadata=None):
 mask=None if metadata is None else metadata.attn_mask[:,None,None,:]
 return F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),attn_mask=mask).transpose(1,2)
torch.manual_seed(12)
results=[]
for lengths in ((3,1),(1,3),(140,19),(2,2)):
 B,S,H,D=2,7,2,8;M=max(lengths)
 q,k,v=[torch.randn(B,S,H,D) for _ in range(3)];ku,vu=[torch.randn(B,M,H,D) for _ in range(2)]
 for i,n in enumerate(lengths):ku[i,n:]=1000;vu[i,n:]=-900
 ref=torch.cat([attention(q[i:i+1],torch.cat((ku[i:i+1,:n],k[i:i+1]),dim=1),torch.cat((vu[i:i+1,:n],v[i:i+1]),dim=1)) for i,n in enumerate(lengths)]).reshape(B,S,-1)
 mask=torch.ones(B,M+S,dtype=torch.bool)
 for i,n in enumerate(lengths):mask[i,n:M]=False
 for mode in ('split','masked'):
  obj=SimpleNamespace(attn=attention,_cfg_text_lengths=lengths,_cfg_attention_mode=mode,_cfg_attention_mask=mask)
  got=run(obj,q,k,v,ku,vu)
  torch.testing.assert_close(got,ref,rtol=1e-5,atol=1e-6)
  # Existing stale attributes must not contaminate a later B=1 request.
  got1=run(obj,q[:1],k[:1],v[:1],ku[:1,:lengths[0]],vu[:1,:lengths[0]])
  torch.testing.assert_close(got1,ref[:1],rtol=1e-5,atol=1e-6)
  results.append((lengths,mode))
print('PASS actual candidate: sentinel-padding isolation, masked/split independent-branch parity, stale-attribute B1 fallback',results)
