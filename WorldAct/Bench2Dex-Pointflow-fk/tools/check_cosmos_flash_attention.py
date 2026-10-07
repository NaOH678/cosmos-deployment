"""Check real Cosmos Flash2 dispatch against FP32 attention on the local GPU."""
import argparse
import json
import os
from pathlib import Path
import sys
import typing
import typing_extensions
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, default=Path('outputs/cosmos_local/cosmos_task21_inference_bundle/source'))
parser.add_argument('--output', type=Path, default=Path('analysis/cosmos_local_validation/flash_attention_check.json'))
args = parser.parse_args()
os.environ['COSMOS_TRAINING']='0'
os.environ['COSMOS_FLASH2_VARLEN']='1'
if not hasattr(typing,'override'): typing.override=typing_extensions.override
sys.path.insert(0,str(args.source.resolve()))
import torch
import flash_attn
from cosmos_framework.model.attention.frontend import attention
from cosmos_framework.model.attention.masks import CausalType

def reference(q,k,v,causal):
    q,k,v=[x.float().transpose(1,2) for x in (q,k,v)]
    if q.shape[1]!=k.shape[1]:
        r=q.shape[1]//k.shape[1]; k=k.repeat_interleave(r,1);v=v.repeat_interleave(r,1)
    logits=q@k.transpose(-1,-2)/q.shape[-1]**0.5
    if causal:
        nq,nk=q.shape[-2],k.shape[-2]
        mask=torch.arange(nk,device=q.device)[None,:] <= torch.arange(nq,device=q.device)[:,None]+nk-nq
        logits.masked_fill_(~mask,float('-inf'))
    return (logits.softmax(-1)@v).transpose(1,2)

torch.manual_seed(123)
results=[]
with torch.inference_mode():
 for packed,causal in [(False,True),(False,False),(True,True),(True,False)]:
    qs=[127,65] if packed else [127]
    ks=qs if causal else ([159,97] if packed else [159])
    q=torch.randn(1,sum(qs),16,128,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(1,sum(ks),8,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
    kwargs={}
    if packed:
        cq=torch.tensor([0,qs[0],sum(qs)],device='cuda',dtype=torch.int32)
        ck=torch.tensor([0,ks[0],sum(ks)],device='cuda',dtype=torch.int32)
        kwargs=dict(cumulative_seqlen_Q=cq,cumulative_seqlen_KV=ck,max_seqlen_Q=max(qs),max_seqlen_KV=max(ks))
    y=attention(q,k,v,is_causal=causal,causal_type=CausalType.DontCare if causal else None,backend='flash2',**kwargs)
    expected=[];qi=ki=0
    for nq,nk in zip(qs,ks):
        expected.append(reference(q[:,qi:qi+nq],k[:,ki:ki+nk],v[:,ki:ki+nk],causal));qi+=nq;ki+=nk
    ref=torch.cat(expected,1)
    error=(y.float()-ref).abs()
    torch.testing.assert_close(y.float(),ref,atol=.02,rtol=.02)
    assert torch.isfinite(y).all()
    results.append(dict(packed=packed,causal=causal,max_absolute_error=error.max().item(),mean_absolute_error=error.mean().item()))
report=dict(torch=torch.__version__,flash_attention=flash_attn.__version__,gpu=torch.cuda.get_device_name(),cases=results,status='passed')
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
