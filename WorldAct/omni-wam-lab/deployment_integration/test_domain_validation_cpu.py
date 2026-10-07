"""CPU checks for isolated domain validation fast path and generic fallback."""
import ast
import json
from pathlib import Path
import torch
from torch import nn
root=Path(__file__).resolve().parent
lab=root.parent
source=Path(json.loads((lab/'upstream.json').read_text())['source'])
def load_definition(path, name):
    tree=ast.parse(path.read_text())
    definition=next(n for n in tree.body if getattr(n,'name',None)==name)
    scope={'torch':torch,'nn':nn}
    exec(compile(ast.Module(body=[definition],type_ignores=[]),str(path),'exec'),scope)
    return scope[name]
rel=Path('vllm_omni/diffusion/models/cosmos3')
Old=load_definition(root/'backend_before/transformer_cosmos3.py','DomainAwareLinear')
New=load_definition(source/rel/'transformer_cosmos3.py','DomainAwareLinear')
make=load_definition(source/rel/'pipeline_cosmos3.py','_validated_wam_domain_tensor')
a=Old(3,5,32); b=New(3,5,32); b.load_state_dict(a.state_dict()); torch.manual_seed(0)
for domain in [0,1,26,31]:
 for shape in [(1,3),(1,33,3)]:
  x=torch.randn(shape,dtype=torch.bfloat16)
  assert torch.equal(a(x,torch.tensor([domain])),b(x,make(domain,32,'cpu')))
for bad in [-1,32,True,2.5]:
 try:make(bad,32,'cpu')
 except ValueError:pass
 else:raise AssertionError(bad)
for bad in [-1,32]:
 try:b(torch.zeros(1,3,dtype=torch.bfloat16),torch.tensor([bad]))
 except ValueError:pass
 else:raise AssertionError('generic fallback range check missing')
try:b(torch.zeros(2,3,dtype=torch.bfloat16),make(26,32,'cpu'))
except ValueError:pass
else:raise AssertionError('batch validation missing')
try:b(torch.zeros(1,2,2,3,dtype=torch.bfloat16),make(26,32,'cpu'))
except ValueError:pass
else:raise AssertionError('rank validation missing')
print('PASS: 4 valid domains x 2 ranks bitwise equal; invalid range/type, generic fallback, batch and rank rejected')
