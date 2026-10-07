"""CPU-only packet A/B. No model load, CUDA context, HTTP, or robotics.

Prototype derives the exact native function for semantic parity. Production should
factor a shared first-frame helper instead of using runtime source rewriting.
"""
import inspect
import json
from pathlib import Path
import sys
import textwrap
import time
from types import SimpleNamespace
import numpy as np
import torch
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root.parent/'WorldAct-sft'))
from cosmos_framework.inference.robot_policy import adapters
from cosmos_framework.inference.robot_policy.omni_http import OmniWamAdapter
from cosmos_framework.inference.robot_policy.config import load_robot_policy_config

torch.set_num_threads(4)
corpus=json.loads((root/'optimization_20261007/corpus.json').read_text())
c=load_robot_policy_config(corpus['config'])
original=textwrap.dedent(inspect.getsource(adapters.SingleRightHandCosmosAdapter._build_batch))
assert original.count('.to(device="cuda")')==1
cpu=original.replace('.to(device="cuda")','')
block='video = torch.zeros((3, target_frames, target_h, target_w), dtype=first_frame.dtype)\n    video[:, :1] = first_frame'
assert cpu.count(block)==1
optimized=cpu.replace(block,'video = first_frame.expand(-1, target_frames, -1, -1)')
# The expanded tensor is a shape-only formatter input and first-frame source.
# It must never be passed to the model as a 33-frame conditioning video.
def make_adapter(source):
    ns=dict(vars(adapters));exec(compile(source,'<packet_cpu_prototype>','exec'),ns)
    obj=SimpleNamespace(config=c,input_video_key='video')
    obj._build_batch=lambda images,state:ns['_build_batch'](obj,images,state)
    return obj
old,new=make_adapter(cpu),make_adapter(optimized)
def packet(obj,images,state):return OmniWamAdapter._make_packet(obj,images,state)
records=[];timings={'baseline':[],'first_frame':[]}
for entry in corpus['observations']:
    with np.load(entry['source'],allow_pickle=False) as z:
        images={'head':z['head'].copy(),'right_wrist':z['right_wrist'].copy()};state=z['state'].copy()
    a,b=packet(old,images,state),packet(new,images,state)
    equal={k:bool(np.array_equal(a[k],b[k])) for k in ('first_frame','action','image_size')}
    equal['metadata']=a['metadata']==b['metadata']
    assert all(equal.values()),equal
    for _ in range(5):packet(old,images,state);packet(new,images,state)
    local={'baseline':[],'first_frame':[]}
    for i in range(40):
        order=[('baseline',old),('first_frame',new)]
        if i%2:order.reverse()
        for name,obj in order:
            start=time.perf_counter();packet(obj,images,state);elapsed=(time.perf_counter()-start)*1000
            local[name].append(elapsed);timings[name].append(elapsed)
    records.append({'sequence':entry['sequence'],'equal':equal,'timings':{k:{'p50_ms':float(np.percentile(v,50)),'p95_ms':float(np.percentile(v,95))} for k,v in local.items()}})
result={'scope':'CPU camera arrays -> full Omni packet. Original native _build_batch with only image_size cuda transfer removed in BOTH variants. No GPU context, model, network or latent recording; timings do not include eliminated GPU roundtrip. 4 torch threads; 5 observations x40 interleaved repeats after5warmups each. Concurrent backend GPU work may contend for CPU.', 'records':records,'aggregate':{k:{'p50_ms':float(np.percentile(v,50)),'p95_ms':float(np.percentile(v,95))} for k,v in timings.items()},'avoided_full_video_allocation_bytes':3*33*736*544,'physical_first_frame_bytes':3*736*544,'production_recommendation':'Factor native compose/pad/prompt/action into shared helper with frame count explicit. For Omni return first frame directly and keep image_size on CPU. Never expose expanded video as actual future frames to model.'}
(root/'profiling/cpu_packet_comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['aggregate'],indent=2));print('All5 observation packets exactly equal including full prompt metadata')
