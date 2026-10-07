"""Record native DA3 chunk metadata without changing model computations."""
import os,sys,json
from pathlib import Path
from types import SimpleNamespace
B=Path(__file__).resolve().parent;N=Path('/mnt/afs/WorldAct-pointflow-native');R=N/'Track4World_portable/Track4World'
sys.path[:0]=[str(R),str(R.parent),str(N/'pf_out')]
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TRACK4WORLD_DA3_MODEL='/data/shichaojian/checkpoints/DA3NESTED-GIANT-LARGE-1.1',TORCH_HOME='/data/shichaojian/checkpoints/torch-hub')
import torch,numpy as np
import demo,run_efep_export as native
from track4world.nets.blocks import InputPadder
# Same seed, model, source pixels, normalization, and native get_fmaps batching.
torch.manual_seed(17);np.random.seed(17);torch.set_num_threads(8);torch.set_grad_enabled(False)
args=SimpleNamespace(coordinate='world_depthanythingv3',ckpt_init=str(R/'checkpoints/track4world_da3.pth'),use_original_backbone=False,metric_scale=True)
model=demo.load_model(args,json.loads((R/'track4world/config/eval/v1.json').read_text()))
item=json.loads((B/'run/efep_raw/task21_episode000000/COMPLETE.json').read_text())
images,h,w=native.read_video(item,np,torch)
images=images.to(torch.float16)/255.;images=(images-model.image_mean)/model.image_std
images=images.contiguous().reshape(-1,3,h,w);images=InputPadder(images.shape).pad(images)[0]
records=[];original=model.forward_point;offset=0
sample={};selected={0,63,100,200,300,316,348,379,400,500,600,712}
def wrapped(images_chunk,*args,**kwargs):
 global offset
 result=original(images_chunk,*args,**kwargs)
 n=images_chunk.shape[1]
 record=dict(start=offset,end=offset+n-1,metric_scale=float(model._metric_scale),normalized_focal=float(model._da3_focal))
 records.append(record);print('CHUNK',record,flush=True)
 for t in selected:
  if offset<=t<offset+n:sample[f'depth_{t}']=result[4][t-offset,2].detach().cpu().float().numpy()
 offset+=n
 return result
model.forward_point=wrapped
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float32):
 features=model.get_fmaps(images,1,len(images),None,False)
del features,images
(B/'chunk_metadata.json').write_text(json.dumps(records,indent=2))
np.savez_compressed(B/'chunk_normalized_depth.npz',**sample)
print('DONE chunk metadata; no change to native output',flush=True)
