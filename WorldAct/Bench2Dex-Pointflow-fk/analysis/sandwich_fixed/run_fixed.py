"""Run native exporter with isolated DA3 metadata fix and diagnostic snapshots."""
import os,sys,json,hashlib
from pathlib import Path
B=Path(__file__).resolve().parent;BASE=B.parent/'native_full_episode';N=Path('/mnt/afs/WorldAct-pointflow-native');R=B.parent/'da3_chunk_fix/Track4World'
sys.path[:0]=[str(R),str(B.parent/'da3_chunk_fix'),str(N/'pf_out')]
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np,torch
np.random.seed(17);torch.manual_seed(17);torch.set_num_threads(8)
import demo,run_efep_export as native
assert Path(demo.__file__).resolve()==R/'demo.py'
native.ROOT=BASE/'local_assets';native.REPO=R;native.rde.ROOT=native.ROOT;native.rde.REPO=R
(B/'snapshots').mkdir(exist_ok=True)
load=demo.load_model
selected={0,63,100,126,127,128,129,200,254,255,256,257,300,382,383,384,385,400,500,510,511,512,513,600,638,639,640,641,650,766,767,768,769,800,894,895,896,897,1000,1022,1023,1024,1025,1150,1151,1152,1153,1191}
def load_instrumented(args,config):
 model=load(args,config);infer=model.infer_pair
 def wrapped(*a,**kw):
  output=infer(*a,**kw);g,m=output
  for key in ['frame_metric_scales','frame_normalized_focals','frame_chunk_ids']:
   np.save(B/(key+'.npy'),g[key].detach().cpu().numpy())
  for t in sorted(selected):
   data=dict(points=g['points'][0,t].float().cpu().numpy(),mask=g['mask'][0,t].cpu().numpy(),intrinsics=g['intrinsics'][0,t].float().cpu().numpy())
   if t<g['points'].shape[1]-1:
    data.update(flow_2d=m['flow_2d'][0,t].float().cpu().numpy(),flow_3d=m['flow_3d'][0,t].float().cpu().numpy(),visconf=m['visconf_maps_e'][0,t].float().cpu().numpy(),target_intrinsics=g['intrinsics'][0,t+1].float().cpu().numpy())
   np.savez_compressed(B/'snapshots'/f'frame{t:04d}.npz',**data)
  return output
 model.infer_pair=wrapped
 return model
demo.load_model=load_instrumented
provenance=dict(native_exporter=str(N/'pf_out/run_efep_export.py'),isolated_model=str(R/'track4world/nets/model.py'),model_sha256=hashlib.sha256((R/'track4world/nets/model.py').read_bytes()).hexdigest(),seed=17,mask_source=str(B/'run/sam2_masks/episode_0013_20260731_133649'),weights_changed=False,external_chunks=0)
(B/'provenance.json').write_text(json.dumps(provenance,indent=2))
sys.argv=['run_efep_export.py','--episode','episode_0013_20260731_133649','--source-root','/data/shichaojian/raw_data/singlerighthand_sandwich_100','--run-dir',str(B/'run'),'--store-device','cpu']
native.main()
