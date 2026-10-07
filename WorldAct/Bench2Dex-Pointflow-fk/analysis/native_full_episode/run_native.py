"""Call native pipeline unchanged, adapting paths and Bench2Dex scene prompts only."""
import sys,os,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
NATIVE=Path('/mnt/afs/WorldAct-pointflow-native')
REPO=NATIVE/'Track4World_portable/Track4World'
sys.path[:0]=[str(NATIVE/'pf_out'),str(REPO/'visualization'),str(REPO),str(REPO.parent),'/data/shichaojian/pylibs_sam2']
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
stage=sys.argv[1]
import numpy as np
import torch
np.random.seed(17);torch.manual_seed(17);torch.set_num_threads(8)
EP='task21_episode000000'
if stage=='export':
 import run_efep_export as m
 m.ROOT=HERE/'local_assets';m.REPO=REPO
 m.rde.ROOT=m.ROOT;m.rde.REPO=REPO
 sys.argv=[str(NATIVE/'pf_out/run_efep_export.py'),'--episode',EP,'--source-root',str(HERE/'source'),'--run-dir',str(HERE/'run')]
 provenance=dict(native_repo=str(NATIVE),stage=stage,seed=17,external_chunks=0,max_frames=0,
   source_files_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [NATIVE/'pf_out/run_efep_export.py',REPO/'demo.py',REPO/'track4world/nets/model.py']},
   adaptations=['Legacy ROOT and REPO mapped to installed local assets; native functions unchanged.'])
 (HERE/'export_provenance.json').write_text(json.dumps(provenance,indent=2))
 m.main()
elif stage in ['segment','segment_refined']:
 import sam2_segment as m
 m.CKPT=Path('/data/shichaojian/checkpoints/sam2.1_hiera_large.pt')
 m.DINO=Path('/data/shichaojian/checkpoints/grounding-dino-base')
 m.HAND_TEXT='white robotic hand.'
 original=m.dino_hand_boxes
 def detect(rgb,n,threshold=.25,text=None):
  return original(rgb,n,threshold=threshold,text=text or m.HAND_TEXT)
 m.dino_hand_boxes=detect
 def box(x0,y0,x1,y1):return [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
 m.DATASETS['bench2dex_task21']=dict(query_frame=0,
   hand_rois=[box(0,0,132,166),box(504,0,640,166)],
   object_groups=[dict(label=3,rois=[box(111,105,175,165),box(169,179,229,228),box(331,108,365,146)]),
                  dict(label=4,rois=[box(505,84,640,244)])])
 if stage=='segment_refined':
  original_prompts=m.prompts
  def refined_prompts(cfg,w,h,rgb,sources,arm):
   prompts=original_prompts(cfg,w,h,rgb,sources,arm)
   result=[]
   for oid,cls,b,pts in prompts:
    if cls==2:
     b=b.copy();b[3]=min(b[3],166.)
     for source in sources:
      if source['obj_id']==oid:source['refinement']='anchor hand box bottom clipped to y=166; excludes wrist and arm'
    result.append((oid,cls,b,pts))
   return result
  m.prompts=refined_prompts
 (HERE/('segmentation_adaptation_'+stage+'.json')).write_text(json.dumps(dict(hand_text=m.HAND_TEXT,config=m.DATASETS['bench2dex_task21'],note='Bench2Dex white hands and task21 objects replace real-data prompts; native DINO/SAM2 propagation unchanged.'),indent=2))
 sys.argv=[str(NATIVE/'pf_out/sam2_segment.py'),'--dataset','bench2dex_task21','--episode-dir',str(HERE/'source'/EP),'--out-dir',str(HERE/'run/sam2_masks'/EP)]
 m.main()
else:raise ValueError(stage)
