"""Single-frame DA3 geometry diagnostic using the local Track4World backbone weights.

This is a depth-backbone probe, not a full Track4World trajectory evaluation.
"""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import sys
from pathlib import Path
sys.path.insert(0,'/data/shichaojian/Track4World_portable/Track4World/track4world/nets/external')
import numpy as np
import cv2
import h5py
import torch
import json
from depth_anything_3.api import DepthAnything3

torch.manual_seed(17)
torch.set_num_threads(4)
OUT=Path(__file__).resolve().parent
conditioned=os.environ.get('PROBE_CONDITIONED')=='1'
suffix='_conditioned' if conditioned else ''
print('Loading nested DA3 pretrained metric branch',flush=True)
model=DepthAnything3.from_pretrained('/data/shichaojian/checkpoints/DA3NESTED-GIANT-LARGE-1.1')
state=torch.load('/data/shichaojian/checkpoints/track4world_da3.pth',map_location='cpu',mmap=True,weights_only=True)
remapped={}
for k,v in state.items():
    if k.startswith('backbone.model.'):
        k=k[len('backbone.'):]
        if not k.startswith(('model.da3.','model.da3_metric.')):k=k.replace('model.','model.da3.',1)
        remapped[k]=v
missing,unexpected=model.load_state_dict(remapped,strict=False)
print('Track4World backbone tensors',len(remapped),'unexpected',len(unexpected),'nonmetric_missing',len([k for k in missing if not k.startswith('model.da3_metric.')]),flush=True)
model=model.to('cuda').eval()
del state,remapped
f=h5py.File('/tmp/bench2dex_replay21_ep0.hdf5')
report={'probe':'single-frame nested DA3 with Track4World anyview weights; no tracker refinement','conditioned_on_true_intrinsics':conditioned,'checkpoint':'/data/shichaojian/checkpoints/track4world_da3.pth','seed':17,'nonmetric_missing_keys':[k for k in missing if not k.startswith('model.da3_metric.')],'unexpected_keys':unexpected,'frames':[]}
for idx in [0,200,400,600]:
    cid='cam_overhead';cam=f['cameras'][cid]
    bgr=cv2.imdecode(cam['rgb'][idx],cv2.IMREAD_COLOR)
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    print('Infer',idx,flush=True)
    with torch.inference_mode():
        if conditioned:
            # Avoid the public API's degenerate single-view Umeyama scale fit.
            imgs_cpu,ext,ins=model._preprocess_inputs([rgb],np.eye(4,dtype=np.float32)[None],cam['intrinsic'][:][None],504,'upper_bound_resize')
            imgs,ext,ins=model._prepare_model_inputs(imgs_cpu,ext,ins)
            raw=model.inference_v2(imgs,ex_t=ext,in_t=ins)
            prediction=model._add_processed_images(model._convert_to_prediction(raw),imgs_cpu)
        else:
            prediction=model.inference([rgb],process_res=504,process_res_method='upper_bound_resize')
    d=prediction.depth[0].astype(np.float32);Kpred=prediction.intrinsics[0].astype(float)
    # Input processor provides image-space transformed supplied calibration.
    _,_,ktrue=model._preprocess_inputs([rgb],None,cam['intrinsic'][:][None],504,'upper_bound_resize')
    Ktrue=np.asarray(ktrue[0],float)
    np.savez_compressed(OUT/f'da3_frame_{idx:04d}{suffix}.npz',depth=d,intrinsic_pred=Kpred,intrinsic_true=Ktrue,extrinsic_world_from_cam=cam['extrinsic_world_from_cam'][idx],rgb=prediction.processed_images[0],frame=idx)
    report['frames'].append({'frame':idx,'shape':list(d.shape),'intrinsic_pred':Kpred.tolist(),'intrinsic_true':Ktrue.tolist(),'median_depth_m':float(np.median(d))})
    print(report['frames'][-1],flush=True)
(OUT/f'da3_probe{suffix}.json').write_text(json.dumps(report,indent=2))
print('Saved DA3 probe',flush=True)
