"""Verify stale chunk metadata against saved native outputs; diagnostic correction only."""
from pathlib import Path
import json,numpy as np
B=Path(__file__).resolve().parent;O=B/'validation'
chunks=json.loads((B/'chunk_metadata.json').read_text());z=np.load(B/'chunk_normalized_depth.npz')
last=chunks[-1];D=B/'run/efep_raw/task21_episode000000'
o=np.load(D/'frame_offsets.npy',mmap_mode='r');uv=np.load(D/'obs_uv.npy',mmap_mode='r');pos=np.load(D/'obs_pos.npy',mmap_mode='r');K=np.load(D/'intrinsics.npy')
def stat(x):
 x=np.asarray(x);x=x[np.isfinite(x)]
 return dict(n=len(x),median=float(np.median(x)),p95=float(np.percentile(x,95)),max=float(np.max(x))) if len(x) else dict(n=0)
report=dict(chunks=chunks,source_issue='get_fmaps overwrites model._metric_scale and model._da3_focal on each 128-frame block; infer_pair reads only the last values for all frames.',
 correction='Diagnostic only: XYZ *= chunk_scale/last_scale, then XY *= last_focal/chunk_focal. Does not rerun motion estimation, change native final outputs, or fit to FK.',samples=[])
old_all=[];new_all=[]
for path in sorted(O.glob('surface_*.npz')):
 t=int(path.stem.split('_')[-1]);a=np.load(path);c=next(c for c in chunks if c['start']<=t<=c['end'])
 pred=a['pred_xyz'];truth=a['mesh_xyz'];u=a['u'];v=a['v'];interior=a['interior']
 corrected=pred.astype(float)*(c['metric_scale']/last['metric_scale']);corrected[:,:2]*=last['normalized_focal']/c['normalized_focal']
 e0=np.linalg.norm(pred-truth,axis=1)*1000;e1=np.linalg.norm(corrected-truth,axis=1)*1000
 # Direct check: native Z should equal saved normalized DA3 Z times LAST block scale.
 expected=z[f'depth_{t}'][v,u]*last['metric_scale'];delta=np.abs(pred[:,2]-expected)*1000
 item=dict(frame=t,chunk_start=c['start'],native_z_vs_last_scale_reconstruction_mm=stat(delta),native_xyz_mm=stat(e0),chunk_metadata_corrected_xyz_mm=stat(e1),corrected_signed_depth_mm=stat((corrected[:,2]-truth[:,2])*1000))
 report['samples'].append(item);old_all.extend(e0[interior]);new_all.extend(e1[interior])
 np.savez_compressed(O/f'chunk_corrected_{t:04d}.npz',u=u,v=v,mesh_xyz=truth,native_xyz=pred,corrected_xyz=corrected)
report['pooled_interior_native_xyz_mm']=stat(old_all);report['pooled_interior_chunk_corrected_xyz_mm']=stat(new_all)
report['native_exported_focal_pixels']=float(K[0,0,0]*640)
report['last_chunk_focal_pixels']=float(last['normalized_focal']*.5*np.hypot(448,640))
report['native_K_same_all_frames']=bool(np.array_equal(K,np.broadcast_to(K[0],K.shape)))
(O/'chunk_recovery_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
