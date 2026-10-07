import json
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent;REF=B.parent/'sandwich_real_audit';EP='episode_0013_20260731_133649';D=B/'run/efep_seg_fixed'/EP
arrays={k:np.load(D/(k+'.npy'),mmap_mode='r') for k in ['frame_offsets','obs_pos','obs_uv','obs_label','obs_valid','obs_unique']}
def stats(x):
 x=np.asarray(x);return dict(n=len(x),median=float(np.median(x)),p95=float(np.percentile(x,95)),mean=float(np.mean(x))) if len(x) else {'n':0}
rows=[];oldall=[];newall=[]
for f in [0,128,300,500,650,800,1000,1191]:
 x=np.load(REF/f'mesh_surface_{f:04d}.npz');s,e=map(int,arrays['frame_offsets'][f:f+2]);ok=(arrays['obs_label'][s:e]==2)&arrays['obs_valid'][s:e]&arrays['obs_unique'][s:e];uv=arrays['obs_uv'][s:e][ok];p=arrays['obs_pos'][s:e][ok]
 grid=np.full((448,640,3),np.nan,np.float32);grid[uv[:,1],uv[:,0]]=p;new=grid[x['uv'][:,1],x['uv'][:,0]];valid=np.isfinite(new).all(1);new=new[valid];old=x['pf'][valid];truth=x['surface'][valid];pix=x['uv'][valid]
 a=np.linalg.norm(old-truth,axis=1)*1000;b=np.linalg.norm(new-truth,axis=1)*1000
 rows.append(dict(frame=f,reference_points=len(valid),matched=int(valid.sum()),old_xyz_mm=stats(a),fixed_xyz_mm=stats(b),old_signed_z_mm=stats((old[:,2]-truth[:,2])*1000),fixed_signed_z_mm=stats((new[:,2]-truth[:,2])*1000)))
 oldall.extend(a);newall.extend(b);np.savez_compressed(B/f'comparison_{f:04d}.npz',uv=pix,old=old,fixed=new,surface=truth,fk=x['fk'])
 snap=np.load(B/'snapshots'/f'frame{f:04d}.npz');assert np.max(np.abs(snap['points'][pix[:,1],pix[:,0]]-new))<1e-6
 print(rows[-1],flush=True)
report=dict(episode=EP,frames=1192,comparison_frames=rows,pooled_same_pixel_old_xyz_mm=stats(oldall),pooled_same_pixel_fixed_xyz_mm=stats(newall),note='Historical 9.24 v2 versus current isolated fixed pipeline. Identical pixel intersections, geometry reference, extrinsic and meshes; not a contemporaneous unmodified rerun. v2 masks reused before point filtering; original v2 applied arm cut after filtering. No fitted scale or rigid alignment.')
(B/'comparison.json').write_text(json.dumps(report,indent=2))
scales=np.load(B/'frame_metric_scales.npy').reshape(-1);focal=np.load(B/'frame_normalized_focals.npy').reshape(-1);ids=np.load(B/'frame_chunk_ids.npy').reshape(-1)
meta=[dict(start=int(np.flatnonzero(ids==i)[0]),end=int(np.flatnonzero(ids==i)[-1]),scale=float(scales[ids==i][0]),focal=float(focal[ids==i][0])) for i in np.unique(ids)]
(B/'chunk_metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps(report,indent=2))
