"""Independent CPU invariants and boundary diagnostics for the fixed GPU run."""
from pathlib import Path
import json,numpy as np
import cv2
B=Path(__file__).resolve().parent;OLD=B.parent/'native_full_episode'
s=np.load(B/'frame_metric_scales.npy');f=np.load(B/'frame_normalized_focals.npy');ids=np.load(B/'frame_chunk_ids.npy')
ref=json.loads((OLD/'chunk_metadata.json').read_text());depth=np.load(OLD/'chunk_normalized_depth.npz')
raw=OLD/'run/efep_raw/task21_episode000000'
ro=np.load(raw/'frame_offsets.npy',mmap_mode='r');ru=np.load(raw/'obs_uv.npy',mmap_mode='r');rp=np.load(raw/'obs_pos.npy',mmap_mode='r')
h,w=448,640;yy,xx=np.indices((h,w));uv=np.stack([(xx+.5)/w,(yy+.5)/h,np.ones((h,w))],-1)
def st(x):
 x=np.asarray(x);x=x[np.isfinite(x)]
 return dict(n=int(x.size),median=float(np.median(x)),p95=float(np.percentile(x,95)),max=float(np.max(x))) if len(x) else dict(n=0)
def expectedK(focal):
 fp=focal*.5*np.hypot(h,w)
 return np.array([[fp/w,0,.5],[0,fp/h,.5],[0,0,1.]])
report={'invariants':[],'boundary_pairs':[],'per_frame_geometry':[]}
assert len(s)==713 and len(f)==713 and len(ids)==713
for c in ref:
 sl=slice(c['start'],c['end']+1)
 # Independent previous probe establishes the unchanged backbone output.
 assert np.allclose(s[sl],c['metric_scale'],rtol=1e-5,atol=1e-6)
 assert np.allclose(f[sl],c['normalized_focal'],rtol=1e-5,atol=1e-6)
report['invariants'].append('All 713 frame scales and focals match independent unmodified-backbone probe.')
for t in sorted(int(k.split('_')[1]) for k in depth.files):
 a=np.load(B/'snapshots'/f'frame{t:04d}.npz');p=a['points'];K=a['intrinsics'];trueK=expectedK(f[t])
 assert np.allclose(K,trueK,rtol=1e-5,atol=1e-6)
 z=depth[f'depth_{t}']*s[t]
 xyz=(uv@np.linalg.inv(trueK).T)*z[...,None]
 err=np.linalg.norm(p-xyz,axis=-1)*1000
 report['per_frame_geometry'].append(dict(frame=t,independent_xyz_error_mm=st(err)))
 assert np.nanmax(err)<2., (t,float(np.nanmax(err)))
report['invariants'].append('Sampled XYZ matches independent own-chunk depth/scale/focal reconstruction within 2 mm.')
# Endpoint projection must use TARGET frame K at every chunk boundary.
for boundary in [128,256,384,512,640]:
 a=np.load(B/'snapshots'/f'frame{boundary-1:04d}.npz');p=a['flow_3d'];K=a['target_intrinsics']
 norm=p@K.T;pix=norm[...,:2]/norm[...,2:]*np.array([w,h]);flow=a['flow_2d'].transpose(1,2,0)
 good=np.isfinite(p).all(-1)&(p[...,2]>0)&np.isfinite(flow).all(-1)
 delta=np.linalg.norm(pix-flow,axis=-1)[good]
 assert np.percentile(delta,99)<1., (boundary,np.percentile(delta,99))
 src=np.load(B/'snapshots'/f'frame{boundary-1:04d}.npz')['points'];dst=np.load(B/'snapshots'/f'frame{boundary:04d}.npz')['points']
 # Central lower table ROI: a temporal diagnostic, not independent geometry truth.
 region=(xx>=260)&(xx<320)&(yy>=300*h/480)&(yy<320*h/480)
 def old_points(t):
  lo,hi=map(int,ro[t:t+2]);xy=ru[lo:hi];dense=np.full((h,w,3),np.nan,np.float32);dense[xy[:,1],xy[:,0]]=rp[lo:hi];return dense
 oldsrc,olddst=old_points(boundary-1),old_points(boundary)
 conf=a['visconf'][0]*a['visconf'][1];mx=flow[...,0].astype(np.float32);my=flow[...,1].astype(np.float32)
 target=cv2.remap(dst,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=float('nan'))
 valid=good&(conf>.3)&np.isfinite(target).all(-1)
 report['boundary_pairs'].append(dict(source=boundary-1,target=boundary,target_K_reprojection_px=st(delta),
  static_roi_native_jump_mm=st(np.linalg.norm(olddst-oldsrc,axis=-1)[region]*1000),
  static_roi_fixed_jump_mm=st(np.linalg.norm(dst-src,axis=-1)[region]*1000),
  endpoint_vs_next_depth_xyz_mm=st(np.linalg.norm(p-target,axis=-1)[valid]*1000)))
report['invariants'].append('All 5 cross-chunk pair endpoints project with the target frame K (P99 <1 px).')
report['limitations']=['World points/poses remain in each DA3 block local world frame; only camera-space export is audited.',
 'Static ROI jumps and endpoint-vs-next-depth are consistency diagnostics, not independent ground truth.',
 'Passing unit invariants does not remove DA3 estimation errors or SAM2 missing hand coverage.']
(B/'output_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
