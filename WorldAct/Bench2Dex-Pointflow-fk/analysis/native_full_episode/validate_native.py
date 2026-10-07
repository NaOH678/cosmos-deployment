"""Audit final native pipeline hand points against FK-posed visual mesh rays."""
from pathlib import Path
import json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/bench2dex_mpl')
import numpy as np
import cv2,trimesh,h5py
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parent;EP='task21_episode000000'
D=B/'run/efep_seg_v61'/EP;R=B/'run/efep_raw'/EP
O=B/'validation';O.mkdir(exist_ok=True)
g=np.load(B/'full_geometry.npz');h,w=448,640
K=g['intrinsic'].copy();K[1]*=h/480;K[1,2]=(g['intrinsic'][1,2]+.5)*h/480-.5
C=g['optical_from_camera_body']
def st(x):
 x=np.asarray(x);x=x[np.isfinite(x)]
 return dict(n=int(x.size),median=float(np.median(x)),mean=float(np.mean(x)),p95=float(np.percentile(x,95))) if len(x) else dict(n=0)
def arrays(d,names):return {n:np.load(d/(n+'.npy'),mmap_mode='r') for n in names}
data=arrays(D,['frame_offsets','obs_uv','obs_pos','obs_label','obs_valid','obs_unique','intrinsics'])
raw=arrays(R,['frame_offsets','obs_uv','obs_pos','obs_clean','obs_conf'])
links=g['link_names'].tolist();hi=[i for i,n in enumerate(links) if 'finger' in n or 'palm' in n]
report=dict(pipeline='native full episode, EFEP + DINO/SAM2 + fill gaps3 + despike0.03 + drop_detached',
 frames=713,seconds=35.65,external_chunks=0,fitted_scale=False,fitted_extrinsics=False,
 limitations=['Single episode and selected frames only; no Isaac depth pass.',
 'Reference is FK-posed hand visual mesh, not sensor depth; hidden hand surfaces can be occluded by non-hand objects.',
 'Same-frame RGB/qpos convention; wrist-camera lag does not establish RGB lag.',
 'SAM2 prompts adapted for white Bench2Dex hands; check preview quality.'],samples=[],per_frame=[])
# Full episode per-frame proximity is diagnostic, not a corresponding-surface error.
for t in range(713):
 s,e=map(int,data['frame_offsets'][t:t+2]);lab=data['obs_label'][s:e]
 keep=(lab==2)&data['obs_valid'][s:e]&data['obs_unique'][s:e]
 p=np.asarray(data['obs_pos'][s:e])[keep];p=p[np.isfinite(p).all(1)]
 fk=g['fk_optical_m'][t,hi]
 near=cKDTree(p).query(fk)[0]*1000 if len(p) else []
 report['per_frame'].append(dict(frame=t,unique_valid_hand_points=len(p),fk_nearest_hand_cloud_mm=st(near)))
 print('proximity',t,flush=True) if t%200==0 else None
frames=[0,63,100,200,300,316,348,379,400,500,600,712]
all_errors=[]
with h5py.File('/tmp/bench2dex_replay21_ep0.hdf5') as f:
 for t in frames:
  s,e=map(int,data['frame_offsets'][t:t+2]);uv=np.asarray(data['obs_uv'][s:e]);p=np.asarray(data['obs_pos'][s:e]);lab=data['obs_label'][s:e]
  keep=(lab==2)&data['obs_valid'][s:e]&data['obs_unique'][s:e]&np.isfinite(p).all(1)
  dense=np.full((h,w,3),np.nan,np.float32);dense[uv[keep,1],uv[keep,0]]=p[keep]
  hand=np.isfinite(dense).all(-1)
  # Grid reduces CPU raycast cost; all retained points still come from the final native output.
  v,u=np.indices((h,w));pick=hand&(u%4==0)&(v%4==0)
  vv,uu=np.nonzero(pick);pix=np.stack([uu,vv,np.ones_like(uu)],-1);rays=pix@np.linalg.inv(K).T
  E=g['extrinsic_world_from_camera_body'][t];dirs=rays@(E[:3,:3]@C.T).T
  triangles=np.load(B/'meshes'/f'frame{t}.npz')['triangles_world'].astype(float)
  mesh=trimesh.Trimesh(vertices=triangles.reshape(-1,3),faces=np.arange(triangles.size//3).reshape(-1,3),process=False)
  if len(uu):loc,rid,_=mesh.ray.intersects_location(np.tile(E[:3,3],(len(dirs),1)),dirs,multiple_hits=False)
  else:loc=np.zeros((0,3));rid=np.zeros(0,int)
  truth=(loc-E[:3,3])@E[:3,:3]@C.T;uu=uu[rid];vv=vv[rid];pred=dense[vv,uu]
  err=np.linalg.norm(pred-truth,axis=-1)*1000;dz=(pred[:,2]-truth[:,2])*1000
  interior=cv2.erode(hand.astype(np.uint8),np.ones((5,5),np.uint8))>0
  ins=interior[vv,uu];all_errors.extend(err[ins].tolist())
  sample=dict(frame=t,sampled_final_hand_pixels=int(pick.sum()),mesh_intersection_count=len(err),
    xyz_error_mm=st(err),signed_depth_error_mm=st(dz),interior_xyz_error_mm=st(err[ins]),
    interior_signed_depth_error_mm=st(dz[ins]),left_image_xyz_error_mm=st(err[uu<320]),right_image_xyz_error_mm=st(err[uu>=320]))
  # Raw values at exactly the same surviving pixels distinguish filtering from changed geometry.
  rs,re=map(int,raw['frame_offsets'][t:t+2]);ru=raw['obs_uv'][rs:re];rp=raw['obs_pos'][rs:re]
  rd=np.full((h,w,3),np.nan,np.float32);rd[ru[:,1],ru[:,0]]=rp
  sample['raw_vs_final_same_pixel_max_m']=float(np.nanmax(np.abs(rd[vv,uu]-pred))) if len(err) else None
  np.savez_compressed(O/f'surface_{t:04d}.npz',u=uu,v=vv,pred_xyz=pred,mesh_xyz=truth,interior=ins)
  im=cv2.resize(cv2.imdecode(f['cameras/cam_overhead/rgb'][t],cv2.IMREAD_COLOR),(w,h))
  for x,y,ee in zip(uu,vv,err):cv2.circle(im,(int(x),int(y)),1,(0,200,0) if ee<30 else (0,180,255) if ee<100 else (0,0,255),-1)
  cv2.putText(im,f'frame {t} median {np.median(err):.1f}mm | n={len(err)}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),2)
  cv2.imwrite(str(O/f'overlay_{t:04d}.jpg'),im)
  report['samples'].append(sample);print(json.dumps(sample),flush=True)
report['pooled_interior_xyz_error_mm']=st(all_errors)
report['frame_median_fk_nearest_mm']=st([r['fk_nearest_hand_cloud_mm'].get('median',np.nan) for r in report['per_frame']])
report['hand_points_per_frame']=st([r['unique_valid_hand_points'] for r in report['per_frame']])
(O/'report.json').write_text(json.dumps(report,indent=2))
fig,axs=plt.subplots(1,2,figsize=(13,4))
axs[0].plot([r['frame'] for r in report['per_frame']],[r['fk_nearest_hand_cloud_mm'].get('median',np.nan) for r in report['per_frame']]);axs[0].set(title='FK nearest final hand cloud (proximity only)',xlabel='Frame',ylabel='Median distance (mm)')
axs[1].plot(frames,[r['xyz_error_mm'].get('median',np.nan) for r in report['samples']],'-o',label='All sampled hand pixels')
axs[1].plot(frames,[r['interior_xyz_error_mm'].get('median',np.nan) for r in report['samples']],'-o',label='Interior pixels');axs[1].set(title='Same-pixel FK mesh vs final PointFlow',xlabel='Frame',ylabel='Median XYZ error (mm)');axs[1].legend()
fig.tight_layout();fig.savefig(O/'error_curves.png',dpi=160)
ims=[cv2.resize(cv2.imread(str(O/f'overlay_{t:04d}.jpg')),(480,336)) for t in frames]
cv2.imwrite(str(O/'surface_contact_sheet.jpg'),np.concatenate([np.concatenate(ims[i:i+3],axis=1) for i in range(0,len(ims),3)],axis=0))
print('DONE',O,flush=True)
