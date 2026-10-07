"""Compare DA3 to known table plane and FK-posed visual hand surfaces.

Table scale is an oracle diagnostic, not a claim of an online calibration method.
Robot surface comparison uses frame 0 and conservative unoccluded hand regions.
"""
from pathlib import Path
import json
import numpy as np
import h5py
import trimesh
import cv2
import os
OUT=Path(__file__).resolve().parent
suffix='_conditioned' if os.environ.get('PROBE_CONDITIONED')=='1' else ''
C=np.array([[0,-1,0],[0,0,-1],[1,0,0.]])

def stat(v):
    a=np.asarray(v)
    return {'count':int(a.size),'mean':float(np.mean(a)),'median':float(np.median(a)),'p95':float(np.percentile(a,95)),'max':float(np.max(a))}

with h5py.File('/tmp/bench2dex_replay21_ep0.hdf5') as f:
    meta=json.loads(f['meta/scene_generalization_sample'][()])
table_z=.75+meta['spatial']['table_height']['height_offset_m']
report={'table_z_world_m':table_z,'table_patch_original_pixels':{'u':[260,420],'v':[250,300]},'frames':[],'limitations':['Four single-frame depth-backbone probes only; not full Track4World tracking.','Table scale uses known scene geometry as an offline oracle diagnostic.','Hand truth is FK-posed visual mesh ray intersection, not joint-center depth.','Frame0 hand mask is conservative and qualitatively inspected; robot RGB time alignment has not been numerically measured.']}
s0=None
for idx in [0,200,400,600]:
    a=np.load(OUT/f'da3_frame_{idx:04d}{suffix}.npz');d=a['depth'];K=a['intrinsic_true'];Kd=a['intrinsic_pred'];E=a['extrinsic_world_from_cam'];h,w=d.shape
    u,v=np.meshgrid(np.arange(w),np.arange(h));uv=np.stack([u,v,np.ones_like(u)],-1)
    rays=uv@np.linalg.inv(K).T
    rays_world=rays@(E[:3,:3]@C.T).T
    ztrue=(table_z-E[2,3])/rays_world[:,:,2]
    mask=(u>=260*w/640)&(u<420*w/640)&(v>=250*h/480)&(v<300*h/480)&np.isfinite(d)&(d>0)
    s_table=float(np.median(ztrue[mask]/d[mask]));s_k=float(np.mean([K[0,0],K[1,1]])/np.mean([Kd[0,0],Kd[1,1]]))
    if s0 is None:s0=s_table
    modes={'raw':1.,'focal_corrected':s_k,'oracle_table_scale_frame0':s0,'oracle_table_scale_per_frame':s_table}
    r={'frame':idx,'focal_true_pixels':float(K[0,0]),'focal_pred_pixels':float(Kd[0,0]),'pred_principal_point':Kd[:2,2].tolist(),'true_principal_point':K[:2,2].tolist(),'true_patch_depth_m':stat(ztrue[mask]),'raw_patch_depth_m':stat(d[mask]),'focal_scale':s_k,'oracle_table_scale':s_table,'table_abs_depth_error_mm':{k:stat(np.abs(d[mask]*s-ztrue[mask])*1000) for k,s in modes.items()}}
    if idx==0:
        triangles=np.load(OUT/'frame0_hand_visual_triangles.npz')['triangles_world'].astype(float)
        mesh=trimesh.Trimesh(vertices=triangles.reshape(-1,3),faces=np.arange(triangles.size//3).reshape(-1,3),process=False)
        # Both hands are open at episode start; exclude central objects and arm/wrist.
        eligible=(((u<110*w/640)|(u>510*w/640))&(v<185*h/480)&(v>8*h/480)&(u%3==0)&(v%3==0))
        vv,uu=np.nonzero(eligible);dirs=rays_world[vv,uu];origins=np.tile(E[:3,3],(len(dirs),1))
        print('Raycast',len(dirs),'rays',len(triangles),'triangles',flush=True)
        locations,ray_ids,_=mesh.ray.intersects_location(origins,dirs,multiple_hits=False)
        pc=(locations-E[:3,3])@E[:3,:3]@C.T
        vz=pc[:,2];vv=vv[ray_ids];uu=uu[ray_ids];zd=d[vv,uu]
        valid=np.isfinite(zd)&(zd>0)&(vz>0);pc=pc[valid];vz=vz[valid];zd=zd[valid];vv=vv[valid];uu=uu[valid]
        pix=np.stack([uu,vv,np.ones_like(uu)],1)
        p_raw=(pix@np.linalg.inv(Kd).T)*zd[:,None]
        p_true_ray=pix@np.linalg.inv(K).T
        hand_modes={'raw_predicted_intrinsics':p_raw,'true_intrinsics_only':p_true_ray*zd[:,None],'true_intrinsics_and_focal_scale':p_true_ray*(zd*s_k)[:,None],'true_intrinsics_and_oracle_table_scale':p_true_ray*(zd*s0)[:,None]}
        r['hand_surface']={'ray_count':int(len(vz)),'true_depth_m':stat(vz),'raw_depth_m':stat(zd),'oracle_hand_scale':float(np.median(vz/zd)),'xyz_error_mm':{k:stat(np.linalg.norm(p-pc,axis=1)*1000) for k,p in hand_modes.items()},'abs_depth_error_mm':{k:stat(np.abs(zd*s-vz)*1000) for k,s in modes.items()}}
        im=cv2.cvtColor(a['rgb'],cv2.COLOR_RGB2BGR)
        for xx,yy in zip(uu,vv):cv2.circle(im,(int(xx),int(yy)),1,(0,220,255),-1)
        cv2.rectangle(im,(int(260*w/640),int(250*h/480)),(int(420*w/640),int(300*h/480)),(255,0,255),1)
        cv2.imwrite(str(OUT/f'da3_depth_probe_regions{suffix}.jpg'),im)
        np.savez_compressed(OUT/f'da3_hand_depth_comparison{suffix}.npz',u=uu,v=vv,true_xyz=pc,raw_depth=zd,raw_xyz=p_raw,true_ray_xyz=p_true_ray*zd[:,None])
    report['frames'].append(r)
    print('frame',idx,'s_table',s_table,'s_focal',s_k,'raw table error',r['table_abs_depth_error_mm']['raw']['median'],flush=True)
(OUT/f'da3_metric_audit{suffix}.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report['frames'][0].get('hand_surface'),indent=2),flush=True)
