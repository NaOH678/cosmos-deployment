"""CPU geometry audit of full EFEP outputs; no scale or rigid fitting."""
from pathlib import Path
import json
import numpy as np
import cv2
import h5py
import trimesh
from scipy.spatial import cKDTree
BASE=Path(__file__).resolve().parent
OUT=BASE/'pointflow_validation'
C=np.array([[0.,-1,0],[0,0,-1],[1,0,0]])
def stat(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return dict(n=int(x.size),median=float(np.median(x)),p95=float(np.percentile(x,95)),mean=float(np.mean(x))) if x.size else dict(n=0)
with h5py.File('/tmp/bench2dex_replay21_ep0.hdf5') as f:
    sample=json.loads(f['meta/scene_generalization_sample'][()])
table_z=.75+sample['spatial']['table_height']['height_offset_m']
report={'method':'Raw camera-space points compared to calibrated FK/visual surfaces; no fitted scale or extrinsics.',
 'limitations':['Two clips from one episode, default Track4World backbone; historical backbone switch unverified.',
 'Mesh checks use two selected frames, hand meshes only; non-hand occlusion is not automatically resolved.',
 'Table patch must be visually checked for occlusion.',
 'FK nearest-cloud distance is a proximity diagnostic, not point correspondence accuracy.',
 'RGB lag remains unresolved; nominal same-frame FK is used.'], 'clips':{}}
for tag,probe in [('start',0),('motion',32)]:
    d=OUT/tag;g=np.load(d/'geometry.npz');run=d/'gpu_track4world_backbone'
    pts=np.load(run/'points.npy',mmap_mode='r');mask=np.load(run/'masks.npy',mmap_mode='r')
    meta=json.loads((run/'complete.json').read_text())
    h,w=pts.shape[1:3];u,v=np.meshgrid(np.arange(w),np.arange(h))
    K=g['intrinsic_original'].copy();K[1]*=h/480
    # cv2 resize pixel centers: original v=(v_new+.5)*480/h-.5.
    K[1,2]=(g['intrinsic_original'][1,2]+.5)*h/480-.5
    rays=np.stack([u,v,np.ones_like(u)],-1)@np.linalg.inv(K).T
    region=(u>=260)&(u<420)&(v>=250*h/480)&(v<300*h/480)
    links=g['link_names'].tolist();hand=[i for i,n in enumerate(links) if 'finger' in n or 'palm' in n]
    frames=[]
    for t,raw in enumerate(g['frame_indices']):
        E=g['extrinsic_world_from_camera_body'][t]
        rw=rays@(E[:3,:3]@C.T).T
        truez=(table_z-E[2,3])/rw[...,2]
        depth=np.asarray(pts[t,:,:,2]);ok=region & mask[t].astype(bool)&np.isfinite(depth)&(depth>0)
        fk=g['fk_optical_candidates_m'][1,t,hand]
        uv=fk@K.T;uv=uv[:,:2]/uv[:,2:]
        roi=np.zeros((h,w),np.uint8)
        for xy in uv:
            if np.isfinite(xy).all():cv2.circle(roi,tuple(np.round(xy).astype(int)),16,1,-1)
        valid=roi.astype(bool)&mask[t].astype(bool)&np.isfinite(pts[t]).all(-1)
        cloud=np.asarray(pts[t])[valid]
        nearest=cKDTree(cloud).query(fk)[0]*1000 if len(cloud) else np.array([])
        frames.append(dict(frame=int(raw),table_signed_depth_error_mm=stat((depth[ok]-truez[ok])*1000),
            table_abs_depth_error_mm=stat(np.abs(depth[ok]-truez[ok])*1000),
            table_true_over_pred_ratio=stat(truez[ok]/depth[ok]),fk_nearest_roi_cloud_mm=stat(nearest)))
    raw=int(g['frame_indices'][probe]);E=g['extrinsic_world_from_camera_body'][probe]
    tri=np.load(BASE/f'frame{raw}_hand_visual_triangles.npz')['triangles_world'].astype(float)
    mesh=trimesh.Trimesh(vertices=tri.reshape(-1,3),faces=np.arange(tri.size//3).reshape(-1,3),process=False)
    sel=(u%6==0)&(v%6==0)
    vv,uu=np.nonzero(sel);dirs=rays[vv,uu]@(E[:3,:3]@C.T).T
    loc,rid,_=mesh.ray.intersects_location(np.tile(E[:3,3],(len(dirs),1)),dirs,multiple_hits=False)
    truth=(loc-E[:3,3])@E[:3,:3]@C.T;vv=vv[rid];uu=uu[rid]
    pred=np.asarray(pts[probe,vv,uu]);valid=mask[probe,vv,uu].astype(bool)&np.isfinite(pred).all(-1)&(truth[:,2]>0)
    truth,pred,vv,uu=truth[valid],pred[valid],vv[valid],uu[valid]
    err=np.linalg.norm(pred-truth,axis=-1)*1000
    surface=dict(frame=raw,xyz_error_mm=stat(err),signed_depth_error_mm=stat((pred[:,2]-truth[:,2])*1000),
        absolute_depth_error_mm=stat(np.abs(pred[:,2]-truth[:,2])*1000),true_depth_m=stat(truth[:,2]),predicted_depth_m=stat(pred[:,2]))
    np.savez_compressed(d/'surface_correspondences.npz',u=uu,v=vv,pred_xyz=pred,mesh_xyz=truth)
    im=cv2.imread(str(d/'rgb'/f'{probe:06d}.png'));im=cv2.resize(im,(w,h))
    for xx,yy,e in zip(uu,vv,err):cv2.circle(im,(int(xx),int(yy)),2,(0,220,0) if e<30 else (0,180,255) if e<100 else (0,0,255),-1)
    cv2.rectangle(im,(260,int(250*h/480)),(420,int(300*h/480)),(255,0,255),1)
    cv2.imwrite(str(d/'surface_error_overlay.jpg'),im)
    # A per-frame table trace exposes temporal drift without correcting the outputs.
    vals=[r['table_signed_depth_error_mm'].get('median',float('nan')) for r in frames]
    report['clips'][tag]=dict(frames=frames,hand_surface=surface,
        table_frame_median_signed_error_mm=stat(vals),
        fk_frame_median_nearest_roi_mm=stat([r['fk_nearest_roi_cloud_mm'].get('median',float('nan')) for r in frames]),
        predicted_focal_pixels=meta['diagnostics']['_da3_focal']*.5*np.hypot(h,w),
        calibrated_focal_pixels=[float(K[0,0]),float(K[1,1])])
    print(tag,json.dumps({k:val for k,val in report['clips'][tag].items() if k!='frames'}),flush=True)
(OUT/'geometry_comparison.json').write_text(json.dumps(report,indent=2))
