"""Read-only sandwich v2 / FK audit. No alignment fitting or source mutations."""
import os,json,hashlib,runpy
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
B=Path(__file__).resolve().parent
PF={k:Path(v['pointflow']) for k,v in json.load(open(B/'inventory.json')).items()}
FK=Path('/data/shichaojian/raw_data/dropper_dagger_mix_fk21')
RAW=Path('/data/shichaojian/raw_data/dropper_dagger_mix')
URDF=Path('/mnt/afs/WorldAct-cosmos3-edge-droid-sft_base/assets/dagger_fk/marvin_wuji_d435_dagger.urdf')
EXT=URDF
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
root=ET.parse(URDF).getroot();jointmap={j.get('name'):j for j in root.findall('joint')};M=np.eye(4)
for name in ['Joint_Stand','head_camera_base_joint','head_camera_bracket_joint','head_d435_mount_joint','head_d435_link_optical_joint']:
 j=jointmap[name];assert j.get('type')=='fixed';o=j.find('origin');A=np.eye(4);A[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ');A[:3,:3]=Rotation.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix();M=M@A
roll=np.diag([-1.,-1.,1.]);R=roll@M[:3,:3].T;t=-R@M[:3,3]
reference=json.load(open(URDF.parent/'validation.json'))['extrinsics']['dagger'];assert np.allclose(R,reference['R'],atol=1e-9) and np.allclose(t,reference['t'],atol=1e-9)
ext={'URDF_MD5':hashlib.md5(URDF.read_bytes()).hexdigest()}
def stats(a):
 a=np.asarray(a);a=a[np.isfinite(a)]
 return dict(n=len(a),median=float(np.median(a)),p05=float(np.percentile(a,5)),p95=float(np.percentile(a,95)),mean=float(a.mean())) if len(a) else dict(n=0)
def intrinsic(ep):
 h=json.load(open(RAW/ep/'auxiliary_camera/metadata.json'))['capture_metadata']['cameras']['head'];ci=h.get('streams',{}).get('color',{}).get('intrinsics')
 if ci:return ci,ep
 for d in sorted(RAW.glob('episode_*')):
  try:h2=json.load(open(d/'auxiliary_camera/metadata.json'))['capture_metadata']['cameras']['head']
  except (FileNotFoundError,KeyError):continue
  ci=h2.get('streams',{}).get('color',{}).get('intrinsics')
  if h2.get('serial_number')==h.get('serial_number') and ci:return ci,d.name
 raise ValueError('No intrinsic '+ep)
def load(ep):
 d=PF[ep];n=np.load(FK/ep/'annotations/wuji_fk21.npz');side=list(n['side_names']).index('right');assert n['side_is_observed'][side]
 assert str(n['coordinate_frame'])=='Link_Base' and str(n['units'])=='metre'
 cam=n['positions'][:,side]@R.T+t
 a={k:np.load(d/(k+'.npy'),mmap_mode='r') for k in ['frame_offsets','frame_indices','timestamps_sec','intrinsics','obs_pos','obs_uv','obs_label','obs_valid','obs_unique']}
 ci,src=intrinsic(ep);K=np.array([[ci['fx']*640/ci['width'],0,(ci['ppx']+.5)*640/ci['width']-.5],[0,ci['fy']*448/ci['height'],(ci['ppy']+.5)*448/ci['height']-.5],[0,0,1.]])
 return a,cam,K,src,n
def main():
 records=[];joints=[];episodes=[];fails=[];boundaries=[]
 
 for ep in sorted(PF):
  try:
   a,cam,K,src,n=load(ep);T=len(a['frame_indices']);assert T==len(cam);assert np.array_equal(a['frame_indices'],np.arange(T))
   samples=sorted(set(np.linspace(0,T-1,20,dtype=int).tolist()+[f for b in range(128,T,128) for f in [b-1,b]]+([650] if T>650 else [])))
   eprows=[];byframe={}
   for f in samples:
    s,e=map(int,a['frame_offsets'][f:f+2]);m=(a['obs_label'][s:e]==2)&a['obs_valid'][s:e]&a['obs_unique'][s:e]
    p=np.asarray(a['obs_pos'][s:e][m]);uv=np.asarray(a['obs_uv'][s:e][m],float);c=cam[f]
    q=c@K.T;q=q[:,:2]/q[:,2:];inside=(c[:,2]>0)&(q[:,0]>=0)&(q[:,0]<640)&(q[:,1]>=0)&(q[:,1]<448)
    projected=p@a['intrinsics'][f].T;projected=projected[:,:2]/projected[:,2:]*[640,448]-.5
    projection_px=np.linalg.norm(projected-uv,axis=1)
    nn=cKDTree(p).query(c)[0]*1000 if len(p) else np.full(21,np.nan)
    row=dict(episode=ep,frame=f,hand_points=len(p),in_image=int(inside.sum()),matched=0,nn3d_median_mm=float(np.nanmedian(nn)))
    items=[]
    if len(p):
     tree=cKDTree(uv)
     for j in np.flatnonzero(inside):
      ids=tree.query_ball_point(q[j],3.)
      if len(ids)<3:continue
      pp=np.median(p[ids],axis=0);pixel=np.median(uv[ids],axis=0)
      ray=np.linalg.solve(K,np.r_[pixel,1]);pc=ray*pp[2]
      item=dict(episode=ep,frame=f,joint=int(j),neighborhood_n=len(ids),signed_depth_mm=float((pp[2]-c[j,2])*1000),raw_xyz_gap_mm=float(np.linalg.norm(pp-c[j])*1000),known_K_xyz_gap_mm=float(np.linalg.norm(pc-c[j])*1000),raw_xy_gap_mm=float(np.linalg.norm(pp[:2]-c[j,:2])*1000),nn3d_mm=float(nn[j]))
      items.append(item)
    row['matched']=len(items)
    for key in ['signed_depth_mm','raw_xyz_gap_mm','known_K_xyz_gap_mm','raw_xy_gap_mm']:
     row[key]=float(np.median([x[key] for x in items])) if items else None
    row['self_projection_median_px']=float(np.median(projection_px)) if len(p) else None
    records.append(row);eprows.append(row);joints.extend(items);byframe[f]=row
   for b in range(128,T,128):
    u,v=byframe[b-1],byframe[b]
    if u['signed_depth_mm'] is not None and v['signed_depth_mm'] is not None:boundaries.append(dict(episode=ep,frame=b,signed_depth_change_mm=v['signed_depth_mm']-u['signed_depth_mm']))
   epj=[x for x in joints if x['episode']==ep]
   rel=n['timestamps']-n['timestamps'][0];drift=rel-np.asarray(a['timestamps_sec'])
   episodes.append(dict(episode=ep,frames=T,sampled_frames=len(samples),intrinsic_source=src,fx_pred_px=float(a['intrinsics'][0,0,0]*640),fx_true_px=float(K[0,0]),fy_pred_px=float(a['intrinsics'][0,1,1]*448),fy_true_px=float(K[1,1]),timestamp_relative_difference_sec=stats(drift),frame_rate=float(n['frame_rate']),matched_joint_samples=len(epj),signed_depth_mm=stats([x['signed_depth_mm'] for x in epj]),raw_xyz_gap_mm=stats([x['raw_xyz_gap_mm'] for x in epj]),known_K_xyz_gap_mm=stats([x['known_K_xyz_gap_mm'] for x in epj])))
   print(ep,'frames',T,'samples',len(samples),'matched',len(epj),'depth',episodes[-1]['signed_depth_mm'].get('median'),flush=True)
  except Exception as ex:
   fails.append(dict(episode=ep,error=repr(ex)));print('FAILED',ep,repr(ex),flush=True)
 summary=dict(episodes=len(episodes),sampled_frames=len(records),matched_joint_samples=len(joints),candidate_in_image=sum(x['in_image'] for x in records),failed=fails,urdf_md5=ext['URDF_MD5'],pointflow_version='merged 9.24/dagger + 10.1/dagger_new efep_seg_v61',extrinsic_source=str(EXT),alignment_fitted=False,metrics={k:stats([x[k] for x in joints]) for k in ['signed_depth_mm','raw_xyz_gap_mm','known_K_xyz_gap_mm','raw_xy_gap_mm','nn3d_mm']},absolute_depth_mm=stats([abs(x['signed_depth_mm']) for x in joints]),boundary_absolute_depth_change_mm=stats([abs(x['signed_depth_change_mm']) for x in boundaries]),limitations=['FK joints are internal anatomical reference, not observed hand surface; values are modality gaps, not pure DA3 reconstruction error.','3px same-pixel neighborhoods require >=3 retained hand pixels; missing/occluded joints excluded and coverage reported.','No independent sensor depth or mesh-surface ground truth used.','v61 hand masks may contain arm/cables; mesh intersection restricts surface audit to URDF hand geometry.','Frame indices matched directly; timestamp irregularity reported, sensor latency not measured.','Known-K reprojection is diagnostic only, no source data changed.'])
 for name,data in [('summary',summary),('episodes',episodes),('frames',records),('joints',joints),('boundaries',boundaries)]: (B/(name+'.json')).write_text(json.dumps(data,indent=2))
 print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":main()
