"""CPU: full-episode FK and selected visual meshes; no estimated alignment."""
from pathlib import Path
import runpy,json
import numpy as np
from pxr import Usd,UsdGeom
HERE=Path(__file__).resolve().parent
AUDIT=HERE.parent/'geometry_audit_20261002'
a=runpy.run_path(str(AUDIT/'audit_geometry.py'))
f=a['f'];stage=a['stage'];Tbase=a['Tnom'];all_fk=a['all_fk'];C=a['C']
links=sorted(all_fk[0]);world=np.array([[n[k][:3,3] for k in links] for n in all_fk])@Tbase[:3,:3].T+Tbase[:3,3]
E=f['cameras/cam_overhead/extrinsic_world_from_cam'][:];K=f['cameras/cam_overhead/intrinsic'][:]
optical=np.einsum('tni,tij->tnj',world-E[:,None,:3,3],E[:,:3,:3])@C.T
np.savez_compressed(HERE/'full_geometry.npz',fk_optical_m=optical,fk_world_m=world,link_names=links,
 intrinsic=K,extrinsic_world_from_camera_body=E,optical_from_camera_body=C,T_world_from_base=Tbase,qpos=a['qpos'],joint_names=a['names'])
local=[]
for p in Usd.PrimRange(stage.GetPseudoRoot(),Usd.TraverseInstanceProxies()):
 path=str(p.GetPath())
 if not p.IsA(UsdGeom.Mesh) or '/visuals/' not in path:continue
 link=path.split('/')[2]
 if not link.startswith(('left_finger','right_finger','left_palm','right_palm')):continue
 m=UsdGeom.Mesh(p);pts=np.array(m.GetPointsAttr().Get(),float)
 idx=np.array(m.GetFaceVertexIndicesAttr().Get(),int);counts=np.array(m.GetFaceVertexCountsAttr().Get(),int)
 assert np.all(counts==3)
 Tmesh=np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
 Tlink=np.array(UsdGeom.Xformable(stage.GetPrimAtPath('/ur5/'+link)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
 T=np.linalg.inv(Tlink)@Tmesh
 local.append((link,pts@T[:3,:3].T+T[:3,3],idx.reshape(-1,3)))
frames=[0,63,100,200,300,316,347,348,349,379,400,500,600,712]
(HERE/'meshes').mkdir(exist_ok=True)
for frame in frames:
 parts=[]
 for link,pts,idx in local:
  T=Tbase@all_fk[frame][link];pw=pts@T[:3,:3].T+T[:3,3];parts.append(pw[idx])
 np.savez_compressed(HERE/'meshes'/f'frame{frame}.npz',triangles_world=np.concatenate(parts).astype(np.float32))
print('Saved full FK and meshes',frames,flush=True)
