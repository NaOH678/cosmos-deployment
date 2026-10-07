"""Export posed visual hand triangles for surface-depth checks (no simulator)."""
import runpy
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom
OUT=Path(__file__).resolve().parent
a=runpy.run_path(str(OUT/'audit_geometry.py'))
stage=a['stage'];Tbase=a['Tnom'];parts=[]
import sys
frame=int(sys.argv[1]) if len(sys.argv)>1 else 0
pose=a['all_fk'][frame]
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
    T=Tbase@pose[link]@np.linalg.inv(Tlink)@Tmesh
    pw=pts@T[:3,:3].T+T[:3,3]
    parts.append(pw[idx.reshape(-1,3)])
tri=np.concatenate(parts).astype(np.float32)
np.savez_compressed(OUT/f'frame{frame}_hand_visual_triangles.npz',triangles_world=tri)
print('Exported',tri.shape,'hand visual triangles',flush=True)
