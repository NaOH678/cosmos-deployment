import sys,json
from pathlib import Path
import numpy as np,cv2
from scipy.spatial import cKDTree
B=Path(__file__).resolve().parent;REF=B.parent/'sandwich_real_audit';sys.path.insert(0,str(REF))
from audit import load,RAW
EP='episode_0013_20260731_133649';_,_,K,_,_=load(EP);meta=json.load(open(RAW/EP/'auxiliary_camera/metadata.json'))['capture_metadata']['cameras']['head']['streams'];dk=meta['depth']['intrinsics'];d=cv2.imread(str(REF/'depth650.png'),-1)*meta['depth']['depth_scale_m'];yy,xx=np.indices(d.shape);ok=(d>.1)&(d<3);z=d[ok];pd=np.stack([(xx[ok]-dk['ppx'])/dk['fx']*z,(yy[ok]-dk['ppy'])/dk['fy']*z,z],-1);e=meta['depth']['extrinsics_to_color'];pd=pd@np.array(e['rotation']).reshape(3,3,order='F').T+e['translation'];uv=pd@K.T;uv=uv[:,:2]/uv[:,2:];tree=cKDTree(uv)
x=np.load(B/'comparison_0650.npz');rows=[]
for i,pix in enumerate(x['uv']):
 ids=tree.query_ball_point(pix,2)
 if len(ids)<3:continue
 depth=np.median(pd[ids,2]);rows.append([(x[k][i,2]-depth)*1000 for k in ['old','fixed','surface']])
a=np.asarray(rows);out={'frame':650,'samples':len(a),'note':'Same-pixel D435 depth neighbourhood, >=3 samples within 2px; sensor corroboration, not perfect truth.'}
for i,k in enumerate(['old_minus_d435','fixed_minus_d435','mesh_minus_d435']):out[k+'_mm']={'signed_median':float(np.median(a[:,i])),'absolute_median':float(np.median(abs(a[:,i]))),'absolute_p95':float(np.percentile(abs(a[:,i]),95))}
(B/'sensor_comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
