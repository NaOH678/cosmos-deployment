from audit import *
import cv2
EP='episode_0013_20260731_133649';f=650
a,cam,K,src,n=load(EP);c=cam[f];q=c@K.T;q=q[:,:2]/q[:,2:]
old=Path('/data/shichaojian/datasets/sandwich_dense_fullseq_10_0298_20260908/outputs')/EP
p=np.load(old/'position.npy',mmap_mode='r')[f].reshape(-1,3);uv=np.load(old/'uv_px.npy',mmap_mode='r')[f].reshape(-1,2);ok=np.load(old/'valid.npy',mmap_mode='r')[f].reshape(-1);ok=ok&np.isfinite(p).all(1)&np.isfinite(uv).all(1)
p,uv=p[ok],uv[ok];print('old ranges',np.min(uv,0),np.max(uv,0))
meta=json.load(open(RAW/EP/'auxiliary_camera/metadata.json'))['capture_metadata']['cameras']['head']['streams'];dk=meta['depth']['intrinsics'];ck=meta['color']['intrinsics'];d=cv2.imread(str(B/'depth650.png'),-1)*meta['depth']['depth_scale_m'];yy,xx=np.indices(d.shape);good=(d>.1)&(d<3);z=d[good];pd=np.stack([(xx[good]-dk['ppx'])/dk['fx']*z,(yy[good]-dk['ppy'])/dk['fy']*z,z],-1)
# RealSense extrinsics.rotation is column-major; explicit order='F'.
e=meta['depth']['extrinsics_to_color'];pd=pd@np.array(e['rotation']).reshape(3,3,order='F').T+e['translation'];duv=pd@K.T;duv=duv[:,:2]/duv[:,2:]
s,e=map(int,a['frame_offsets'][f:f+2]);m=(a['obs_label'][s:e]==2)&a['obs_valid'][s:e]&a['obs_unique'][s:e]
res={}
for name,pts,uvs in [('old_3d_ff',p,uv),('current_v2',np.asarray(a['obs_pos'][s:e][m]),np.asarray(a['obs_uv'][s:e][m])),('D435',pd,duv)]:
 tree=cKDTree(uvs);vals=[]
 for j in range(21):
  if not(0<=q[j,0]<640 and 0<=q[j,1]<448):continue
  ix=tree.query_ball_point(q[j],3)
  if len(ix)<3:continue
  vals.append(dict(joint=j,signed_depth_mm=float((np.median(pts[ix,2])-c[j,2])*1000),n=len(ix)))
 res[name]=dict(values=vals,stats=stats([x['signed_depth_mm'] for x in vals]))
print(json.dumps(res,indent=2));(B/'version_and_sensor_check650.json').write_text(json.dumps(res,indent=2))
