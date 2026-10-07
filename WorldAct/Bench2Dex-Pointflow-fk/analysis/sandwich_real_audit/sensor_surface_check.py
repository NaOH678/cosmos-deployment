from audit import *
import cv2
EP='episode_0013_20260731_133649';a,cam,K,src,n=load(EP);meta=json.load(open(RAW/EP/'auxiliary_camera/metadata.json'))['capture_metadata']['cameras']['head']['streams'];dk=meta['depth']['intrinsics'];d=cv2.imread(str(B/'depth650.png'),-1)*meta['depth']['depth_scale_m'];yy,xx=np.indices(d.shape);good=(d>.1)&(d<3);z=d[good];pd=np.stack([(xx[good]-dk['ppx'])/dk['fx']*z,(yy[good]-dk['ppy'])/dk['fy']*z,z],-1);e=meta['depth']['extrinsics_to_color'];pd=pd@np.array(e['rotation']).reshape(3,3,order='F').T+e['translation'];duv=pd@K.T;duv=duv[:,:2]/duv[:,2:]
x=np.load(B/'mesh_surface_0650.npz');tree=cKDTree(duv);rows=[];points=[]
for i,uv in enumerate(x['uv']):
 ids=tree.query_ball_point(uv,2)
 if len(ids)<3:continue
 d435=np.median(pd[ids],axis=0);rows.append([float((x['pf'][i,2]-d435[2])*1000),float((d435[2]-x['surface'][i,2])*1000),float((x['pf'][i,2]-x['surface'][i,2])*1000)]);points.append(d435)
y=np.array(rows);res=dict(episode=EP,frame=650,n=len(y),pf_minus_d435_z_mm=stats(y[:,0]),d435_minus_mesh_z_mm=stats(y[:,1]),pf_minus_mesh_z_mm=stats(y[:,2]),note='Same retained hand pixels with URDF first-hit surface. Sensor is an independent cross-check, not perfect ground truth; 2px neighbourhoods, >=3 sensor samples. RealSense column-major extrinsics.')
(B/'sensor_surface_check650.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
