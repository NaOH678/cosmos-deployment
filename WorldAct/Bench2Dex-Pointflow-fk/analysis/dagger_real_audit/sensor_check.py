from audit import *
import cv2
records=[]
for row in json.load(open(B/'sensor_inventory.json')):
 if 'error' in row:records.append(row);continue
 ep,f=row['episode'],row['frame'];a,cam,K,src,n=load(ep);meta=json.load(open(RAW/ep/'auxiliary_camera/metadata.json'))['capture_metadata']['cameras']['head']['streams'];dk=meta['depth']['intrinsics'];d=cv2.imread(row['file'],-1)*meta['depth']['depth_scale_m'];yy,xx=np.indices(d.shape);good=(d>.1)&(d<3);z=d[good];pd=np.stack([(xx[good]-dk['ppx'])/dk['fx']*z,(yy[good]-dk['ppy'])/dk['fy']*z,z],-1);e=meta['depth']['extrinsics_to_color'];pd=pd@np.array(e['rotation']).reshape(3,3,order='F').T+e['translation'];duv=pd@K.T;duv=duv[:,:2]/duv[:,2:];tree=cKDTree(duv);x=np.load(B/'surfaces'/ep/f'frame{f:04d}.npz');vals=[]
 for i,pix in enumerate(x['uv']):
  ids=tree.query_ball_point(pix,2)
  if len(ids)<3:continue
  dep=np.median(pd[ids,2]);vals.append([(x['pf'][i,2]-dep)*1000,(dep-x['surface'][i,2])*1000])
 vals=np.asarray(vals);row.update(n=len(vals))
 if len(vals):row.update(pf_minus_d435_z_mm=stats(vals[:,0]),d435_minus_mesh_z_mm=stats(vals[:,1]))
 records.append(row)
(B/'sensor_check.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
