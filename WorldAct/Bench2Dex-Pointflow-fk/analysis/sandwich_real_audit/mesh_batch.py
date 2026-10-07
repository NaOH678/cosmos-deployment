from audit import *
import xml.etree.ElementTree as ET
import trimesh
from scipy.spatial.transform import Rotation
EP='episode_0013_20260731_133649'
root=ET.parse(URDF).getroot();js=root.findall('joint');links={x.get('name'):x for x in root.findall('link')}
def origin(el):
 M=np.eye(4)
 if el is not None:
  M[:3,3]=np.fromstring(el.get('xyz','0 0 0'),sep=' ');M[:3,:3]=Rotation.from_euler('xyz',np.fromstring(el.get('rpy','0 0 0'),sep=' ')).as_matrix()
 return M

meshlocal={}
for name,l in links.items():
 if not name.startswith('right_hand'):continue
 for i,v in enumerate(l.findall('visual')):
  me=v.find('geometry/mesh')
  if me is None:continue
  path=URDF.parent.parent.parent/me.get('filename').replace('package://','')
  mesh=trimesh.load(str(path),force='mesh',process=False);vs=mesh.vertices*np.fromstring(me.get('scale','1 1 1'),sep=' ');O=origin(v.find('origin'));vs=vs@O[:3,:3].T+O[:3,3];meshlocal[(name,i)]=(vs,mesh.faces)

import cv2
from concurrent.futures import ProcessPoolExecutor

def worker(ep):
 qpos=np.load(B/'qpos'/(ep+'.npy'));a,cam,K,src,n=load(ep);res=[]
 frames=sorted(set(np.linspace(0,len(cam)-1,6,dtype=int).tolist()+([650] if len(cam)>650 else [])))
 for frame in frames:
  q=qpos[frame];vals={f'Joint{i+1}_R':q[27+i] for i in range(7)};vals.update({f'right_hand_finger{i+1}_joint{j+1}':q[34+i*4+j] for i in range(5) for j in range(4)})
  poses={'Link_Base':np.eye(4)};pending=js.copy()
  while pending:
   progressed=False
   for j in pending[:]:
    par=j.find('parent').get('link');child=j.find('child').get('link')
    if par not in poses:continue
    M=origin(j.find('origin'));A=np.eye(4)
    if j.get('type') in ['revolute','continuous']:
     ax=np.fromstring(j.find('axis').get('xyz'),sep=' ');A[:3,:3]=Rotation.from_rotvec(ax*vals.get(j.get('name'),0)).as_matrix()
    poses[child]=poses[par]@M@A;pending.remove(j);progressed=True
   assert progressed
  # Match explicit FK landmarks against link origins; diagnostic only, no fitting.
  pts=np.array([p[:3,3] for name,p in poses.items() if name.startswith('right_hand')]);dist=cKDTree(pts).query(n['positions'][frame,1])[0]*1000
  verts=[];faces=[];count=0
  for (name,i),(v,fa) in meshlocal.items():
   P=poses[name];v=(v@P[:3,:3].T+P[:3,3])@R.T+t;verts.append(v);faces.append(fa+count);count+=len(v)
  mesh=trimesh.Trimesh(vertices=np.concatenate(verts),faces=np.concatenate(faces),process=False)
  s,e=map(int,a['frame_offsets'][frame:frame+2]);uv=a['obs_uv'][s:e];keep=(a['obs_label'][s:e]==2)&a['obs_valid'][s:e]&a['obs_unique'][s:e]&(uv[:,0]%4==0)&(uv[:,1]%4==0)
  mask=np.zeros((448,640),np.uint8);allhand=(a['obs_label'][s:e]==2)&a['obs_valid'][s:e]&a['obs_unique'][s:e];mask[uv[allhand,1],uv[allhand,0]]=1;interior=cv2.erode(mask,np.ones((5,5),np.uint8));keep=keep&(interior[uv[:,1],uv[:,0]]>0)
  uv=np.asarray(uv[keep]);pf=np.asarray(a['obs_pos'][s:e][keep]);rays=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(K).T
  loc,ids,_=mesh.ray.intersects_location(np.zeros_like(rays),rays,multiple_hits=False)
  dz=(pf[ids,2]-loc[:,2])*1000;xyz=np.linalg.norm(pf[ids]-loc,axis=1)*1000
  res.append(dict(frame=frame,n_rays=len(uv),n_hits=len(ids),fk_link_origin_nearest_mm=stats(dist),surface_depth_difference_mm=stats(dz),surface_xyz_gap_mm=stats(xyz)))
 
 

 return dict(episode=ep,frames=res)

if __name__=='__main__':
 eps=[e['episode'] for e in json.load(open(B/'episodes.json'))];results=[]
 with ProcessPoolExecutor(max_workers=4) as pool:
  for result in pool.map(worker,eps):
   results.append(result);print(result['episode'],len(result['frames']),flush=True)
 (B/'mesh_batch.json').write_text(json.dumps(results,indent=2))
 print('DONE',len(results),flush=True)
