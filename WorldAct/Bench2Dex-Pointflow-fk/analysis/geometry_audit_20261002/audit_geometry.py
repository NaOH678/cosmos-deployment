"""Read-only numerical audit of downloaded Bench2Dex Wuji assets and one episode.

Run with PYTHONPATH=/tmp/bench2dex_audit_deps and the existing Cosmos Python.
This checks kinematics without running Isaac physics or DA3 inference.
"""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np
import h5py
from pxr import Usd, UsdGeom, UsdPhysics

OUT = Path(__file__).resolve().parent

def rot(axis, angle):
    angle=float(angle)
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    x, y, z = a; K = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)

def rpy(v):
    r,p,y = v
    return rot([0,0,1],y) @ rot([0,1,0],p) @ rot([1,0,0],r)

def tf(R=None, t=None):
    T=np.eye(4)
    if R is not None:T[:3,:3]=R
    if t is not None:T[:3,3]=t
    return T

def quat(q):
    w=float(q.GetReal()); v=np.array(q.GetImaginary(),float)
    n=np.sqrt(w*w+v@v);w/=n;v/=n
    x,y,z=v; K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+2*w*K+2*(K@K)

def angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1))))

def stats(x):
    x=np.asarray(x)
    return dict(mean=float(x.mean()),median=float(np.median(x)),p95=float(np.percentile(x,95)),max=float(x.max()))

def fit_rigid(x,y):
    xc=x.mean(0);yc=y.mean(0)
    u,_,vt=np.linalg.svd((x-xc).T@(y-yc))
    R=vt.T@u.T
    if np.linalg.det(R)<0:vt[-1]*=-1;R=vt.T@u.T
    return tf(R,yc-R@xc)

root=ET.parse('/tmp/bench2dex_wuji.urdf').getroot()
uj={}
for j in root.findall('joint'):
    o=j.find('origin');a=j.find('axis')
    xyz=np.fromstring(o.get('xyz','0 0 0'),sep=' ') if o is not None else np.zeros(3)
    rr=np.fromstring(o.get('rpy','0 0 0'),sep=' ') if o is not None else np.zeros(3)
    uj[j.get('name')]=dict(parent=j.find('parent').get('link'),child=j.find('child').get('link'),origin=tf(rpy(rr),xyz),axis=np.fromstring(a.get('xyz'),sep=' ') if a is not None else None,type=j.get('type'))

stage=Usd.Stage.Open('/tmp/bench2dex_usd/Multi_UR5_wuji_with_flange.usd')
sj={}
for p in stage.Traverse():
    if p.IsA(UsdPhysics.Joint):
        j=UsdPhysics.Joint(p);b0=j.GetBody0Rel().GetTargets();b1=j.GetBody1Rel().GetTargets()
        if not b0 or not b1:continue
        axis=p.GetAttribute('physics:axis').Get()
        sj[p.GetName()]=dict(parent=b0[0].name,child=b1[0].name,J0=tf(quat(j.GetLocalRot0Attr().Get()),j.GetLocalPos0Attr().Get()),J1=tf(quat(j.GetLocalRot1Attr().Get()),j.GetLocalPos1Attr().Get()),axis=np.eye(3)['XYZ'.index(axis)] if axis else None,type=p.GetTypeName())

def fk(q, source='urdf'):
    nodes={'base_link':np.eye(4)};pending=dict(uj if source=='urdf' else sj)
    while pending:
        progress=False
        for n,j in list(pending.items()):
            if j['parent'] not in nodes:continue
            moving=j['type'] in ['revolute','continuous','PhysicsRevoluteJoint']
            motion=tf(rot(j['axis'],q.get(n,0))) if moving else np.eye(4)
            rel=j['origin']@motion if source=='urdf' else j['J0']@motion@np.linalg.inv(j['J1'])
            nodes[j['child']]=nodes[j['parent']]@rel
            del pending[n];progress=True
        if not progress:raise ValueError(list(pending))
    return nodes

f=h5py.File('/tmp/bench2dex_replay21_ep0.hdf5')
g=h5py.File('/tmp/bench2dex_origin21_ep0.hdf5')
names=[v.decode() for v in f['robot/joint_names'][:]]
qpos=f['robot/qpos'][:]
defs={x['camera_id']:x for x in json.loads(f['meta/camera_definitions'][()])}
sample=json.loads(f['meta/scene_generalization_sample'][()])
sample_c=sample['spatial']['camera']['cameras']
report={'episode':'task21/replay-generalization/episode_000000.hdf5','frames':len(qpos),'qpos_shape':list(qpos.shape),'qpos_origin_replay_max_abs':float(np.max(np.abs(qpos-g['robot/qpos'][:]))),'joint_names_equal_origin_replay':names==[v.decode() for v in g['robot/joint_names'][:]],'urdf_missing_recorded_joints':sorted(set(names)-set(uj)),'recorded_missing_urdf_movable_joints':sorted({n for n,j in uj.items() if j['type']!='fixed'}-set(names)),'urdf_joint_count':len(uj),'usd_joint_count':len(sj),'usd_missing_urdf_joints':sorted(set(uj)-set(sj)),'usd_meters_per_unit':UsdGeom.GetStageMetersPerUnit(stage),'usd_base_link_world_transform':np.array(UsdGeom.Xformable(stage.GetPrimAtPath('/ur5/base_link')).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T.tolist()}
errs=[];rots=[]
for idx in [0,100,200,400,600,712]:
    q=dict(zip(names,qpos[idx]));u=fk(q);s=fk(q,'usd')
    for n in u.keys()&s.keys():
        errs.append(np.linalg.norm(u[n][:3,3]-s[n][:3,3])*1000)
        rots.append(angle(u[n][:3,:3]@s[n][:3,:3].T))
report['usd_urdf_fk_link_position_error_mm']=stats(errs)
report['usd_urdf_fk_link_orientation_error_deg']=stats(rots)
report['usd_joint_name_sanitize_matches']=all(n.replace('-','_') in sj for n in uj)
report['usd_joint_name_note']='Eight fixed-joint names use underscores instead of URDF hyphens; child link topology is used for FK.'

# Independent recorded wrist-camera centers constrain the robot base in world.
points_base=[];points_world=[];frame_ids=[];side_ids=[];all_fk=[]
for i,qrow in enumerate(qpos):
    nodes=fk(dict(zip(names,qrow)));all_fk.append(nodes)
    for side in ['right','left']:
        cid='cam_wrist_'+side;d=defs[cid];noise=sample_c[cid]
        offset=np.array(d['offset_xyz'])+noise['offset_xyz_offset_m']
        pb=(nodes[d['parent_link']]@np.r_[offset,1])[:3]
        points_base.append(pb);points_world.append(f['cameras'][cid]['extrinsic_world_from_cam'][i,:3,3]);frame_ids.append(i);side_ids.append(side)
x=np.array(points_base);y=np.array(points_world);frame_ids=np.array(frame_ids);side_ids=np.array(side_ids)
Tnom=tf(rpy([0,0,np.pi/2]),[.5,-.43,.75])
Tfit=fit_rigid(x[frame_ids<100],y[frame_ids<100])
for tag,T in [('current_code_nominal',Tnom),('fit_first100_frames',Tfit)]:
    error=np.linalg.norm(x@T[:3,:3].T+T[:3,3]-y,axis=-1)*1000
    report[tag]={'T_world_from_base':T.tolist(),'all_center_error_mm':stats(error),'heldout_center_error_mm':stats(error[frame_ids>=100]),'right_error_mm':stats(error[side_ids=='right']),'left_error_mm':stats(error[side_ids=='left'])}
report['fitted_vs_nominal_translation_m']=(Tfit[:3,3]-Tnom[:3,3]).tolist()
report['fitted_vs_nominal_rotation_deg']=angle(Tfit[:3,:3]@Tnom[:3,:3].T)
pred_centers=(x@Tnom[:3,:3].T+Tnom[:3,3]).reshape(-1,2,3)
actual_centers=y.reshape(-1,2,3)
report['wrist_extrinsic_lag_scan']={}
for lag in range(-4,5):
    pa=pred_centers[max(0,lag):min(len(qpos),len(qpos)+lag)]
    ya=actual_centers[max(0,-lag):min(len(qpos),len(qpos)-lag)]
    report['wrist_extrinsic_lag_scan'][str(lag)]={'definition':'qpos index minus camera extrinsic index','center_error_mm':stats(np.linalg.norm(pa-ya,axis=-1)*1000)}
Taligned=fit_rigid(x.reshape(-1,2,3)[:100].reshape(-1,3),actual_centers[1:101].reshape(-1,3))
heldout=x.reshape(-1,2,3)[100:-1].reshape(-1,3)@Taligned[:3,:3].T+Taligned[:3,3]
report['base_fit_after_lag_correction']={'calibration':'qpos frames 0..99 versus camera extrinsics 1..100, both wrists','T_world_from_base':Taligned.tolist(),'heldout_center_error_mm':stats(np.linalg.norm(heldout-actual_centers[101:].reshape(-1,3),axis=-1)*1000),'translation_difference_from_nominal_m':(Taligned[:3,3]-Tnom[:3,3]).tolist(),'rotation_difference_from_nominal_deg':angle(Taligned[:3,:3]@Tnom[:3,:3].T)}
# The recorded camera matrix stores Isaac camera-body axes; convert optical axes
# before comparing the URDF wrist and recorded mounting rotation.
C=np.array([[0,-1,0],[0,0,-1],[1,0,0.]])
orientation_errors=[]
for idx in range(len(qpos)-1):
    for side in ['right','left']:
        cid='cam_wrist_'+side;d=defs[cid];noise=sample_c[cid]
        Roff=rpy(np.array(d['offset_rpy'])+np.radians(noise['rpy_offset_deg']))
        Rpred=Tnom[:3,:3]@all_fk[idx][d['parent_link']][:3,:3]@Roff
        Ractual=f['cameras'][cid]['extrinsic_world_from_cam'][idx+1,:3,:3]@C.T
        orientation_errors.append(angle(Rpred@Ractual.T))
report['lag_corrected_wrist_orientation_error_deg']=stats(orientation_errors)
report['cameras']={c:{'rgb_shape':list(f['cameras'][c]['rgb'].shape),'intrinsic':f['cameras'][c]['intrinsic'][:].tolist(),'extrinsics_finite':bool(np.isfinite(f['cameras'][c]['extrinsic_world_from_cam'][:]).all()),'camera_model':f['cameras'][c]['camera_model'][()].decode()} for c in f['cameras']}
report['depth_available']=any('depth_m' in f['cameras'][c] for c in f['cameras'])
report['metadata_created_at']=f['meta/created_at'][()].decode()
report['metadata_has_asset_or_code_hash']=any('hash' in k or 'commit' in k for k in f['meta'])
import hashlib
report['inputs_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('/tmp/bench2dex_wuji.urdf'),Path('/tmp/bench2dex_usd/Multi_UR5_wuji_with_flange.usd'),*Path('/tmp/bench2dex_usd/configuration').glob('*.usd'),Path('/tmp/bench2dex_replay21_ep0.hdf5'),Path('/tmp/bench2dex_origin21_ep0.hdf5')]}
report['limitations']=['One task21 episode sampled; not a full dataset audit.','USD joint transforms evaluated algebraically without an Isaac physics rollout.','RGB skeleton overlay is qualitative; RGB frame lag is not established by the extrinsic lag test.','Separate DA3 depth-backbone probe and measurements are stored in da3_probe*.json and da3_metric_audit*.json; no full Track4World trajectory evaluation.']

# Project keypoint chains onto actual RGB with nominal and fitted base transforms.
import cv2
C=np.array([[0,-1,0],[0,0,-1],[1,0,0.]])
panels=[]
for idx in [0,200,400,600]:
    row=[]
    for cid in ['cam_overhead','cam_stereo_left']:
        cam=f['cameras'][cid];raw=cam['rgb'][idx]
        im=cv2.imdecode(np.asarray(raw,np.uint8),cv2.IMREAD_COLOR)
        if im is None:raise ValueError('RGB decode failed')
        E=cam['extrinsic_world_from_cam'][idx];K=cam['intrinsic'][:]
        for side,color in [('left',(255,180,0)),('right',(0,220,255))]:
            for finger in range(1,6):
                links=[side+'_palm_link']+[f'{side}_finger{finger}_link{k}' for k in range(1,5)]+[f'{side}_finger{finger}_tip_link']
                p=np.array([all_fk[idx][n][:3,3] for n in links]);pw=p@Tnom[:3,:3].T+Tnom[:3,3]
                pc=(pw-E[:3,3])@E[:3,:3]@C.T;uv=pc@K.T;uv=uv[:,:2]/uv[:,2:]
                pts=np.round(uv).astype(np.int32)
                for a,b in zip(pts[:-1],pts[1:]):cv2.line(im,tuple(a),tuple(b),color,1,cv2.LINE_AA)
                for p2 in pts:cv2.circle(im,tuple(p2),2,color,-1,cv2.LINE_AA)
        cv2.putText(im,f'{cid} frame {idx} | nominal base',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
        row.append(im)
    panels.append(np.concatenate(row,axis=1))
cv2.imwrite(str(OUT/'fk_projection.jpg'),np.concatenate(panels,axis=0))
np.savez(OUT/'wrist_camera_correspondences.npz',points_base=x,points_world=y,frame_ids=frame_ids,side_ids=side_ids)
(OUT/'audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='cameras'},indent=2))
