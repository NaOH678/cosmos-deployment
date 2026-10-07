"""Held-out D435 calibration experiment. Read-only sources, CPU only, seed 17."""
from pathlib import Path
import os,json,pickle,hashlib
import numpy as np,cv2,lmdb
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from concurrent.futures import ProcessPoolExecutor
B=Path(__file__).resolve().parent
S=Path('/data/shichaojian/pf_out/9.24/sandwich/efep_seg_v2');D=json.load(open(B.parent/'dagger_real_audit/inventory.json'))
def choose(items,n):
 items=sorted(items);return [items[i] for i in np.linspace(0,len(items)-1,n,dtype=int)]
selection=[]
for group,eps in [('sandwich',choose([p.name for p in S.glob('episode_*')],12)),('dagger',choose([k for k,v in D.items() if v['source']=='9.24'],6)+choose([k for k,v in D.items() if v['source']=='10.1'],6))]:
 for i,ep in enumerate(eps):selection.append(dict(group=group,episode=ep,split='calibration' if i%2==0 else 'test_episode',pf=str(S/ep) if group=='sandwich' else D[ep]['pointflow'],raw=str(Path('/data/shichaojian/raw_data')/('singlerighthand_sandwich_100' if group=='sandwich' else 'dropper_dagger_mix')/ep)))

def stats(x):
 x=np.asarray(x);return dict(n=len(x),median=float(np.median(x)),p05=float(np.percentile(x,5)),p95=float(np.percentile(x,95)),mean=float(x.mean())) if len(x) else dict(n=0)
def worker(item):
 ep=item['episode'];raw=Path(item['raw']);pf=Path(item['pf']);out=B/'samples'/item['group']/ep;out.mkdir(parents=True,exist_ok=True)
 arrays={k:np.load(pf/(k+'.npy'),mmap_mode='r') for k in ['frame_offsets','frame_indices','obs_pos','obs_uv','obs_label','obs_valid','obs_unique','intrinsics']};T=len(arrays['frame_indices']);assert np.array_equal(arrays['frame_indices'],np.arange(T))
 meta=json.load(open(raw/'auxiliary_camera/metadata.json'));head=meta['capture_metadata']['cameras']['head'];streams=head['streams'];dk=streams['depth']['intrinsics'];ck=streams.get('color',{}).get('intrinsics');ci_source=ep
 if ck is None:
  for candidate in sorted(raw.parent.glob('episode_*')):
   try:h=json.load(open(candidate/'auxiliary_camera/metadata.json'))['capture_metadata']['cameras']['head']
   except (FileNotFoundError,KeyError):continue
   c=h.get('streams',{}).get('color',{}).get('intrinsics')
   if c and h.get('serial_number')==head.get('serial_number'):ck=c;ci_source=candidate.name;break
 assert ck is not None
 K=np.array([[ck['fx']*640/ck['width'],0,(ck['ppx']+.5)*640/ck['width']-.5],[0,ck['fy']*448/ck['height'],(ck['ppy']+.5)*448/ck['height']-.5],[0,0,1.]])
 assert np.max(np.abs(dk.get('coeffs',[0])))<1e-9 and np.max(np.abs(ck.get('coeffs',[0])))<1e-9
 e=streams['depth']['extrinsics_to_color'];R=np.array(e['rotation']).reshape(3,3,order='F');tr=np.array(e['translation']);scale=streams['depth']['depth_scale_m'];sync=json.load(open(raw/'sync_timestamps.json'));rgbts=np.array([r['camera_head'] for r in sync]);assert len(rgbts)==T
 env=lmdb.open(str(raw/'auxiliary_camera/depth.lmdb'),readonly=True,lock=False,readahead=False)
 matches=[];results=[];samples=[];rng=np.random.default_rng(17)
 with env.begin() as tx:
  ts=pickle.loads(tx.get(b'index/depth/head/timestamps'));keys=[k for k,_ in tx.cursor() if k.startswith(b'depth/head/')];ids=np.array([int(k.rsplit(b'/',1)[1]) for k in keys]);chosen=np.unique(np.linspace(0,len(ids)-1,24,dtype=int))
  for rank,idx in enumerate(chosen):
   depth_frame=int(ids[idx]);stamp=float(ts[depth_frame]);frame=int(np.argmin(abs(rgbts-stamp)));delta=float((rgbts[frame]-stamp)*1000)
   matches.append(dict(depth_frame=depth_frame,rgb_frame=frame,dt_ms=delta,index_offset=frame-depth_frame))
   if not np.isfinite(stamp) or abs(delta)>20:continue
   buf=tx.get(f'depth/head/{depth_frame:06d}'.encode());dep=cv2.imdecode(np.frombuffer(buf,np.uint8),cv2.IMREAD_UNCHANGED)*scale;yy,xx=np.indices(dep.shape);valid=(dep>.1)&(dep<3)
   z=dep[valid];p=np.stack([(xx[valid]-dk['ppx'])/dk['fx']*z,(yy[valid]-dk['ppy'])/dk['fy']*z,z],-1);p=p@R.T+tr;proj=p@K.T;uv=proj[:,:2]/proj[:,2:];ok=(p[:,2]>.1)&(uv[:,0]>=0)&(uv[:,0]<640)&(uv[:,1]>=0)&(uv[:,1]<448);p,uv=p[ok],uv[ok]
   # Colour-view z-buffer: keep the frontmost depth sample per rounded pixel.
   pix=np.rint(uv).astype(int);inside=(pix[:,0]<640)&(pix[:,1]<448);p,uv,pix=p[inside],uv[inside],pix[inside];flat=pix[:,1]*640+pix[:,0];order=np.lexsort((p[:,2],flat));_,first=np.unique(flat[order],return_index=True);keep=order[first];p,uv=p[keep],uv[keep];tree=cKDTree(uv)
   s,t=map(int,arrays['frame_offsets'][frame:frame+2]);good=arrays['obs_valid'][s:t]&arrays['obs_unique'][s:t];labels=arrays['obs_label'][s:t];pixels=arrays['obs_uv'][s:t];positions=arrays['obs_pos'][s:t]
   for cls in [2,3,4]:
    mask=np.zeros((448,640),np.uint8);ix=good&(labels==cls);mask[pixels[ix,1],pixels[ix,0]]=1;inner=cv2.erode(mask,np.ones((5,5),np.uint8));ix=ix&(inner[pixels[:,1],pixels[:,0]]>0)&(pixels[:,0]%2==0)&(pixels[:,1]%2==0);ids2=np.flatnonzero(ix)
    if len(ids2)>512:ids2=rng.choice(ids2,512,replace=False)
    if len(ids2)==0:continue
    query=np.asarray(pixels[ids2],float);pred=np.asarray(positions[ids2],float);distance,nn=tree.query(query,k=4,distance_upper_bound=1.25);nnok=np.isfinite(distance);safe=np.minimum(nn,len(p)-1);depths=np.where(nnok,p[safe,2],np.nan);count=nnok.sum(1)
    with np.errstate(all='ignore'):
     target=np.nanmedian(depths,axis=1);spread=np.nanmax(depths,axis=1)-np.nanmin(depths,axis=1)
    good2=(count>=2)&np.isfinite(target)&(spread<(.02+.02*target));query,pred,target=query[good2],pred[good2],target[good2]
    if not len(target):continue
    rays=np.c_[query,np.ones(len(query))]@np.linalg.inv(K).T;ref=rays*target[:,None]
    split='fit' if item['split']=='calibration' and rank<12 else ('test_time' if item['split']=='calibration' else 'test_episode')
    data=dict(frame=frame,depth_frame=depth_frame,dt_ms=delta,cls=cls,split=split,n=len(target),signed_z_mm=stats((pred[:,2]-target)*1000),raw_xyz_mm=stats(np.linalg.norm(pred-ref,axis=1)*1000));results.append(data)
    samples.append(dict(pred=pred,target=target,uv=query,ray=rays,frame=np.full(len(target),frame),cls=np.full(len(target),cls),split=np.full(len(target),{'fit':0,'test_time':1,'test_episode':2}[split])))
    if rank in [0,12] and cls==2:np.savez_compressed(out/f'preview_{frame:04d}.npz',uv=query,pred=pred,target=target,ray=rays)
 env.close()
 fields={k:np.concatenate([s[k] for s in samples]) for k in samples[0]} if samples else {}
 if fields:np.savez_compressed(out/'points.npz',**fields)
 summary=dict(**item,intrinsic_source=ci_source,K=K.tolist(),depth_rgb_matches=matches,accepted_sensor_frames=len(set(r['frame'] for r in results)),rows=results)
 (out/'report.json').write_text(json.dumps(summary,indent=2));return summary

if __name__=='__main__':
 (B/'selection.json').write_text(json.dumps(selection,indent=2))
 with ProcessPoolExecutor(max_workers=4) as pool:
  reports=[]
  for r in pool.map(worker,selection):reports.append(r);print(r['group'],r['episode'],r['accepted_sensor_frames'],'frames',flush=True)
 (B/'extraction_reports.json').write_text(json.dumps(reports,indent=2));print('EXTRACTION_COMPLETE',flush=True)
