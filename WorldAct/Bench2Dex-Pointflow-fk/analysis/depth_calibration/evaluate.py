from pathlib import Path
import json,numpy as np
from scipy.optimize import least_squares
B=Path(__file__).resolve().parent
selection=json.load(open(B/'selection.json'))
def stat(a):
 a=np.asarray(a);return dict(n=len(a),median=float(np.median(a)),p95=float(np.percentile(a,95)),mean=float(np.mean(a))) if len(a) else dict(n=0)
def fit(x,y,w,kind):
 if kind=='raw':return [1.,0.]
 if kind=='offset':
  r=least_squares(lambda p:(x+p[0]-y)*w,[0.],loss='soft_l1',f_scale=.02);return [1.,float(r.x[0])]
 if kind=='scale':
  r=least_squares(lambda p:(x*p[0]-y)*w,[1.],bounds=(.01,np.inf),loss='soft_l1',f_scale=.02);return [float(r.x[0]),0.]
 r=least_squares(lambda p:(x*p[0]+p[1]-y)*w,[1.,0.],bounds=([.01,-np.inf],[np.inf,np.inf]),loss='soft_l1',f_scale=.02);return list(map(float,r.x))
allresults=[];modelrecords=[];frame_results=[];timechecks=[]
for group in ['sandwich','dagger']:
 eps=[s for s in selection if s['group']==group];loaded={}
 for item in eps:
  ep=item['episode'];d=np.load(B/'samples'/group/ep/'points.npz');loaded[ep]={k:d[k] for k in d.files}
  rep=json.load(open(B/'samples'/group/ep/'report.json'));timechecks.extend([dict(group=group,episode=ep,**r) for r in rep['depth_rgb_matches']])
 for fit_scope in ['all_classes','hand_only']:
  xs=[];ys=[];ws=[]
  for ep,d in loaded.items():
   for f in np.unique(d['frame'][d['split']==0]):
    for cls in ([2,3,4] if fit_scope=='all_classes' else [2]):
     keep=(d['split']==0)&(d['frame']==f)&(d['cls']==cls);idx=np.flatnonzero(keep)[::max(1,int(keep.sum())//128)]
     if len(idx)<10:continue
     xs.extend(d['pred'][idx,2]);ys.extend(d['target'][idx]);ws.extend(np.full(len(idx),1/len(idx)))
  x,y,w=np.asarray(xs),np.asarray(ys),np.asarray(ws);w=np.sqrt(w/w.mean())
  for kind in ['raw','offset','scale','affine']:
   a,b=fit(x,y,w,kind);modelrecords.append(dict(group=group,fit_scope=fit_scope,model=kind,scale=a,offset_m=b,fit_points=len(x)))
   collected={}
   for ep,d in loaded.items():
    z=d['pred'][:,2]*a+b;original_ray=d['pred']/d['pred'][:,2,None];metric_ref=d['ray']*d['target'][:,None];xyz_native=original_ray*z[:,None];xyz_known=d['ray']*z[:,None]
    for split_id,split in [(0,'fit'),(1,'test_time'),(2,'test_episode')]:
     for cls in [2,3,4]:
      m=(d['split']==split_id)&(d['cls']==cls)
      if not m.any():continue
      dz=(z[m]-d['target'][m])*1000;native=np.linalg.norm(xyz_native[m]-metric_ref[m],axis=1)*1000;known=np.linalg.norm(xyz_known[m]-metric_ref[m],axis=1)*1000
      row=dict(group=group,fit_scope=fit_scope,model=kind,episode=ep,split=split,cls=cls,depth_abs_mm=stat(abs(dz)),depth_signed_mm=stat(dz),native_ray_xyz_mm=stat(native),known_K_xyz_mm=stat(known));allresults.append(row)
      for f in np.unique(d['frame'][m]):
       v=d['frame'][m]==f
       frame_results.append(dict(group=group,fit_scope=fit_scope,model=kind,episode=ep,split=split,cls=cls,frame=int(f),n=int(v.sum()),depth_abs_median_mm=float(np.median(abs(dz[v]))),signed_depth_median_mm=float(np.median(dz[v])),native_xyz_median_mm=float(np.median(native[v])),known_K_xyz_median_mm=float(np.median(known[v])),depth_abs_p95_mm=float(np.percentile(abs(dz[v]),95))))
aggregate=[]
for group in ['sandwich','dagger']:
 for scope in ['all_classes','hand_only']:
  for kind in ['raw','offset','scale','affine']:
   for split in ['fit','test_time','test_episode']:
    for cls in [2,3,4]:
     r=[v for v in allresults if (v['group'],v['fit_scope'],v['model'],v['split'],v['cls'])==(group,scope,kind,split,cls)]
     if not r:continue
     aggregate.append(dict(group=group,fit_scope=scope,model=kind,split=split,cls=cls,episodes=len(r),episode_median_abs_depth_mm=stat([v['depth_abs_mm']['median'] for v in r]),episode_median_native_xyz_mm=stat([v['native_ray_xyz_mm']['median'] for v in r]),episode_median_knownK_xyz_mm=stat([v['known_K_xyz_mm']['median'] for v in r]),episode_p95_abs_depth_mm=stat([v['depth_abs_mm']['p95'] for v in r])))
summary=dict(selection=selection,calibration='6 calibration episodes per dataset: first 12 sampled sensor frames fit, later frames test_time; 6 completely held-out episodes test_episode. DAgger stratified 9.24/10.1. Fixed seed17; no test-based parameter fitting.',fit='Weighted robust least squares (soft_l1, 20mm), equal weighting per frame/class fit group; fit scopes all_classes or hand_only.',reference='D435 depth transformed to colour using column-major extrinsics, z-buffer and calibrated K; closest recorded colour timestamp <=20ms. Not perfect ground truth.',timestamp_abs_delta_ms=stat([abs(r['dt_ms']) for r in timechecks if np.isfinite(r['dt_ms'])]),timestamp_rejected=sum(abs(r['dt_ms'])>20 or not np.isfinite(r['dt_ms']) for r in timechecks),timestamp_index_offset_counts={str(k):sum(r['index_offset']==k for r in timechecks) for k in sorted(set(r['index_offset'] for r in timechecks))},models=modelrecords,aggregate=aggregate,limitations=['D435 black-glove/reflective/thin-object holes and noise mean reference is imperfect.','Mask interiors and local depth smoothness gates preferentially retain easier pixels; no rejection based on PF residual.','No interpolation across time, no FK-based fitting, no source data modification.','Results are for existing pointflow, not the isolated chunk-fix rerun.','Known-K XYZ diagnostic changes rays as well as corrected depth; native-ray result isolates only depth correction.'])
for name,data in [('results',summary),('episode_results',allresults),('frame_results',frame_results),('timestamp_checks',timechecks)]: (B/(name+'.json')).write_text(json.dumps(data,indent=2))
for r in aggregate:
 if r['split']=='test_episode' and r['fit_scope']=='all_classes':print(r['group'],r['model'],r['cls'],'depth',r['episode_median_abs_depth_mm']['median'],'XYZnative',r['episode_median_native_xyz_mm']['median'],'XYZknown',r['episode_median_knownK_xyz_mm']['median'],flush=True)
print('EVALUATION_COMPLETE')
