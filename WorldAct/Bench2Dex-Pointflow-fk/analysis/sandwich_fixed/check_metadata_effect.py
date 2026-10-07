from pathlib import Path
import json,numpy as np
B=Path(__file__).resolve().parent;REF=B.parent/'sandwich_real_audit';EP='episode_0013_20260731_133649'
s=np.load(B/'frame_metric_scales.npy').reshape(-1);f=np.load(B/'frame_normalized_focals.npy').reshape(-1)
rows=[]
for t in [0,128,300,500,650,800,1000,1191]:
 x=np.load(REF/f'mesh_surface_{t:04d}.npz');uv=x['uv'];new=np.load(B/'snapshots'/f'frame{t:04d}.npz')['points'][uv[:,1],uv[:,0]].astype(float)
 # Re-introduce only the original LAST-chunk metadata error, for diagnosis.
 reconstructed=new*(s[-1]/s[t]);reconstructed[:,:2]*=f[t]/f[-1]
 err=np.linalg.norm(reconstructed-x['pf'],axis=1)*1000
 rows.append(dict(frame=t,own_scale=float(s[t]),last_scale=float(s[-1]),own_focal=float(f[t]),last_focal=float(f[-1]),historical_vs_reintroduced_bug_xyz_mm=dict(median=float(np.median(err)),p95=float(np.percentile(err,95)),max=float(np.max(err))),historical_z_m=float(np.median(x['pf'][:,2])),fixed_z_m=float(np.median(new[:,2]))))
(B/'metadata_effect_check.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
