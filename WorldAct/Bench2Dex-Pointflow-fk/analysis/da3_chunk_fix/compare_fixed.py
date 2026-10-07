"""Matched-pixel comparison; avoids attributing changed filtering to geometry."""
from pathlib import Path
import json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/bench2dex_mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
B=Path(__file__).resolve().parent;OLD=B.parent/'native_full_episode';O=B/'validation'
def st(x):
 a=np.asarray(x);a=a[np.isfinite(a)]
 return dict(n=int(a.size),median=float(np.median(a)),p95=float(np.percentile(a,95)),mean=float(np.mean(a))) if len(a) else dict(n=0)
result={'samples':[]};before=[];after=[];frames=[]
for p in sorted(O.glob('surface_*.npz')):
 t=int(p.stem.split('_')[1]);new=np.load(p);old=np.load(OLD/'validation'/p.name)
 oldkey=old['v']*640+old['u'];newkey=new['v']*640+new['u'];key,i,j=np.intersect1d(oldkey,newkey,return_indices=True)
 gt=old['mesh_xyz'][i];np.testing.assert_allclose(gt,new['mesh_xyz'][j],atol=1e-7)
 a=old['pred_xyz'][i];b=new['pred_xyz'][j];inside=old['interior'][i]&new['interior'][j]
 ea=np.linalg.norm(a-gt,axis=1)*1000;eb=np.linalg.norm(b-gt,axis=1)*1000
 result['samples'].append(dict(frame=t,matched_pixels=len(key),native_error_mm=st(ea),fixed_error_mm=st(eb),interior_native_mm=st(ea[inside]),interior_fixed_mm=st(eb[inside])))
 before.extend(ea[inside]);after.extend(eb[inside])
 data=[]
 for pos,color,name in [(gt,'#2474b5','FK visual surface'),(a,'#ef8525','Native baseline'),(b,'#2d9a54','Fixed full rerun')]:
  data.append(go.Scatter3d(x=pos[:,0],y=pos[:,1],z=pos[:,2],mode='markers',marker=dict(size=3,color=color),name=name))
 frames.append(go.Frame(name=str(t),data=data))
result['pooled_matched_interior_native_mm']=st(before);result['pooled_matched_interior_fixed_mm']=st(after)
result['limitations']=['Same episode only; inherited SAM2 right-hand coverage failures remain.', 'FK visual meshes are geometric references, not independent rendered depth.', 'Retains native per-block mean focal convention; no per-frame DA3 K recalibration.', 'World pose registration across DA3 blocks is not performed.']
(O/'matched_comparison.json').write_text(json.dumps(result,indent=2))
fig=go.Figure(data=frames[0].data,frames=frames)
fig.update_layout(title='Full episode rerun: matched pixels, unchanged masks and model weights',scene=dict(xaxis_title='Camera X (m)',yaxis_title='Camera Y (m)',zaxis_title='Camera Z (m)',aspectmode='data'),height=820,
 sliders=[dict(steps=[dict(method='animate',args=[[f.name],dict(mode='immediate',frame=dict(duration=0,redraw=True),transition=dict(duration=0))],label=f.name) for f in frames],currentvalue=dict(prefix='Frame: '))])
fig.write_html(str(O/'fixed_vs_native_3d.html'),include_plotlyjs=True)
samples=result['samples'];xx=[s['frame'] for s in samples]
fig,ax=plt.subplots(figsize=(10,4))
ax.plot(xx,[s['native_error_mm']['median'] for s in samples],'-o',label='Native baseline',c='#ef8525')
ax.plot(xx,[s['fixed_error_mm']['median'] for s in samples],'-o',label='Fixed full rerun',c='#2d9a54')
ax.set(xlabel='Frame',ylabel='Matched-pixel median XYZ error (mm)',title='No fitted scale or extrinsics; identical episode and SAM2 masks');ax.legend();fig.tight_layout();fig.savefig(O/'fixed_vs_native.png',dpi=160)
print(json.dumps(result,indent=2))
