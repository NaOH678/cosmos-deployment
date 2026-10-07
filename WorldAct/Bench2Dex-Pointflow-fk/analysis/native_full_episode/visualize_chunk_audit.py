from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/bench2dex_mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
B=Path(__file__).resolve().parent;O=B/'validation';r=json.loads((O/'chunk_recovery_audit.json').read_text())
fig,ax=plt.subplots(1,3,figsize=(16,5))
for col,t in enumerate([0,348]):
 a=np.load(O/f'chunk_corrected_{t:04d}.npz')
 for key,color,label in [('mesh_xyz','#2474b5','FK hand surface'),('native_xyz','#ef8525','Native output'),('corrected_xyz','#2d9a54','Chunk metadata correction')]:
  p=a[key];ax[col].scatter(p[:,0],p[:,2],s=5,c=color,label=label,alpha=.65)
 ax[col].set(title=f'Frame {t}: camera X-Z',xlabel='Camera X (m)',ylabel='Camera Z (m)');ax[col].axis('equal');ax[col].legend(fontsize=8)
ss=r['samples'];xx=[s['frame'] for s in ss]
ax[2].plot(xx,[s['native_xyz_mm']['median'] for s in ss],'-o',c='#ef8525',label='Native output')
ax[2].plot(xx,[s['chunk_metadata_corrected_xyz_mm']['median'] for s in ss],'-o',c='#2d9a54',label='Chunk metadata correction')
ax[2].set(xlabel='Source frame',ylabel='Median XYZ error (mm)',title='Same pixels, no fitted scale / extrinsics');ax[2].legend(fontsize=8)
fig.suptitle('713-frame native run: incorrect shared metadata confirmed; offline diagnostic, not a production fix')
fig.tight_layout();fig.savefig(O/'chunk_recovery_comparison.png',dpi=160)
frames=[]
for s in ss:
 t=s['frame'];a=np.load(O/f'chunk_corrected_{t:04d}.npz');data=[]
 for key,color,label in [('mesh_xyz','#2474b5','FK visual hand surface'),('native_xyz','#ef8525','Native final PointFlow'),('corrected_xyz','#2d9a54','Offline chunk metadata correction')]:
  p=a[key];data.append(go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='markers',marker=dict(size=3,color=color),name=label))
 frames.append(go.Frame(name=str(t),data=data))
fig=go.Figure(data=frames[0].data,frames=frames)
fig.update_layout(title='Full episode: native output and chunk-metadata diagnostic (no FK fitting)',scene=dict(xaxis_title='Camera X (m)',yaxis_title='Camera Y (m)',zaxis_title='Camera Z (m)',aspectmode='data'),height=820,
 sliders=[dict(steps=[dict(method='animate',args=[[f.name],dict(mode='immediate',frame=dict(duration=0,redraw=True),transition=dict(duration=0))],label=f.name) for f in frames],currentvalue=dict(prefix='Frame: '))],
 annotations=[dict(text='Retained hand pixels only. Right hand masks fail in some frames; residual errors remain.',xref='paper',yref='paper',x=.5,y=0,showarrow=False)])
fig.write_html(str(O/'chunk_recovery_3d.html'),include_plotlyjs=True)
summary=dict(status='Formal native full episode completed; geometry quality not passed.',episode='task21 episode000000',frames=713,fps=20,duration_seconds=35.65,
 formal_pipeline=['native run_efep_export: full sequence, no application truncation/chunks','native sam2_segment: white hand prompts, refined anchor boxes','native efep_seg_label: fill-gaps3, despike0.03','native drop_detached: defaults'],
 findings=['Native DA3 internal 128-frame batches overwrite scale and focal; final conversion uses last batch metadata for all frames.',
 'Saved normalized-depth probe times last batch scale reproduces tested native output depth exactly.',
 'SAM2 right-hand coverage fails in many frames; data not suitable for batch training yet.'],
 pooled_interior_raw_median_mm=r['pooled_interior_native_xyz_mm']['median'],pooled_interior_diagnostic_median_mm=r['pooled_interior_chunk_corrected_xyz_mm']['median'],
 diagnostic_p95_mm=r['pooled_interior_chunk_corrected_xyz_mm']['p95'],shared_native_code_modified=False,
 gpu_tasks_complete=True,production_fix_applied=False,
 next_steps=['Preserve per-block/per-frame scale and intrinsics in model output and revalidate motion across internal batch boundaries.',
 'Repair SAM2 hand visibility/coverage with additional verified prompts.', 'Re-evaluate against independent rendered depth before attributing residuals to simulation domain shift.'])
(B/'completion_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
