from pathlib import Path
import json,numpy as np,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
B=Path(__file__).resolve().parent;EP='episode_0013_20260731_133649';video=Path('/data/shichaojian/raw_data/singlerighthand_sandwich_100')/EP/'videos/head.mp4'
report=json.load(open(B/'comparison.json'));frames=[0,650];fig=plt.figure(figsize=(18,13),constrained_layout=True);fig.set_constrained_layout_pads(h_pad=0.18,w_pad=0.1,hspace=0.12);gs=fig.add_gridspec(2,3);fig.suptitle('Real sandwich: before / after DA3 chunk-metadata fix | same pixels, no fitted alignment',fontsize=17);cap=cv2.VideoCapture(str(video))
for row,f in enumerate(frames):
 d=np.load(B/f'comparison_{f:04d}.npz');cloud=np.concatenate([d[k] for k in ['old','fixed','surface']]);lo=cloud.min(0)-.02;hi=cloud.max(0)+.02
 ax=fig.add_subplot(gs[row,0],projection='3d')
 for key,color,label in [('surface','#1f77b4','FK hand surface'),('old','#ff7f0e','Before (9.24 v2)'),('fixed','#16a34a','After fix')]:
  p=d[key];ax.scatter(p[:,0],p[:,1],p[:,2],s=6,c=color,label=label,depthshade=False)
 ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),xlabel='Camera X (m)',ylabel='Camera Y (m)',zlabel='Camera Z (m)');ax.set_box_aspect(hi-lo,zoom=0.8);ax.tick_params(labelsize=8);ax.view_init(22,-62);ax.set_title(f'Frame {f} | {len(d["uv"])} matched pixels');ax.legend(fontsize=8)
 cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,img=cap.read();assert ok;img=cv2.cvtColor(cv2.resize(img,(640,448)),cv2.COLOR_BGR2RGB);uv=d['uv'];x0=max(0,int(uv[:,0].min())-35);x1=min(640,int(uv[:,0].max())+36);y0=max(0,int(uv[:,1].min())-35);y1=min(448,int(uv[:,1].max())+36)
 for col,key,title in [(1,'old','Before'),(2,'fixed','After fix')]:
  err=np.linalg.norm(d[key]-d['surface'],axis=1)*1000;c=np.where(err<30,'#16a34a',np.where(err<100,'#f5ab00','#dc2626'));ax=fig.add_subplot(gs[row,col]);ax.imshow(img);ax.scatter(uv[:,0],uv[:,1],c=c,s=14,marker='D',linewidths=0);ax.set(xlim=(x0,x1),ylim=(y1,y0),title=f'{title}: median {np.median(err):.1f} mm');ax.axis('off')
cap.release();fig.supxlabel('RGB colours: green <30 mm | orange 30–100 mm | red ≥100 mm (3D surface discrepancy)',fontsize=12);fig.savefig(B/'before_after.png',dpi=150);plt.close(fig)
rows=report['comparison_frames'];fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
for key,label in [('old_xyz_mm','Before (9.24 v2)'),('fixed_xyz_mm','After fix')]:ax.plot([r['frame'] for r in rows],[r[key]['median'] for r in rows],'o-',label=label)
ax.set(xlabel='Frame',ylabel='Same-pixel median 3D surface gap (mm)',title='Sandwich 0013: matched surface correspondences');ax.legend();ax.grid(alpha=.2);fig.savefig(B/'error_comparison.png',dpi=160);plt.close(fig)
fig=go.Figure();steps=[]
for row in rows:
 f=row['frame'];d=np.load(B/f'comparison_{f:04d}.npz');start=len(fig.data)
 for key,color,label in [('surface','#1f77b4','FK hand surface'),('old','#ff7f0e','Before (9.24 v2)'),('fixed','#16a34a','After fix')]:
  p=d[key];fig.add_trace(go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='markers',marker=dict(size=2,color=color),name=label,visible=f==0))
 steps.append((f,start))
fig.update_layout(title='Sandwich: before / after fix vs FK surface',scene=dict(xaxis_title='X (m)',yaxis_title='Y (m)',zaxis_title='Z (m)',aspectmode='data'),sliders=[dict(steps=[dict(label=str(f),method='update',args=[{'visible':[start<=i<start+3 for i in range(len(fig.data))]},{'title':f'Frame {f}: blue FK / orange before / green after fix'}]) for f,start in steps])]);fig.write_html(B/'before_after_3d.html',include_plotlyjs=True)
print('VISUALS_COMPLETE')
