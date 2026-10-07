from audit import *
import cv2,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
inventory=json.load(open(B/'inventory.json'))
selected=['episode_0002_20260830_155725','episode_0005_20260903_185554']
for ep in selected:
 a,cam,K,src,n=load(ep);frames=[0,650];ds=[np.load(B/'surfaces'/ep/f'frame{f:04d}.npz') for f in frames];cloud=np.concatenate([d[k] for d in ds for k in ['pf','surface']]);lo=cloud.min(0)-.025;hi=cloud.max(0)+.025
 fig=plt.figure(figsize=(16,12),constrained_layout=True);gs=fig.add_gridspec(2,2,height_ratios=[1.1,1]);fig.suptitle(f'DAgger: raw PointFlow vs FK-posed hand surface | no scale / rigid fitting\n{ep} | source {inventory[ep]["source"]}',fontsize=16);cap=cv2.VideoCapture(str(RAW/ep/'videos/head.mp4'))
 for col,(f,d) in enumerate(zip(frames,ds)):
  p,s,uv=d['pf'],d['surface'],d['uv'];err=np.linalg.norm(p-s,axis=1)*1000
  ax=fig.add_subplot(gs[0,col],projection='3d')
  for pts,color,label in [(s,'#1f77b4','FK hand surface'),(p,'#ff7f0e','PointFlow')]:ax.scatter(pts[:,0],pts[:,1],pts[:,2],s=9,c=color,label=label,depthshade=False)
  ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),xlabel='Camera X (m)',ylabel='Camera Y (m)',zlabel='Camera Z (m)');ax.set_box_aspect(hi-lo,zoom=.9);ax.view_init(22,-62);ax.set_title(f'Frame {f} | median 3D gap {np.median(err):.1f} mm | {len(err)} points',fontsize=14);ax.legend(fontsize=10)
  cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,img=cap.read();assert ok;img=cv2.cvtColor(cv2.resize(img,(640,448)),cv2.COLOR_BGR2RGB);c=np.where(err<30,'#16a34a',np.where(err<100,'#f5ab00','#dc2626'));ax=fig.add_subplot(gs[1,col]);ax.imshow(img);ax.scatter(uv[:,0],uv[:,1],c=c,s=10,marker='D',linewidths=0);ax.set(xlim=(0,640),ylim=(448,0));ax.axis('off');ax.set_title('Surface gap: green <30 mm, orange 30–100 mm, red ≥100 mm',fontsize=11)
  # All FK landmarks are shown separately as thin white crosses for projection QA.
  q=cam[f]@K.T;q=q[:,:2]/q[:,2:];ax.scatter(q[:,0],q[:,1],c='white',s=14,marker='+',linewidths=.6)
 cap.release();fig.savefig(B/f'{ep}_geometry.png',dpi=150);plt.close(fig)
# interactive view includes all saved sample frames for both representative episodes
fig=go.Figure();steps=[]
for ep in selected:
 for path in sorted((B/'surfaces'/ep).glob('frame*.npz')):
  f=int(path.stem[5:]);d=np.load(path);start=len(fig.data)
  for key,color,label in [('surface','#1f77b4','FK hand surface'),('pf','#ff7f0e','PointFlow'),('fk','#333333','FK joints')]:
   p=d[key];fig.add_trace(go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='markers',marker=dict(size=3 if key=='fk' else 2,color=color),name=label,visible=start==0))
  steps.append((f'{inventory[ep]["source"]} ep{ep[8:12]} f{f}',start))
fig.update_layout(title='DAgger: PointFlow vs supplied-URDF FK surface',scene=dict(xaxis_title='Camera X (m)',yaxis_title='Camera Y (m)',zaxis_title='Camera Z (m)',aspectmode='data'),sliders=[dict(steps=[dict(label=label,method='update',args=[{'visible':[start<=i<start+3 for i in range(len(fig.data))]},{'title':label+' | blue FK surface / orange PointFlow'}]) for label,start in steps])]);fig.write_html(B/'geometry_3d.html',include_plotlyjs=True)
print('visuals complete')
