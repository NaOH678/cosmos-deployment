from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parent
EP='episode_0013_20260731_133649'
video=Path('/data/shichaojian/raw_data/singlerighthand_sandwich_100')/EP/'videos/head.mp4'
frames=[0,650]
data=[np.load(B/f'mesh_surface_{f:04d}.npz') for f in frames]
cloud=np.concatenate([d[k] for d in data for k in ['pf','surface']]);lo=cloud.min(0)-.025;hi=cloud.max(0)+.025
fig=plt.figure(figsize=(16,12),constrained_layout=True)
gs=fig.add_gridspec(2,2,height_ratios=[1.1,1])
fig.suptitle('Real sandwich: raw PointFlow vs FK-posed hand surface | no scale / rigid fitting\nEpisode 0013 (2026-07-31), existing 9.24 v2 data',fontsize=17)
cap=cv2.VideoCapture(str(video))
for col,(f,d) in enumerate(zip(frames,data)):
 p,s,uv=d['pf'],d['surface'],d['uv'];err=np.linalg.norm(p-s,axis=1)*1000
 ax=fig.add_subplot(gs[0,col],projection='3d');ax.scatter(s[:,0],s[:,1],s[:,2],s=9,c='#1f77b4',label='FK hand surface',depthshade=False);ax.scatter(p[:,0],p[:,1],p[:,2],s=9,c='#ff7f0e',label='PointFlow',depthshade=False)
 ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),xlabel='Camera X (m)',ylabel='Camera Y (m)',zlabel='Camera Z (m)');ax.set_box_aspect(hi-lo);ax.view_init(elev=22,azim=-62);ax.set_title(f'Frame {f} | median 3D gap {np.median(err):.1f} mm',fontsize=15);ax.legend(loc='upper right',fontsize=11)
 cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=cap.read();assert ok
 # Saved correspondences use 640x448; map pixel centres back to original RGB.
 rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);h,w=rgb.shape[:2];u=(uv[:,0]+.5)*w/640-.5;v=(uv[:,1]+.5)*h/448-.5
 colors=np.where(err<30,'#16a34a',np.where(err<100,'#f5ab00','#dc2626'))
 ax=fig.add_subplot(gs[1,col]);ax.imshow(rgb);ax.scatter(u,v,c=colors,s=9,marker='D',linewidths=0);ax.set_xlim(0,w);ax.set_ylim(h,0);ax.axis('off');ax.set_title('Surface gap: green <30 mm, orange 30–100 mm, red ≥100 mm',fontsize=11)
 # Enlarged hand inset preserves a full-scene view while making point colours readable.
 x0=max(0,int(u.min())-15);x1=min(w,int(u.max())+16);y0=max(0,int(v.min())-15);y1=min(h,int(v.max())+16)
 ins=ax.inset_axes([.025,.025,.37,.40]);ins.imshow(rgb);ins.scatter(u,v,c=colors,s=14,marker='D',linewidths=0);ins.set_xlim(x0,x1);ins.set_ylim(y1,y0);ins.set_xticks([]);ins.set_yticks([]);ins.set_title('Hand detail',fontsize=9)
 for sp in ins.spines.values():sp.set_edgecolor('white');sp.set_linewidth(2)
cap.release()
fig.savefig(B/'hand_geometry_visualization.png',dpi=160)
fig.savefig(B/'hand_geometry_visualization.jpg',dpi=150)
print(B/'hand_geometry_visualization.png')
