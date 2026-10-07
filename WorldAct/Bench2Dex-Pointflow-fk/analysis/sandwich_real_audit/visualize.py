from audit import *
import cv2,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
E=json.load(open(B/'episodes.json'));J=json.load(open(B/'joints.json'));F=json.load(open(B/'frames.json'))
fig,axs=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
dz=np.array([x['signed_depth_mm']/10 for x in J]);xyz=np.array([x['raw_xyz_gap_mm']/10 for x in J]);axs[0,0].hist(dz,bins=90,color='#e67e22');axs[0,0].axvline(0,color='k');axs[0,0].set(xlabel='PointFlow surface Z - FK joint Z (cm)',ylabel='Joint/frame samples',title=f'101 episodes, {len(J):,} matched samples; median {np.median(dz):.1f} cm')
med=np.array([x['signed_depth_mm']['median']/10 for x in E]);axs[0,1].plot(np.sort(med),'.');axs[0,1].axhline(0,color='k');axs[0,1].set(xlabel='Episode rank (sorted)',ylabel='Median signed depth difference (cm)',title='Every episode median is negative')
ep='episode_0013_20260731_133649';frames=[x for x in F if x['episode']==ep and x['signed_depth_mm'] is not None];axs[1,0].plot([x['frame'] for x in frames],[x['signed_depth_mm']/10 for x in frames],'.-')
for x in range(128,1192,128):axs[1,0].axvline(x,color='gray',alpha=.25)
axs[1,0].axhline(0,color='k');axs[1,0].set(xlabel='Frame',ylabel='Median signed depth difference (cm)',title='Episode 0013 (July 31); grey = 128-frame boundaries')
axs[1,1].scatter([x['fx_pred_px'] for x in E],med,s=18);axs[1,1].axvline(E[0]['fx_true_px'],color='k',ls='--',label='D435 colour fx');axs[1,1].set(xlabel='PointFlow fx (pixels at width 640)',ylabel='Episode median depth difference (cm)',title='Estimated focal differs from calibrated camera');axs[1,1].legend()
fig.suptitle('Sandwich v2: modality discrepancy, NOT pure surface reconstruction error',fontsize=14);fig.savefig(B/'summary.png',dpi=160);plt.close(fig)
# Same episode through time: RGB supports spatial correspondence; XZ shows discrepancy.
a,cam,K,src,n=load(ep);cap=cv2.VideoCapture(str(RAW/ep/'videos/head.mp4'));chosen=[0,128,300,500,650,800];fig,axes=plt.subplots(6,2,figsize=(13,20),constrained_layout=True);interactive=go.Figure();steps=[]
edges=[(0,b) for b in [1,5,9,13,17]]+[(b+i,b+i+1) for b in [1,5,9,13,17] for i in range(3)]
for idx,f in enumerate(chosen):
 s,e=map(int,a['frame_offsets'][f:f+2]);m=(a['obs_label'][s:e]==2)&a['obs_valid'][s:e]&a['obs_unique'][s:e];p=np.asarray(a['obs_pos'][s:e][m]);uv=np.asarray(a['obs_uv'][s:e][m]);c=cam[f];q=c@K.T;q=q[:,:2]/q[:,2:]
 cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,img=cap.read();assert ok;img=cv2.cvtColor(cv2.resize(img,(640,448)),cv2.COLOR_BGR2RGB);ax=axes[idx,0];ax.imshow(img);ax.scatter(uv[::4,0],uv[::4,1],s=.4,c='#16a085',alpha=.35)
 for u,v in edges:ax.plot(q[[u,v],0],q[[u,v],1],color='#ff3030',lw=1)
 ax.scatter(q[:,0],q[:,1],s=6,c='#ff3030');ax.set(xlim=(0,640),ylim=(448,0),title=f'Frame {f}: RGB + hand pixels (green) + calibrated FK (red)');ax.axis('off')
 ax=axes[idx,1];ax.scatter(p[::3,0],p[::3,2],s=1,c='#e67e22',label='PointFlow surface')
 for u,v in edges:ax.plot(c[[u,v],0],c[[u,v],2],color='#1565c0')
 ax.scatter(c[:,0],c[:,2],s=10,c='#1565c0',label='FK joints');ax.set(xlabel='Camera X (m)',ylabel='Camera Z / depth (m)',title='Same camera coordinates; no fitted transform');ax.axis('equal');ax.legend(loc='best',fontsize=8)
 start=len(interactive.data);interactive.add_trace(go.Scatter3d(x=p[::3,0],y=p[::3,1],z=p[::3,2],mode='markers',marker=dict(size=2,color='#e67e22'),name='PointFlow hand surface',visible=idx==0))
 sk=[]
 for u,v in edges:sk.extend([c[u],c[v],[None,None,None]])
 sk=np.array(sk,dtype=object);interactive.add_trace(go.Scatter3d(x=sk[:,0],y=sk[:,1],z=sk[:,2],mode='lines+markers',line=dict(color='#1565c0',width=5),marker=dict(size=3,color='#1565c0'),name='FK joints',visible=idx==0));steps.append((f,start))
cap.release();fig.savefig(B/'rgb_and_depth_comparison.jpg',dpi=130);plt.close(fig)
interactive.update_layout(title='Sandwich 0013 (July 31): PointFlow vs FK, metres, no alignment fitting',scene=dict(xaxis_title='camera X (m)',yaxis_title='camera Y (m)',zaxis_title='camera Z (m)',aspectmode='data'),sliders=[dict(steps=[dict(label=str(f),method='update',args=[{'visible':[i in [start,start+1] for i in range(len(interactive.data))]},{'title':f'Sandwich 0013 frame {f}: orange PointFlow surface / blue FK joints'}]) for f,start in steps])],margin=dict(l=0,r=0,b=0,t=60));interactive.write_html(B/'pointflow_vs_fk_3d.html',include_plotlyjs=True)
print('visuals complete')
