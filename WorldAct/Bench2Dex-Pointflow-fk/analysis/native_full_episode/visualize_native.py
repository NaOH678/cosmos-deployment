from pathlib import Path
import json,os,subprocess
os.environ.setdefault('MPLCONFIGDIR','/tmp/bench2dex_mpl')
import numpy as np,cv2,h5py
import plotly.graph_objects as go
B=Path(__file__).resolve().parent;O=B/'validation'
report=json.loads((O/'report.json').read_text())
frames=[]
for r in report['samples']:
 t=r['frame'];a=np.load(O/f'surface_{t:04d}.npz')
 data=[]
 for key,color,label in [('mesh_xyz','#2474b5','FK visual hand surface'),('pred_xyz','#ef8525','Native final PointFlow')]:
  p=a[key];data.append(go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='markers',marker=dict(size=3,color=color),name=label))
 frames.append(go.Frame(name=str(t),data=data))
fig=go.Figure(data=frames[0].data,frames=frames)
fig.update_layout(title='Native full episode: same-pixel hand surface comparison (no fitted alignment)',
 scene=dict(xaxis_title='Camera X (m)',yaxis_title='Camera Y (m)',zaxis_title='Camera Z (m)',aspectmode='data'),
 sliders=[dict(steps=[dict(method='animate',args=[[f.name],dict(mode='immediate',frame=dict(duration=0,redraw=True),transition=dict(duration=0))],label=f.name) for f in frames],currentvalue=dict(prefix='Source frame: '))],
 annotations=[dict(text='Final SAM2 hand masks lose the right hand in some frames. Comparison covers retained pixels only.',xref='paper',yref='paper',x=.5,y=0,showarrow=False)],height=820)
fig.write_html(str(O/'native_hand_3d.html'),include_plotlyjs=True)
# Full 35.65s video: final labels alongside FK skeleton; no fake XYZ alignment.
d=B/'run/efep_seg_v61/task21_episode000000'
names=['frame_offsets','obs_uv','obs_pos','obs_label','obs_valid','obs_unique'];arr={k:np.load(d/(k+'.npy'),mmap_mode='r') for k in names}
g=np.load(B/'full_geometry.npz');links=g['link_names'].tolist();K=g['intrinsic'].copy();K[1]*=448/480;K[1,2]=(g['intrinsic'][1,2]+.5)*448/480-.5
proc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pixel_format','bgr24','-video_size','1280x448','-framerate','20','-i','-','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(O/'native_full_episode_review.mp4')],stdin=subprocess.PIPE)
with h5py.File('/tmp/bench2dex_replay21_ep0.hdf5') as f:
 for t in range(713):
  image=cv2.resize(cv2.imdecode(f['cameras/cam_overhead/rgb'][t],cv2.IMREAD_COLOR),(640,448));left=image.copy();right=image.copy()
  s,e=map(int,arr['frame_offsets'][t:t+2]);keep=arr['obs_valid'][s:e]&arr['obs_unique'][s:e];uv=arr['obs_uv'][s:e][keep];lab=arr['obs_label'][s:e][keep]
  for c,color in [(2,[255,160,20]),(3,[20,160,255]),(4,[0,200,235])]:
   xy=uv[lab==c];left[xy[:,1],xy[:,0]]=(.4*left[xy[:,1],xy[:,0]]+.6*np.array(color)).astype(np.uint8)
  for side,color in [('left',(255,180,0)),('right',(0,220,255))]:
   for finger in range(1,6):
    chain=[side+'_palm_link']+[f'{side}_finger{finger}_link{k}' for k in range(1,5)]+[f'{side}_finger{finger}_tip_link']
    p=g['fk_optical_m'][t,[links.index(k) for k in chain]];xy=p@K.T;xy=np.round(xy[:,:2]/xy[:,2:]).astype(int)
    for a,b in zip(xy[:-1],xy[1:]):cv2.line(right,tuple(a),tuple(b),color,1,cv2.LINE_AA)
  cv2.putText(left,f'Native final points | frame {t}',(10,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),2)
  cv2.putText(right,'FK skeleton | calibrated camera',(10,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),2)
  proc.stdin.write(np.concatenate([left,right],axis=1).tobytes())
proc.stdin.close();assert proc.wait()==0
print('Saved HTML and video',flush=True)
