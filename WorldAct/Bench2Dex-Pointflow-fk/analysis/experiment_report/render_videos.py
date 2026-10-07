from pathlib import Path
import json,runpy,subprocess
from concurrent.futures import ProcessPoolExecutor
import numpy as np,cv2
B=Path(__file__).resolve().parent;A=B.parent
specs=[('sandwich','episode_0013_20260731_133649','sandwich_review'),('dagger','episode_0002_20260830_155725','dagger_924_review'),('dagger','episode_0005_20260903_185554','dagger_101_review')]
edges=[(0,b) for b in [1,5,9,13,17]]+[(b+i,b+i+1) for b in [1,5,9,13,17] for i in range(3)]
def worker(spec):
 group,ep,name=spec;ns=runpy.run_path(str(A/('sandwich_real_audit' if group=='sandwich' else 'dagger_real_audit')/'audit.py'));arr,cam,K,_,_=ns['load'](ep);raw=ns['RAW'];cap=cv2.VideoCapture(str(raw/ep/'videos/head.mp4'));T=len(cam);out=B/'media'/(name+'.mp4')
 cmd=['ffmpeg','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','bgr24','-s','1280x512','-r','10','-i','-','-an','-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(out)];proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
 fixed=None
 if group=='sandwich':
  d=A/'sandwich_fixed/run/efep_seg_fixed'/ep;fixed={k:np.load(d/(k+'.npy'),mmap_mode='r') for k in ['frame_offsets','obs_pos','obs_uv','obs_label','obs_valid','obs_unique']}
 def points(a,f):
  s,e=map(int,a['frame_offsets'][f:f+2]);uv=a['obs_uv'][s:e];ok=(a['obs_label'][s:e]==2)&a['obs_valid'][s:e]&a['obs_unique'][s:e]&(uv[:,0]%3==0)&(uv[:,1]%3==0);return np.asarray(a['obs_pos'][s:e][ok]),np.asarray(uv[ok])
 def side(p):return np.c_[690+(p[:,0]+.3)/.9*530,450-(p[:,2]-.3)/.95*350]
 rendered=0
 for f in range(T):
  ok,img=cap.read()
  if not ok:raise RuntimeError(f'video ended {f}/{T}')
  if f%3:continue
  canvas=np.full((512,1280,3),250,np.uint8);canvas[32:512,:640]=cv2.resize(img,(640,480));p,uv=points(arr,f)
  for u,v in uv:cv2.circle(canvas,(int(u),32+round((int(v)+.5)*480/448-.5)),1,(0,150,255),-1)
  q=cam[f]@K.T;q=q[:,:2]/q[:,2:];q[:,1]=(q[:,1]+.5)*480/448-.5+32
  for u,v in edges:
   if np.all(q[[u,v],0]>=0)&np.all(q[[u,v],0]<640)&np.all(q[[u,v],1]>=32)&np.all(q[[u,v],1]<512):cv2.line(canvas,tuple(q[u].astype(int)),tuple(q[v].astype(int)),(210,75,20),1,cv2.LINE_AA)
  for x in [-.2,0,.2,.4]:
   uvv=side(np.array([[x,0,.3],[x,0,1.2]]));cv2.line(canvas,tuple(uvv[0].astype(int)),tuple(uvv[1].astype(int)),(225,225,225),1);cv2.putText(canvas,f'{x:.1f}',(int(uvv[0,0])-10,472),0,.4,(90,90,90),1)
  for z in [.4,.6,.8,1.0,1.2]:
   uvv=side(np.array([[-.3,0,z],[.6,0,z]]));cv2.line(canvas,tuple(uvv[0].astype(int)),tuple(uvv[1].astype(int)),(225,225,225),1);cv2.putText(canvas,f'{z:.1f}',(648,int(uvv[0,1])+4),0,.4,(90,90,90),1)
  sets=[(p,(0,150,255))]
  if fixed is not None:sets.append((points(fixed,f)[0],(40,160,40)))
  for pts,color in sets:
   for x,y in side(pts):
    if 690<=x<1220 and 90<=y<450:cv2.circle(canvas,(round(x),round(y)),1,color,-1)
  sk=side(cam[f])
  for u,v in edges:cv2.line(canvas,tuple(sk[u].astype(int)),tuple(sk[v].astype(int)),(210,75,20),2,cv2.LINE_AA)
  cv2.putText(canvas,f'{group} ep{ep[8:12]} | frame {f}/{T-1} | t={f/30:.2f}s',(12,22),0,.62,(35,35,35),1,cv2.LINE_AA)
  cv2.putText(canvas,'X-Z view (metres): orange=existing PF, blue=FK JOINTS',(650,24),0,.48,(35,35,35),1,cv2.LINE_AA)
  if fixed is not None:cv2.putText(canvas,'green=chunk-fix rerun',(650,48),0,.48,(40,140,40),1,cv2.LINE_AA)
  cv2.putText(canvas,'Camera X (m)',(890,498),0,.5,(70,70,70),1);proc.stdin.write(canvas.tobytes());rendered+=1
 cap.release();proc.stdin.close();assert proc.wait()==0
 return dict(file=str(out),group=group,episode=ep,source_frames=T,rendered_frames=rendered,fps=10,source_stride=3,note='Every third real RGB frame shown at10fps, original speed; no point interpolation. Blue is FK21 joint skeleton, not mesh-surface error ground truth. Orange existing pointflow; sandwich green fixed rerun.')
if __name__=='__main__':
 with ProcessPoolExecutor(max_workers=3) as pool:results=list(pool.map(worker,specs))
 (B/'video_manifest.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
