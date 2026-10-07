from pathlib import Path
import subprocess,json,hashlib
import h5py,cv2,numpy as np
out=Path(__file__).resolve().parent
src=Path('/tmp/bench2dex_replay21_ep0.hdf5')
video=out/'source/task21_episode000000/videos/head.mp4'
with h5py.File(src) as f:
 n=len(f['robot/qpos']);fps=float(f['meta/effective_fps'][()])
 proc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pixel_format','bgr24','-video_size','640x480','-framerate',str(fps),'-i','-','-c:v','libx264rgb','-crf','0','-preset','fast',str(video)],stdin=subprocess.PIPE)
 for i in range(n):
  im=cv2.imdecode(f['cameras/cam_overhead/rgb'][i],cv2.IMREAD_COLOR)
  assert im is not None
  proc.stdin.write(im.tobytes())
 proc.stdin.close();assert proc.wait()==0
 cap=cv2.VideoCapture(str(video));assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==n
 for i in [0,348,n-1]:
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,got=cap.read();assert ok
  expected=cv2.imdecode(f['cameras/cam_overhead/rgb'][i],cv2.IMREAD_COLOR)
  assert np.array_equal(got,expected), (i,np.abs(got.astype(float)-expected).max())
 cap.release()
 report=dict(source=str(src),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),video=str(video),frames=n,fps=fps,seconds=n/fps,camera='cam_overhead',encoding='lossless libx264rgb crf0',decode_exact_frames=[0,348,n-1],truncated=False)
(out/'input_manifest.json').write_text(json.dumps(report,indent=2));print(report)
