import json,lmdb
from pathlib import Path
B=Path(__file__).resolve().parent;RAW=Path('/data/shichaojian/raw_data/dropper_dagger_mix');records=[]
for ep in ['episode_0002_20260830_155725','episode_0005_20260903_185554']:
 env=lmdb.open(str(RAW/ep/'auxiliary_camera/depth.lmdb'),readonly=True,lock=False,readahead=False)
 with env.begin() as tx:
  keys=[k for k,_ in tx.cursor() if k.startswith(b'depth/head/')]
  for frame in [0,650]:
   if not keys:records.append(dict(episode=ep,frame=frame,error='no head depth'));continue
   key=min(keys,key=lambda k:abs(int(k.rsplit(b'/',1)[1])-frame));actual=int(key.rsplit(b'/',1)[1]);path=B/(ep+f'_depth{frame:04d}.png');path.write_bytes(tx.get(key));records.append(dict(episode=ep,frame=frame,depth_frame=actual,delta_frames=actual-frame,file=str(path)))
 env.close()
(B/'sensor_inventory.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
