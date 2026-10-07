from pathlib import Path
import json,os,hashlib
from concurrent.futures import ThreadPoolExecutor
B=Path(__file__).resolve().parent;I=json.load(open(B/'inventory.json'));RAW=Path('/data/shichaojian/raw_data/dropper_dagger_mix');cache=json.load(open('/data/shichaojian/datasets/dropper-dagger-mix-cosmos-cache/manifest.json'));C={r['name']:r['num_frames'] for r in cache['episodes']}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  while b:=f.read(4*1024*1024):h.update(b)
 return h.hexdigest()
def work(item):
 ep,r=item;report=json.load(open(Path(r['pointflow'])/'report.json'));old=Path(report['export']['video'].replace('/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian','/data/shichaojian'));now=RAW/ep/'videos/head.mp4';d=dict(episode=ep,pointflow_frames=r['frames'],cache_frames=C.get(ep),cache_frames_match=C.get(ep)==r['frames'],export_video=str(old),evaluation_video=str(now))
 if old.exists():
  d['samefile']=os.path.samefile(old,now)
  if d['samefile']:d['video_identical']=True
  else:d['video_sha256']=sha(now);d['export_video_sha256']=sha(old);d['video_identical']=d['video_sha256']==d['export_video_sha256']
 else:d['video_identical']=None
 return d
with ThreadPoolExecutor(max_workers=4) as ex:rows=list(ex.map(work,I.items()))
(B/'source_checks.json').write_text(json.dumps(rows,indent=2));print('cache mismatches',sum(not r['cache_frames_match'] for r in rows),'video mismatches',sum(r['video_identical'] is False for r in rows),'unavailable',sum(r['video_identical'] is None for r in rows),'samefile',sum(r.get('samefile',False) for r in rows))
