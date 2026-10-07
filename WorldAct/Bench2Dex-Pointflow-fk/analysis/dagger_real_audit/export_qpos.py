import lmdb,pickle,json
import numpy as np
from pathlib import Path
B=Path(__file__).resolve().parent;D=B/'qpos';D.mkdir(exist_ok=True)
for ep in [dict(episode=k,frames=v['frames']) for k,v in json.load(open(B/'inventory.json')).items()]:
 name=ep['episode'];src=Path('/data/shichaojian/raw_data/dropper_dagger_mix')/name/'lmdb'
 env=lmdb.open(str(src),readonly=True,lock=False,readahead=False)
 with env.begin() as tx:q=np.asarray(pickle.loads(tx.get(b'/observations/qpos')))
 env.close();assert q.shape==(ep['frames'],54);np.save(D/(name+'.npy'),q)
print('Exported qpos for merged dagger episodes, no source mutations')
