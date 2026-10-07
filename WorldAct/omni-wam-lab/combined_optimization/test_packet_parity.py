import hashlib,json,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
root=Path(__file__).resolve().parent;lab=root.parent
sys.path.insert(0,str(lab.parent/'WorldAct-sft'))
from cosmos_framework.inference.robot_policy.omni_http import OmniWamAdapter
from cosmos_framework.inference.robot_policy.config import load_robot_policy_config
from first_frame_packet import build_first_frame_packet
corpus=json.loads((lab/'optimization_20261007/corpus.json').read_text());config=load_robot_policy_config(corpus['config'])
a=OmniWamAdapter.__new__(OmniWamAdapter);a.config=config;a.input_video_key='video';orig=torch.Tensor.to
rows=[]
def cpu_to(tensor,*args,**kwargs):
 if kwargs.get('device')=='cuda':kwargs['device']='cpu'
 return orig(tensor,*args,**kwargs)
for entry in corpus['observations']:
 with np.load(entry['source']) as z:images={k:z[k].copy() for k in ('head','right_wrist')};state=z['state'].copy()
 with patch.object(torch.Tensor,'to',cpu_to):baseline=a._make_packet(images,state)
 candidate=build_first_frame_packet(config,images,state)
 assert baseline['metadata']==candidate['metadata']
 for key in ('first_frame','action','image_size'):np.testing.assert_array_equal(baseline[key],candidate[key])
 hashes={key:hashlib.sha256(candidate[key].tobytes()).hexdigest() for key in ('first_frame','action','image_size')}
 hashes['metadata']=hashlib.sha256(json.dumps(candidate['metadata'],sort_keys=True).encode()).hexdigest()
 rows.append({'sequence':entry['sequence'],'all_equal':True,'sha256':hashes})
(root/'packet_parity.json').write_text(json.dumps(rows,indent=2));print('PASS 5 inputs arrays and full metadata equal')
