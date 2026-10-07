import os,sys,subprocess
from pathlib import Path
B=Path(__file__).resolve().parent;N=Path('/mnt/afs/WorldAct-pointflow-native');EP='episode_0013_20260731_133649'
os.environ['PYTHONPATH']=str(N/'Track4World_portable/Track4World/visualization');os.environ['OPENBLAS_NUM_THREADS']='8';os.environ['PYTHONDONTWRITEBYTECODE']='1'
assert (B/'run/efep_raw'/EP/'COMPLETE.json').exists()
for cmd in [[sys.executable,str(N/'pf_out/efep_seg_label.py'),'--run-dir',str(B/'run'),'--episode',EP,'--fill-gaps','3','--despike','0.03','--out-name','efep_seg_fixed'],[sys.executable,str(N/'pf_out/drop_detached.py'),'--data-dir',str(B/'run/efep_seg_fixed'/EP)]]:
 print('RUN',cmd,flush=True);subprocess.run(cmd,check=True,env=os.environ.copy())
print('CPU_COMPLETE',flush=True)
