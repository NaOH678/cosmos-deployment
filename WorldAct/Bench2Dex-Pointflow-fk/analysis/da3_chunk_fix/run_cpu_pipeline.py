"""Run native labeling and cleanup after full export + segmentation complete."""
import os,sys,subprocess,json
from pathlib import Path
B=Path(__file__).resolve().parent;N=Path('/mnt/afs/WorldAct-pointflow-native');EP='task21_episode000000'
os.environ['PYTHONPATH']=str(N/'Track4World_portable/Track4World/visualization')
os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['OPENBLAS_NUM_THREADS']='8'
assert (B/'run/efep_raw'/EP/'COMPLETE.json').exists()
assert (B/'run/sam2_masks'/EP/'meta.json').exists()
commands=[
 [sys.executable,str(N/'pf_out/efep_seg_label.py'),'--run-dir',str(B/'run'),'--episode',EP,'--fill-gaps','3','--despike','0.03','--out-name','efep_seg_v61'],
 [sys.executable,str(N/'pf_out/drop_detached.py'),'--data-dir',str(B/'run/efep_seg_v61'/EP)],
 [sys.executable,str(B/'validate_native.py')]]
for i,cmd in enumerate(commands):
 print('STAGE',i,cmd,flush=True)
 subprocess.run(cmd,check=True,env=os.environ.copy())
print('ALL_CPU_STAGES_COMPLETE',flush=True)
