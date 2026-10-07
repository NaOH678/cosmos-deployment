"""Offline authenticated JPEG HTTP ABBA test, no robot/ROS imports or commands."""
import copy, importlib.util, json, os, secrets, subprocess, sys, time, urllib.request, uuid
from pathlib import Path
import numpy as np
import yaml
root=Path(__file__).resolve().parent;lab=root.parent;worktrees=lab.parent.parent
sys.path.insert(0,str(lab.parent/'WorldAct-sft'))
script=worktrees/'wuji-hand-teleop-pipeline/src/scripts/benchmark_cosmos_http.py'
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
bench=load(script,'offline_bench');transport=load(worktrees/'wuji-hand-teleop-pipeline/src/wuji_data_pipeline/wuji_data_pipeline/policy_transport.py','policy_transport_abba')
corpus=json.loads((lab/'optimization_20261007/corpus.json').read_text());cfg=yaml.safe_load(Path(corpus['config']).read_text());dep=cfg['deployment']
inputs=[];hashes=[]
for entry in corpus['observations']:
 request,digest=bench.prepare_observation(entry['source'],entry['metadata'],cfg,95);inputs.append(request);hashes.append(digest)
key=secrets.token_hex(32);port=18026;url=f'http://127.0.0.1:{port}'
report={'scope':'Full JPEG HTTP request pickle/exchange/unpickle; JPEG encoded once before timer. Recording and video latent capture disabled in all stages. No robot. Native18005 kept resident idle. Same5 observations in same order, each stage new server and session,5warm+20measured. ABBA to mitigate drift.','jpeg_sha256':hashes,'stages':[]}
for idx,label in enumerate(('baseline','combined','combined','baseline')):
 directory=root/f'{idx}_{label}';directory.mkdir(exist_ok=True)
 stage_cfg=copy.deepcopy(cfg);stage_cfg['model']['service_mode']='full';stage_cfg['model']['output_dir']=str(directory/'server_output');(directory/'config.yaml').write_text(yaml.safe_dump(stage_cfg))
 env=os.environ.copy()
 for name in list(env):
  if name.startswith('COSMOS_RECORD') or name.startswith('COSMOS_VIDEO_LATENT') or name.startswith('WAM_'):env.pop(name,None)
 env.update(COSMOS_POLICY_API_KEY=key,OMP_NUM_THREADS='4',WAM_COMBINED_CANDIDATE='1' if label=='combined' else '0',PYTHONPATH=os.pathsep.join((str(root),str(lab),str(lab.parent/'WorldAct-sft'))))
 selected_lab=root/'candidate_lab' if label=='combined' else lab
 command=[str(lab/'.venv/bin/python'),str(root/'serve_candidate.py'),'--config',str(directory/'config.yaml'),'--omni-root',str(selected_lab),'--omni-model',str(lab/'artifacts/4w-ema-omni'),'--host','127.0.0.1','--port',str(port)]
 log=(directory/'server.log').open('w');server=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
 (directory/'server.pid').write_text(str(server.pid));client=None
 try:
  deadline=time.monotonic()+240
  while time.monotonic()<deadline:
   if server.poll() is not None:raise RuntimeError(f'Server failed; inspect {directory}/server.log')
   try:
    with urllib.request.urlopen(url+'/readyz',timeout=1) as r:
     if r.status==200:break
   except Exception:pass
   time.sleep(.5)
  else:raise TimeoutError('server warmup')
  client=transport.HttpPolicyTransport(url,timeout_ms=120000,policy_path=cfg['service'].get('endpoint','/v1/robot-policy'),api_key=key,max_response_bytes=32*1024*1024)
  session=uuid.uuid4().hex
  hello=dict(protocol_version=2,message_type='hello',session_id=session,request_id=1,robot_layout=dep['robot_layout'],camera_names=dep['camera_names'],action_space='eef',arm_command_mode='eef')
  ack=client.exchange(hello).response;assert ack.get('message_type')=='hello_ack',ack
  stage={'index':idx,'label':label,'samples':[]};report['stages'].append(stage)
  for i in range(25):
   payload=dict(inputs[i%5],session_id=session,request_id=i+2,timestamp=time.time(),client_monotonic=time.monotonic())
   t=time.perf_counter();exchange=client.exchange(payload);ms=(time.perf_counter()-t)*1000
   response=exchange.response
   if response.get('error'):raise RuntimeError(str(response.get('error_code')))
   values=bench.wire_vector(response);np.save(directory/f'wire_{i:02}.npy',values)
   stage['samples'].append({'index':i,'observation':i%5,'warmup':i<5,'rtt_ms':ms,'server_timing':response.get('server_timing')})
   (root/'report.json').write_text(json.dumps(report,indent=2))
  stage['rtt']=bench.statistics([s['rtt_ms'] for s in stage['samples'] if not s['warmup']]);print(label,stage['rtt'],flush=True)
 finally:
  if client:client.close()
  if server.poll() is None:server.terminate()
  try:server.wait(timeout=40)
  except subprocess.TimeoutExpired:raise RuntimeError('Owned benchmark server failed shutdown; stop and investigate, do not launch next')
  log.close()
 report['stages'][-1]['exit_code']=server.returncode
 (root/'report.json').write_text(json.dumps(report,indent=2))
comparisons=[]
for stage in range(1,4):
 for i in range(5,25):
  a=np.load(root/f'0_baseline/wire_{i:02}.npy');b=np.load(root/f'{stage}_{report["stages"][stage]["label"]}/wire_{i:02}.npy')
  comparisons.append({'stage':stage,'index':i,'equal':bool(np.array_equal(a,b)),'max_abs':float(np.abs(a-b).max())})
report['parity']=comparisons
report['aggregate']={label:bench.statistics([s['rtt_ms'] for st in report['stages'] if st['label']==label for s in st['samples'] if not s['warmup']]) for label in ('baseline','combined')}
(root/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['aggregate']),flush=True);print('ALL_WIRE_EQUAL',all(x['equal'] for x in comparisons),flush=True)
