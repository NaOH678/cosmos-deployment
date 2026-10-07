import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

_spec=importlib.util.spec_from_file_location('http_benchmark',Path(__file__).resolve().parents[2]/'scripts/benchmark_cosmos_http.py')
bench=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(bench)


def sample(tmp_path):
    state=np.r_[.5,.1,.2,0.,0.,0.,1.,np.arange(20)/100].astype(np.float32)
    measured={}
    for side in ['left','right']:
        measured['arm_state_'+side]={'eef':state[:7].tolist(),'joint_pos':[0.]*7}
        measured['hand_state_'+side]={'joint_pos':state[7:].tolist()}
    rgb=np.zeros((16,16,3),np.uint8);rgb[:,:,0]=255
    np.savez(tmp_path/'input.npz',head=rgb,right_wrist=rgb,state=state)
    (tmp_path/'input.json').write_text(json.dumps({'measured_state':measured,'request_timestamp':100.}))
    return {'deployment':{'robot_layout':{},'camera_names':['head','right_wrist']}}


def test_payload_preserves_radian_state_and_rgb_channel_meaning(tmp_path):
    config=sample(tmp_path)
    req,hashes=bench.prepare_observation(tmp_path/'input.npz',tmp_path/'input.json',config)
    assert req['hand_state_right']['joint_pos'][10]==pytest.approx(.1)
    assert np.array_equal(req['arm_state_right']['ee_pos'],req['arm_state_right']['eef'][:3])
    assert np.array_equal(req['arm_state_right']['joint_torque'],[0.]*7)
    decoded=cv2.imdecode(np.frombuffer(req['images']['head']['data'],np.uint8),cv2.IMREAD_COLOR)
    assert decoded[:,:,2].mean()>250 and decoded[:,:,0].mean()<5
    req2,hashes2=bench.prepare_observation(tmp_path/'input.npz',tmp_path/'input.json',config)
    assert hashes==hashes2 and req['images']['head']['data']==req2['images']['head']['data']


def test_wrong_metadata_state_is_not_silently_replayed(tmp_path):
    config=sample(tmp_path);p=tmp_path/'input.json';m=json.loads(p.read_text())
    m['measured_state']['hand_state_right']['joint_pos'][0]=1.;p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='exactly reproduce'):
        bench.prepare_observation(tmp_path/'input.npz',p,config)


def test_wire_vector_rejects_missing_steps_or_nonfinite():
    action={'arm_action_right':{'ee_pos':[0.,0.,0.],'ee_quat':[0.,0.,0.,1.]},'hand_action_right':[0.]*20}
    assert bench.wire_vector({'action_chunk':[action]*32}).shape==(32,27)
    with pytest.raises(ValueError):bench.wire_vector({'action_chunk':[action]*31})
    action['arm_action_right']['ee_pos'][0]=float('nan')
    with pytest.raises(ValueError):bench.wire_vector({'action_chunk':[action]*32})
