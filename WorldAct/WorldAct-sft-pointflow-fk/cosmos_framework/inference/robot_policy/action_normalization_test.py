import hashlib
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from cosmos_framework.inference.robot_policy.action_normalization import NAMES, load_stats, load_for_deployment


def payload():
    return dict(schema="singlerighthand_action_quantile_v1", method="action_quantile_scale_floor",
                arm_action_space="eef", action_names=NAMES,
                units=["meter"]*3+["unitless_quaternion"]*4+["radian"]*20,
                preserved_dimensions=[3,4,5,6], fit_split="train", state_transform="same_as_action",
                forward_clamp=None, offset=[0.4]*3+[0.0]*4+[0.5]*20,
                scale=[0.1]*3+[1.0]*4+[0.2]*20)


def save(tmp_path, stats):
    p=tmp_path/"stats.json"; raw=json.dumps(stats).encode(); p.write_bytes(raw)
    return p, hashlib.sha256(raw).hexdigest()


def test_physical_roundtrip_and_preserved_quaternion(tmp_path):
    normalizer=load_stats(*save(tmp_path,payload()))
    state=np.linspace(-0.5,1.0,27,dtype=np.float32)
    normalized=normalizer.normalize(state)
    np.testing.assert_allclose(normalizer.denormalize(normalized),state,atol=1e-7)
    np.testing.assert_array_equal(normalized[3:7],state[3:7])
    assert np.abs(normalized).max()>1 # No unintended clipping.
    np.testing.assert_allclose(normalizer.denormalize(np.zeros((32,27))),np.tile(normalizer.offset,(32,1)))


@pytest.mark.parametrize("change", [
    {"state_transform":"separate"}, {"forward_clamp":1}, {"action_names":list(reversed(NAMES))},
    {"arm_action_space":"joint"}, {"scale":[0]*27}, {"offset":[float("nan")]*27},
    {"scale":[2]*27}, {"units":["degree"]*27}, {"preserved_dimensions":[]},
])
def test_reject_incompatible_stats(tmp_path,change):
    stats=payload(); stats.update(change)
    with pytest.raises(ValueError): load_stats(*save(tmp_path,stats))


def test_checksum_and_frozen_config_fail_closed(tmp_path):
    path,digest=save(tmp_path,payload())
    with pytest.raises(ValueError): load_stats(path,"0"*64)
    frozen=tmp_path/"config.yaml"
    cfg=SimpleNamespace(model=SimpleNamespace(config_file=str(frozen)),deployment=SimpleNamespace(action_space="eef"))
    frozen.write_text("{}")
    assert load_for_deployment(cfg) is None
    frozen.write_text(json.dumps({"dataloader_train":{"dataloader":{"datasets":{"singlerighthand":{"dataset":{
        "action_stats_path":"stats.json", "action_stats_sha256":digest}}}}}}))
    assert load_for_deployment(cfg).sha256==digest
    path.unlink()
    with pytest.raises(FileNotFoundError):load_for_deployment(cfg)


def test_adapter_denormalizes_after_condition_row_removal(tmp_path,monkeypatch):
    import torch
    from cosmos_framework.inference.robot_policy.adapters import SingleRightHandCosmosAdapter
    adapter=object.__new__(SingleRightHandCosmosAdapter)
    adapter._action_normalizer=load_stats(*save(tmp_path,payload()))
    adapter._lock=threading.Lock()
    adapter.config=SimpleNamespace(model=SimpleNamespace(guidance=3,seed=0,num_steps=4,shift=5,native_chunk_size=32,native_action_dim=27))
    # The condition row must not be returned, and exactly one inverse is applied.
    result=torch.zeros((1,33,27));result[:,0,:]=123
    adapter.model=SimpleNamespace(generate_samples_from_batch=lambda *a,**kw:{"action":[result]})
    adapter._build_batch=lambda *args:{}
    monkeypatch.setattr(torch.cuda,"synchronize",lambda:None)
    actions,_=adapter._infer_native({},np.zeros(27))
    np.testing.assert_allclose(actions,np.tile(adapter._action_normalizer.offset,(32,1)))
    adapter._action_normalizer=None
    raw,_=adapter._infer_native({},np.zeros(27))
    np.testing.assert_array_equal(raw,np.zeros((32,27)))


def test_adapter_normalizes_state_before_padding(tmp_path,monkeypatch):
    import torch
    from cosmos_framework.inference.robot_policy.adapters import SingleRightHandCosmosAdapter
    from cosmos_framework.data.generator.action import transforms
    adapter=object.__new__(SingleRightHandCosmosAdapter)
    adapter._action_normalizer=load_stats(*save(tmp_path,payload()))
    adapter.input_video_key="video"
    adapter.config=SimpleNamespace(model=SimpleNamespace(native_chunk_size=32,native_action_dim=27,max_action_dim=64,
        resolution="480",task="make a sandwich",domain_name="singlerighthand"),deployment=SimpleNamespace(action_rate_hz=15.0))
    monkeypatch.setattr(transforms,"find_closest_target_size",lambda *a:(16,32))
    def pad(batch,*args):batch["image_size"]=torch.tensor([32,16])
    monkeypatch.setattr(transforms,"reflection_pad_to_target",pad)
    original_to=torch.Tensor.to
    def cpu_to(self,*args,**kwargs):
        if kwargs.get("device")=="cuda":kwargs["device"]="cpu"
        return original_to(self,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,"to",cpu_to)
    state=np.linspace(0,1,27,dtype=np.float32)
    batch=adapter._build_batch({"head":np.zeros((16,16,3),np.uint8),"right_wrist":np.zeros((16,16,3),np.uint8)},state)
    action=batch["action"][0][0].numpy()
    np.testing.assert_allclose(action[0,:27],adapter._action_normalizer.normalize(state))
    np.testing.assert_array_equal(action[:,27:],0)
    np.testing.assert_array_equal(action[1:],0)
