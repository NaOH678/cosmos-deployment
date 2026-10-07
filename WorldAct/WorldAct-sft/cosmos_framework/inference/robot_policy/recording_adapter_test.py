"""CPU-only adapter integration checks for recording; no model/hardware load."""

import json
import threading

import numpy as np
import pytest

from cosmos_framework.inference.robot_policy import adapters
from cosmos_framework.inference.robot_policy.recording import AsyncRequestRecorder
from cosmos_framework.inference.robot_policy.robot_policy_test import _config


@pytest.mark.parametrize("bad_output", [False, True])
def test_actual_infer_records_raw_before_postprocess(tmp_path, monkeypatch, bad_output):
    config = _config()
    adapter = object.__new__(adapters.SingleRightHandCosmosAdapter)
    adapter.config = config
    adapter._ready = True
    adapter._inference_timing = threading.local()
    adapter._recorder = AsyncRequestRecorder(tmp_path, {})
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[:, :, 0] = 123
    state = np.arange(27, dtype=np.float32)
    raw = np.zeros((32, 27), dtype=np.float32)
    raw[:, 6] = 1
    if bad_output:
        raw[0, 0] = np.nan
    monkeypatch.setattr(adapters, "_decode_jpeg_rgb", lambda *_: rgb)
    adapter._infer_native = lambda *_: (raw, 12.0)

    def wire(_observation, value, _config):
        assert value is raw
        if bad_output:
            raise adapters.ProtocolError("COSMOS_INVALID_ACTION", "invalid")
        return [{"postprocessed": 17.0}]

    observation = {
        "images": {"head": {}, "right_wrist": {}},
        "session_id": "s1",
        "request_id": "r1",
        "authorization": "DO_NOT_RECORD_SECRET",
    }
    if bad_output:
        with pytest.raises(adapters.ProtocolError):
            adapter._infer_recorded(observation, lambda _: state, wire)
    else:
        result = adapter._infer_recorded(observation, lambda _: state, wire)
        assert result.action_chunk == [{"postprocessed": 17.0}]
    adapter.close_recording()
    path = adapter._recorder.directory
    info = json.loads((path / "00000001.json").read_text())
    assert info["session_id"] == "s1" and info["request_id"] == "r1"
    assert info["status"] == ("failed" if bad_output else "complete")
    assert info["finite"]["raw_actions"] is not bad_output
    assert info["timings"]["model_ms"] == 12.0
    assert info["wire_actions"] == (None if bad_output else [{"postprocessed": 17.0}])
    assert "DO_NOT_RECORD_SECRET" not in (path / "00000001.json").read_text()
    with np.load(path / "00000001.npz", allow_pickle=False) as stored:
        np.testing.assert_equal(stored["raw_actions"], raw)
        np.testing.assert_array_equal(stored["state"], state)
        np.testing.assert_array_equal(stored["head"], rgb)


def test_sigterm_uses_server_close_and_ignores_repeated_signal(monkeypatch):
    import importlib
    import signal
    from types import SimpleNamespace

    from cosmos_framework.inference.common import init

    monkeypatch.setattr(init, "init_script", lambda: None)
    entry = importlib.import_module("cosmos_framework.scripts.action_policy_server_protocol_v2")
    config = _config()
    config.auth.api_key_env = "COSMOS_RECORDING_TEST_KEY"
    monkeypatch.setenv("COSMOS_RECORDING_TEST_KEY", "test-only")
    monkeypatch.setattr(entry, "_parse_args", lambda: SimpleNamespace(config="unused"))
    monkeypatch.setattr(entry, "load_robot_policy_config", lambda _: config)
    monkeypatch.setattr(entry, "_apply_overrides", lambda value, _: value)
    monkeypatch.setattr(entry, "create_model_adapter", lambda _: object())
    closed = []

    class Server:
        def serve_forever(self, **_):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

        def server_close(self):
            assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
            assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
            closed.append(True)

    old_term, old_int = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
    monkeypatch.setattr(entry, "create_http_server", lambda *_: Server())
    entry.main()
    assert closed == [True]
    assert signal.getsignal(signal.SIGTERM) == old_term
    assert signal.getsignal(signal.SIGINT) == old_int
