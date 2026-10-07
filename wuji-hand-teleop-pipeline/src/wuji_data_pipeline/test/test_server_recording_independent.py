"""Offline contract tests against the active old-source request recorder."""
import importlib.util
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

_source = (Path(__file__).resolve().parents[4] / 'WorldAct/WorldAct-sft/'
           'cosmos_framework/inference/robot_policy/recording.py')
if not _source.is_file():
    pytest.skip('WorldAct recorder contract requires the separate inference checkout', allow_module_level=True)
_spec = importlib.util.spec_from_file_location('independent_cosmos_recording', _source)
recording = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recording)


def inputs():
    return {'head': np.arange(36, dtype=np.uint8).reshape(3, 4, 3),
            'right_wrist': np.zeros((3, 5, 3), dtype=np.uint8)}, np.arange(27, dtype=np.float32), np.zeros((32, 27), dtype=np.float32)


def test_exact_array_and_identity_roundtrip_is_immutable(tmp_path):
    writer = recording.AsyncRequestRecorder(tmp_path, {'recording_run_id': 'r1'}, capacity=2)
    images, state, raw = inputs()
    expected = {**{k: v.copy() for k, v in images.items()}, 'state': state.copy(), 'raw_actions': raw.copy()}
    metadata = {'session_id': 'session-a', 'request_id': 42, 'server_generation': 3,
                'status': 'ok', 'wire_actions': [{'hand': [1, 2]}]}
    assert writer.submit(images, state, raw, metadata)
    state[:] = -100; images['head'][:] = 0; raw[:] = 1
    metadata['wire_actions'][0]['hand'][0] = -1
    assert writer.close()
    with np.load(writer.directory / '00000001.npz', allow_pickle=False) as saved:
        for key, value in expected.items():
            assert np.array_equal(saved[key], value)
    info = json.loads((writer.directory / '00000001.json').read_text())
    assert (info['session_id'], info['request_id'], info['server_generation']) == ('session-a', 42, 3)
    assert info['wire_actions'][0]['hand'][0] == 1
    assert info['shapes']['raw_actions'] == [32, 27]
    assert writer.stats()['written'] == 1


def test_queue_is_bounded_and_disk_io_stays_on_worker(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = recording.AsyncRequestRecorder._write_record
    thread_ids = []
    def blocked(self, *args):
        thread_ids.append(threading.get_ident())
        entered.set()
        assert release.wait(2)
        original(self, *args)
    monkeypatch.setattr(recording.AsyncRequestRecorder, '_write_record', blocked)
    writer = recording.AsyncRequestRecorder(tmp_path, {}, capacity=1)
    try:
        assert writer.submit(*inputs(), {'request_id': 1})
        assert entered.wait(2)
        assert writer.submit(*inputs(), {'request_id': 2})
        assert not writer.submit(*inputs(), {'request_id': 3})
        assert writer.stats()['pending'] == 1
        assert writer.stats()['dropped'] == 1
    finally:
        release.set()
        assert writer.close()
    assert all(t != threading.get_ident() for t in thread_ids)
    assert writer.stats()['written'] == 2


def test_disk_write_failure_does_not_escape_submit_and_is_counted(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError('test disk full')
    monkeypatch.setattr(recording.AsyncRequestRecorder, '_write_record', fail)
    writer = recording.AsyncRequestRecorder(tmp_path, {}, capacity=1)
    assert writer.submit(*inputs(), {'request_id': 1})
    assert writer.close()
    assert writer.stats()['errors'] >= 1
    assert writer.stats()['written'] == 0


def test_nonfinite_output_is_preserved_for_failure_analysis(tmp_path):
    writer = recording.AsyncRequestRecorder(tmp_path, {}, capacity=1)
    images, state, raw = inputs(); raw[0, 0] = np.nan
    assert writer.submit(images, state, raw, {'status': 'failed', 'metric': float('nan')})
    assert writer.close()
    info = json.loads((writer.directory / '00000001.json').read_text())
    assert info['finite']['raw_actions'] is False
    assert info['metric'] is None
    with np.load(writer.directory / '00000001.npz', allow_pickle=False) as saved:
        assert np.isnan(saved['raw_actions'][0, 0])


def test_environment_manifest_excludes_authentication(tmp_path, monkeypatch):
    monkeypatch.setenv('COSMOS_RECORDING_DIR', str(tmp_path))
    config = SimpleNamespace(model_dump=lambda **kw: {'deployment': {'action_space': 'eef'},
        'model': {'seed': 0}, 'auth': {'api_key': 'must-never-appear'},
        'server': {'authorization': 'must-never-appear'}})
    writer = recording.AsyncRequestRecorder.from_environment(config)
    assert writer is not None and writer.close()
    assert 'must-never-appear' not in (writer.directory / 'manifest.json').read_text()


def test_disabled_recording_has_no_thread_or_directory(tmp_path, monkeypatch):
    monkeypatch.delenv('COSMOS_RECORDING_DIR', raising=False)
    assert recording.AsyncRequestRecorder.from_environment(None) is None
    assert not list(tmp_path.iterdir())
