import importlib.util
import json
from pathlib import Path

import pytest
import yaml

PATH = Path(__file__).resolve().parents[2] / 'scripts/prepare_cosmos_recording.py'
spec = importlib.util.spec_from_file_location('prepare_cosmos_recording', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture_config(tmp_path):
    config = tmp_path / 'src/wuji_data_pipeline/config/test.yaml'
    config.parent.mkdir(parents=True)
    config.write_text('deployment:\n  action_rate_hz: 15\n  publish_rate_hz: 120\n  policy_http_api_key_env: COSMOS_POLICY_API_KEY\n')
    return config, '/home/wuji/ros2_ws/src/wuji_data_pipeline/config/test.yaml'


def test_prepare_correlates_paths_without_changing_control_or_original(tmp_path):
    source, container = fixture_config(tmp_path)
    original = source.read_bytes()
    identity, run, generated = module.prepare(tmp_path, container, '/reference')
    config = yaml.safe_load((run/'deployment.yaml').read_text())['deployment']
    assert config['recording_run_id'] == identity
    assert str(generated).startswith('/home/wuji/datasets/tianji_wuji/diagnostics/cosmos_runs/')
    assert config['action_rate_hz'] == 15 and config['publish_rate_hz'] == 120
    assert config['diagnostic_policy_chunk_enabled'] is True
    assert config['recording_source_pipeline_config'] == container
    assert source.read_bytes() == original
    assert module.host_config(tmp_path, str(generated)) == run/'deployment.yaml'
    manifest = json.loads((run/'manifest.json').read_text())
    assert manifest['server_capture_expected'] is True
    assert manifest['status'] == 'prepared'


def test_secret_snapshots_redact_and_do_not_copy_environment(tmp_path, monkeypatch):
    _, container = fixture_config(tmp_path)
    model = tmp_path/'model.yaml'
    model.write_text('model:\n  token: TOP_SECRET\n  seed: 0\n')
    monkeypatch.setenv('COSMOS_POLICY_API_KEY', 'ENV_SECRET')
    _, run, _ = module.prepare(tmp_path, container, '/reference', str(model))
    combined = ''.join(p.read_text() for p in run.rglob('*') if p.is_file())
    assert 'TOP_SECRET' not in combined and 'ENV_SECRET' not in combined
    assert '<redacted>' in combined


def test_inline_pipeline_secret_is_rejected_before_creating_run(tmp_path):
    source, container = fixture_config(tmp_path)
    source.write_text('deployment:\n  api_key: INLINE_SECRET\n')
    with pytest.raises(ValueError, match='inline credentials'):
        module.prepare(tmp_path, container, '/reference')
    assert not (tmp_path/'datasets').exists()


def test_finalize_indexes_evidence_without_claiming_complete(tmp_path):
    _, container = fixture_config(tmp_path)
    _, run, _ = module.prepare(tmp_path, container, '/reference', attached=True)
    (run/'client/test.jsonl').write_text('{}\n')
    module.finalize(run, 130)
    result = json.loads((run/'manifest.json').read_text())
    assert result['launcher_exit_code'] == 130
    assert result['server_capture_expected'] is False
    assert 'client/test.jsonl' in result['files']
    assert 'Inspect' in result['recording_completeness']


def test_paths_cannot_escape_container_mount(tmp_path):
    with pytest.raises(ValueError, match='escapes'):
        module.host_config(tmp_path, '/home/wuji/ros2_ws/src/../../secrets.yaml')


def test_server_recording_must_be_ready_and_match_run(tmp_path):
    _, container = fixture_config(tmp_path)
    identity, run, _ = module.prepare(tmp_path, container, '/reference')
    with pytest.raises(RuntimeError, match='did not initialize'):
        module.verify_server(run)
    server = run/'server/run_example'
    server.mkdir()
    (server/'manifest.json').write_text(json.dumps({'recording_run_id': identity, 'enabled': True}))
    (server/'stats.json').write_text(json.dumps({'closed': False, 'errors': 0, 'stats_errors': 0}))
    module.verify_server(run)
    (server/'stats.json').write_text(json.dumps({'closed': False, 'errors': 1}))
    with pytest.raises(RuntimeError, match='did not initialize'):
        module.verify_server(run)
