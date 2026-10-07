"""Recording correlation tests; no ROS graph is created."""
from wuji_data_pipeline.deployment_state_trace import _recording_identity


def test_explicit_recording_identity_and_source_survive_generated_config():
    config = {'_config_path': '/runs/one/deployment.yaml', 'deployment': {
        'recording_run_id': 'one', 'recording_source_pipeline_config': '/config/50k.yaml'}}
    assert _recording_identity(config, '/runs/one/client') == {
        'recording_run_id': 'one', 'source_pipeline_config': '/config/50k.yaml'}


def test_run_directory_fallback_is_scoped_to_launcher_layout():
    assert _recording_identity({}, '/diagnostics/cosmos_runs/run-1/client')['recording_run_id'] == 'run-1'
    assert _recording_identity({}, '/diagnostics/client')['recording_run_id'] == ''
    assert _recording_identity({}, '/diagnostics')['recording_run_id'] == ''
