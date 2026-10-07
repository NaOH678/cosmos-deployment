#!/usr/bin/env python3
"""Prepare a per-run Cosmos diagnostics bundle; never starts services/hardware."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid
import yaml


def redact(value):
    if isinstance(value, dict):
        return {k: ('<redacted>' if str(k).lower() in {
            'api_key', 'token', 'password', 'secret', 'authorization', 'access_token',
            'refresh_token', 'bearer_token', 'hf_token', 'huggingface_token',
            'aws_secret_access_key', 'secret_key'} else redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def host_config(repo, value):
    for prefix, root in [('/home/wuji/ros2_ws/src/', repo / 'src'),
                         ('/home/wuji/datasets/', repo / 'datasets')]:
        if value.startswith(prefix):
            result = (root / value[len(prefix):]).resolve()
            if not result.is_relative_to(root.resolve()):
                raise ValueError('configuration escapes its container mount')
            return result
    raise ValueError('recording requires a config under /home/wuji/ros2_ws/src/ or /home/wuji/datasets/')


def prepare(repo, config, inference_repo, model_config='', deployment_config='', checkpoint='', model_package='', attached=False, service_mode='full'):
    repo = Path(repo).resolve()
    source = host_config(repo, config)
    data = yaml.safe_load(source.read_text())
    if not isinstance(data, dict) or not isinstance(data.get('deployment'), dict):
        raise ValueError('pipeline config must have a deployment mapping')
    if redact(data) != data:
        raise ValueError('inline credentials cannot be saved; use API-key environment references')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    relative = Path('tianji_wuji/diagnostics/cosmos_runs') / run_id
    run = repo / 'datasets' / relative
    run.mkdir(parents=True, exist_ok=False)
    (run/'client').mkdir()
    (run/'server').mkdir()
    container_run = Path('/home/wuji/datasets') / relative
    dep = data['deployment']
    dep.update(recording_run_id=run_id, recording_source_pipeline_config=config,
               diagnostic_trace_enabled=True, diagnostic_policy_chunk_enabled=True,
               diagnostic_trace_directory=str(container_run/'client'),
               state_action_trace_directory=str(container_run/'client'),
               state_action_trace_rate_hz=120.0)
    (run/'deployment.yaml').write_text(yaml.safe_dump(data, sort_keys=False))
    manifest = dict(schema_version=1, run_id=run_id, status='prepared',
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    source_pipeline_config=str(source), container_pipeline_config=config,
                    inference_repo=str(inference_repo), checkpoint_dir=str(checkpoint),
                    model_package=str(model_package), attached_existing_server=attached,
                    server_capture_expected=not attached and service_mode == 'full',
                    service_mode=service_mode,
                    server_capture_directory=str(run/'server'),
                    note='No secrets/environment dump. Prepared config is not evidence of a hardware run. Check component error/drop/completion summaries.',
                    config_files={})
    for label, name in [('pipeline_source', source), ('model_config', model_config),
                        ('server_deployment', deployment_config)]:
        if not name:
            continue
        path = Path(name).expanduser().resolve()
        entry = {'path': str(path), 'exists': path.is_file()}
        if path.is_file():
            content = path.read_bytes()
            entry['sha256'] = hashlib.sha256(content).hexdigest()
            copied = run / (label+'.yaml')
            copied.write_text(yaml.safe_dump(redact(yaml.safe_load(content)), sort_keys=False))
            entry['redacted_snapshot'] = copied.name
        manifest['config_files'][label] = entry
    manifest['source_sha256'] = {}
    roots = [(repo, ['src/scripts/start_local_cosmos_deployment.sh',
                     'src/wuji_data_pipeline/wuji_data_pipeline/deployment_node.py',
                     'src/wuji_data_pipeline/wuji_data_pipeline/deployment_protocol.py']),
             (Path(inference_repo), ['cosmos_framework/inference/robot_policy/adapters.py',
                                    'cosmos_framework/inference/robot_policy/recording.py',
                                    'cosmos_framework/model/generator/omni_mot_model.py'])]
    for base, names in roots:
        for name in names:
            path = base / name
            if path.is_file():
                manifest['source_sha256'][str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = Path(checkpoint) / 'model/.metadata'
    if checkpoint and metadata.is_file():
        manifest['checkpoint_metadata_sha256'] = hashlib.sha256(metadata.read_bytes()).hexdigest()
    manifest['checkpoint_full_weight_hash_computed'] = False
    (run/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return run_id, run, container_run/'deployment.yaml'


def verify_server(run):
    run = Path(run)
    expected = json.loads((run/'manifest.json').read_text())['run_id']
    for path in (run/'server').glob('run_*/manifest.json'):
        manifest = json.loads(path.read_text())
        stats_path = path.parent/'stats.json'
        if not stats_path.exists():
            continue
        stats = json.loads(stats_path.read_text())
        if (manifest.get('recording_run_id') == expected and manifest.get('enabled')
                and not stats.get('closed') and not stats.get('errors')
                and not stats.get('stats_errors')):
            return
    raise RuntimeError('Server recording did not initialize successfully; refusing to start the deployment session')


def finalize(run, exit_code):
    run = Path(run).resolve()
    path = run/'manifest.json'
    manifest = json.loads(path.read_text())
    manifest.update(status='launcher_exited', launcher_exit_code=exit_code,
                    stopped_utc=datetime.now(timezone.utc).isoformat())
    manifest['files'] = [str(p.relative_to(run)) for p in sorted(run.rglob('*')) if p.is_file() and p != path]
    manifest['recording_completeness'] = 'Inspect server stats and client trace summaries; missing or nonzero drop/error counts mean incomplete evidence.'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, indent=2)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo')
    parser.add_argument('--config')
    parser.add_argument('--inference-repo', default='')
    parser.add_argument('--model-config', default='')
    parser.add_argument('--deployment-config', default='')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--model-package', default='')
    parser.add_argument('--attached', action='store_true')
    parser.add_argument('--service-mode', default='full')
    parser.add_argument('--finalize-dir')
    parser.add_argument('--verify-server-dir')
    parser.add_argument('--exit-code', type=int, default=0)
    args = parser.parse_args()
    if args.verify_server_dir:
        verify_server(args.verify_server_dir)
    elif args.finalize_dir:
        finalize(args.finalize_dir, args.exit_code)
    else:
        if not args.repo or not args.config:
            parser.error('--repo and --config are required')
        for value in prepare(args.repo, args.config, args.inference_repo, args.model_config,
                             args.deployment_config, args.checkpoint, args.model_package, args.attached, args.service_mode):
            print(value)


if __name__ == '__main__':
    main()
