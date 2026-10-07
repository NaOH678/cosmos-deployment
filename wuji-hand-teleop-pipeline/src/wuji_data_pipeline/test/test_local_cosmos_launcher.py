"""Exercise launch routing without Docker, ROS, GPU or a real HTTP service."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
CLIENT = ROOT / "src/scripts/start_local_cosmos_deployment.sh"
SERVER = ROOT.parent / "WorldAct/WorldAct-sft-pointflow-fk/script/start_cosmos_local_policy_server.sh"


@pytest.mark.parametrize("run,override,expected", [
    ("singlerighthand-edge-droid-50k-retrain-v1-0831", None, "WorldAct-sft"),
    ("singlerighthand-edge-droid-16n-1001", None, "WorldAct-sft-pointflow-fk"),
    ("singlerighthand-edge-droid-50k-retrain-v1-0831", "explicit source", "explicit source"),
])
def test_check_only_routes_source_before_any_hardware_start(tmp_path, run, override, expected):
    runtime = tmp_path / "WorldAct-sft-pointflow-fk"
    script = runtime / "script/start_cosmos_local_policy_server.sh"
    script.parent.mkdir(parents=True)
    script.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    script.chmod(0o755)
    env = dict(os.environ, WORLDACT_ROOT=str(runtime))
    env.pop("COSMOS_INFERENCE_REPO", None)
    command = ["bash", str(CLIENT), "--checkpoint-dir", str(tmp_path / run / "iter_000040000"),
               "--model-config-file", str(tmp_path / "config.yaml"), "--check-only"]
    if override:
        command.extend(["--inference-repo", str(tmp_path / override)])
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    args = result.stdout.splitlines()
    assert args[args.index("--inference-repo") + 1] == str(tmp_path / expected)
    assert args[-1] == "--check-only"
    assert "Generated a per-run" not in result.stdout


def test_check_only_rejects_attach_existing():
    result = subprocess.run(["bash", str(CLIENT), "--attach-existing", "--check-only"],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "cannot verify" in result.stderr


@pytest.mark.parametrize("check_only", [True, False])
def test_server_executes_selected_checkout_not_environment_checkout(tmp_path, check_only):
    if not SERVER.is_file():
        pytest.skip("WorldAct local launcher is in a separate checkout")
    source = tmp_path / "reference source"
    package = source / "cosmos_framework"
    scripts = package / "scripts"
    scripts.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (scripts / "__init__.py").write_text("")
    marker = tmp_path / "entry.json"
    (scripts / "action_policy_server_protocol_v2.py").write_text(
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['TEST_ENTRY_MARKER']).write_text("
        "json.dumps({'cwd': os.getcwd(), 'args': sys.argv[1:]}))\n"
    )
    config = tmp_path / "manifest.yaml"
    config.write_text("{}\n")
    env = dict(os.environ, COSMOS_PYTHON=sys.executable, COSMOS_POLICY_API_KEY="test-only",
               H_API_KEY="", TEST_ENTRY_MARKER=str(marker))
    command = ["bash", str(SERVER), "--inference-repo", str(source), "--service-mode", "hold",
               "--deployment-config", str(config), "--checkpoint-dir", "unused",
               "--model-config-file", "unused.yaml"]
    if check_only:
        command.append("--check-only")
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    assert f"Verified inference import: {package / '__init__.py'}" in result.stdout
    if check_only:
        assert not marker.exists()
    else:
        entry = json.loads(marker.read_text())
        assert entry["cwd"] == str(source)
        assert entry["args"][entry["args"].index("--config") + 1] == str(config)
