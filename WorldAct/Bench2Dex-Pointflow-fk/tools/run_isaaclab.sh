#!/usr/bin/env bash
# Use the environment's C++ runtime before Kit loads the older system runtime.
# Override ISAACLAB_PYTHON when the environment lives elsewhere.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export TMPDIR="${ISAACLAB_RUNTIME_DIR:-$repo_root/.cache/sim_environment/runtime}"
mkdir -p "$TMPDIR"
sim_python="${ISAACLAB_PYTHON:-$HOME/miniconda3/envs/env_isaaclab/bin/python}"
sim_prefix="$("$sim_python" -c 'import sys; print(sys.prefix)')"
if [[ -f "$sim_prefix/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="$sim_prefix/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
exec "$sim_python" -u "$@"
