#!/usr/bin/env bash
# Interactive real-robot experiment. Recovery/enable remain user keyboard actions.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RTC_POINTER=/tmp/cosmos5090_rtc_hybrid_active_path
[[ -f "$RTC_POINTER" ]] || { echo "RTC service pointer missing; start the RTC service first." >&2; exit 2; }
RTC_ROOT="$(cat "$RTC_POINTER")"
[[ -f "$RTC_ROOT/status.json" && -f "$RTC_ROOT/server.yaml" ]] || { echo "RTC service has not completed warmup." >&2; exit 2; }
export COSMOS_POLICY_API_KEY="$(cat /tmp/cosmos-cloud-trial-credentials/api_key)"
export COSMOS_DEPLOYMENT_CONFIG="$RTC_ROOT/server.yaml"
echo "RTC hybrid trial: exact VJP at step 3/4, single 5090, iter20000, window16 / lead16, queue-prefix alignment, blend8/2."
echo "Server observations and video latents: $RTC_ROOT"
exec "$REPO_ROOT/src/scripts/start_local_cosmos_deployment.sh"   --attach-existing --port 18004 --service-mode full   --checkpoint-dir /home/pjlab/ros2_ws/worktrees/WorldAct/model_ch/real_ckpt/singlerighthand-edge-droid-16n-1001/iter_000020000   --model-config-file /home/pjlab/ros2_ws/worktrees/WorldAct/model_ch/real_ckpt/singlerighthand-edge-droid-16n-1001/config.deploy.yaml   --model-package /home/pjlab/ros2_ws/worktrees/WorldAct/models/cosmos3-edge-droid   --config /home/wuji/ros2_ws/src/wuji_data_pipeline/config/cosmos_protocol_v2_16n20k_rtc_hybrid.yaml "$@"
