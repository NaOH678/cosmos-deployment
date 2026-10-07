#!/usr/bin/env bash
# Compatibility alias; use start_local_cosmos_deployment.sh directly.
set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/start_local_cosmos_deployment.sh" "$@"
