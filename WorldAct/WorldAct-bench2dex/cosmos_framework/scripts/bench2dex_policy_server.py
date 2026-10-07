"""Serve Cosmos through Bench2Dex's existing RPC protocol, without editing Bench2Dex."""

import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench2dex-root", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9000)
    args = p.parse_args()
    root = Path(args.bench2dex_root).resolve()
    if not (root / "script/policy_model_server.py").is_file():
        raise ValueError("Invalid Bench2Dex root")
    sys.path.insert(0, str(root))
    from cosmos_framework.inference.common.init import init_script

    init_script()
    from cosmos_framework.inference.robot_policy import bench2dex

    from script.policy_model_server import PolicyModelServer
    from script.policy_sessions import DefaultRemotePolicySession

    config = json.loads(Path(args.config).read_text())
    server = PolicyModelServer(DefaultRemotePolicySession(bench2dex, config), args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
