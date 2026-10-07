"""Serve a recorded LMDB episode through the deployment ZMQ protocol."""

from __future__ import annotations

import argparse
import logging
import pickle
import time

from .episode import load_episode
from .deployment_protocol import PROTOCOL_VERSION
from .replay_core import ReplayEngine


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
)
log = logging.getLogger("wuji_replay_server")


def _require_zmq():
    try:
        import zmq
    except ImportError as exc:
        raise RuntimeError(
            "pyzmq is required; rebuild the project Docker image after the dependency update"
        ) from exc
    return zmq


def run_server(args) -> None:
    arrays, metadata = load_episode(args.episode_dir)
    if args.arm_command_mode == "joint" and "qpos" not in arrays:
        raise ValueError(
            "joint replay requires the episode qpos sequence"
        )
    rebase = bool(args.rebase and args.arm_command_mode == "eef")
    engine = ReplayEngine(
        arrays["action"],
        metadata,
        zsp=arrays.get("zsp"),
        qpos=(
            arrays.get("qpos")
            if args.arm_command_mode == "joint"
            else None
        ),
        rate_hz=args.rate_hz,
        rebase=rebase,
        start_hold_s=args.start_hold_s,
        loop=args.loop,
    )
    log.info(
        "Loaded %s: frames=%d action_dim=%d sides=%s rate=%.2fHz "
        "arm_command_mode=%s rebase=%s action_rate=%.2fHz",
        args.episode_dir,
        arrays["action"].shape[0],
        arrays["action"].shape[1],
        engine.layout.sides,
        engine.rate_hz,
        args.arm_command_mode,
        rebase,
        float(args.action_rate_hz),
    )
    if args.dry_run:
        log.info("Dry-run validation passed; no socket opened")
        return

    zmq = _require_zmq()
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(args.bind)
    log.info("Replay server listening on %s", args.bind)
    requests = 0
    try:
        while True:
            raw = socket.recv()
            try:
                observation = pickle.loads(raw)
                if not isinstance(observation, dict):
                    raise ValueError("observation is not a mapping")
                if observation.get("message_type") == "hello":
                    socket.send(
                        pickle.dumps(
                            {
                                "protocol_version": PROTOCOL_VERSION,
                                "message_type": "hello_ack",
                                "session_id": observation.get("session_id"),
                                "request_id": observation.get("request_id"),
                                "model_id": "lmdb-replay",
                                "action_rate_hz": float(
                                    args.action_rate_hz
                                ),
                            },
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                    )
                    continue
                now = time.monotonic()
                action_rate_hz = float(args.action_rate_hz)
                action_chunk = [
                    engine.step(
                        now + index / action_rate_hz,
                        observation,
                    )
                    for index in range(args.chunk_size)
                ]
                action = action_chunk[0]
                response = {
                    "protocol_version": PROTOCOL_VERSION,
                    "message_type": "action_chunk",
                    "session_id": observation.get("session_id"),
                    "request_id": observation.get("request_id"),
                    "model_id": "lmdb-replay",
                    "action_rate_hz": action_rate_hz,
                    "action_chunk": action_chunk,
                }
                socket.send(
                    pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
                )
                requests += 1
                if requests % 100 == 0:
                    log.info(
                        "requests=%d frame=%.2f/%d rebase=%s finished=%s",
                        requests,
                        action.get("frame_idx_f", -1.0),
                        action.get("frame_total", 0),
                        action.get("rebase_done", False),
                        action.get("finished", False),
                    )
            except Exception as exc:
                log.exception("Replay request failed")
                socket.send(pickle.dumps({"error": str(exc)}))
    except KeyboardInterrupt:
        log.info("Interrupted")
    finally:
        socket.close()
        context.term()


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="LMDB replay policy server")
    parser.add_argument("--episode-dir", required=True)
    parser.add_argument("--bind", default="tcp://0.0.0.0:5555")
    parser.add_argument(
        "--playback-rate-hz",
        "--rate-hz",
        dest="rate_hz",
        type=float,
        default=None,
        help=(
            "recorded source frames advanced per second; defaults to the "
            "episode frame rate"
        ),
    )
    parser.add_argument(
        "--arm-command-mode",
        choices=("eef", "joint"),
        default="eef",
        help="arm replay source: recorded EEF action (default) or motor qpos",
    )
    parser.add_argument(
        "--rebase",
        action="store_true",
        default=False,
        help="legacy relative-motion debug mode; absolute Replay leaves this off",
    )
    parser.add_argument("--no-rebase", action="store_false", dest="rebase")
    parser.add_argument("--start-hold-s", type=float, default=1.0)
    # Local replay is sampled into a 30 Hz action stream. Playback rate and
    # action rate are deliberately independent: e.g. 6 source frames/s is a
    # 0.2x replay of a 30 Hz recording, still published as smooth 30 Hz
    # waypoints for the 120 Hz deployment/controller path.
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--action-rate-hz", type=float, default=30.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.rate_hz is not None and args.rate_hz <= 0.0:
        parser.error("--playback-rate-hz must be positive")
    if args.start_hold_s < 0.0:
        parser.error("--start-hold-s must be non-negative")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.action_rate_hz <= 0.0:
        parser.error("--action-rate-hz must be positive")
    return args


def main(argv=None):
    run_server(_parse_args(argv))


if __name__ == "__main__":
    main()
