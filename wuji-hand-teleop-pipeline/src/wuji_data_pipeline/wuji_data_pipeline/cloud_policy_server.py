"""Generic ZMQ cloud-policy host for Tianji + Wuji.

Model repositories provide a small ``module:factory`` adapter.  The factory
returns an object with ``infer(observation)``; observations contain decoded BGR
images and hardware-neutral Tianji/Wuji state dictionaries.  The adapter may
return structured actions or an ``Nx54`` action array.

This server uses pickle for compatibility with replay and therefore must run
behind a trusted VPN or authenticated tunnel.  Never expose its port directly
to the public internet.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import pickle
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .deployment_protocol import (
    PROTOCOL_VERSION,
    decode_observation_images,
    import_policy_factory,
)
from .schema import RobotLayout, normalize_quaternion_xyzw


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
)
log = logging.getLogger("wuji_cloud_policy")


def _require_zmq():
    try:
        import zmq
    except ImportError as exc:
        raise RuntimeError("pyzmq is required by cloud_policy_server") from exc
    return zmq


def action_vector_to_mapping(
    vector: Any,
    layout: RobotLayout | None = None,
) -> dict[str, Any]:
    """Convert one canonical 54-D EEF+hand vector into wire actions."""

    robot_layout = layout or RobotLayout()
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    if value.shape != (robot_layout.action_dim,):
        raise ValueError(
            f"policy action must have {robot_layout.action_dim} values, "
            f"got {value.shape}"
        )
    if not np.all(np.isfinite(value)):
        raise ValueError("policy action contains NaN or infinity")
    result: dict[str, Any] = {}
    cursor = 0
    for side in robot_layout.sides:
        eef = value[cursor : cursor + robot_layout.eef_dof]
        cursor += robot_layout.eef_dof
        hand = value[cursor : cursor + robot_layout.hand_dof]
        cursor += robot_layout.hand_dof
        result[f"arm_action_{side}"] = {
            "ee_pos": eef[:3].copy(),
            "ee_quat": normalize_quaternion_xyzw(eef[3:7]),
        }
        result[f"hand_action_{side}"] = hand.tolist()
    return result


def normalize_policy_actions(
    output: Any,
    *,
    layout: RobotLayout,
) -> list[Mapping[str, Any]]:
    """Normalize common adapter outputs into structured action mappings."""

    if isinstance(output, Mapping):
        if "action_chunk" in output:
            output = output["action_chunk"]
        elif "actions" in output:
            output = output["actions"]
        else:
            return [output]
    if isinstance(output, np.ndarray):
        array = np.asarray(output, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2:
            raise ValueError(f"policy actions must be Nx54, got {array.shape}")
        return [action_vector_to_mapping(row, layout) for row in array]
    if isinstance(output, Sequence) and not isinstance(
        output, (str, bytes, bytearray)
    ):
        result = []
        for item in output:
            if isinstance(item, Mapping):
                result.append(item)
            else:
                result.append(action_vector_to_mapping(item, layout))
        if not result:
            raise ValueError("policy returned an empty action chunk")
        return result
    raise ValueError(f"unsupported policy output type: {type(output).__name__}")


class HoldPolicy:
    """Safe protocol/transport test adapter: hold measured EEF and hand pose."""

    model_id = "hold-current-pose"

    def __init__(self, chunk_size: int) -> None:
        self.chunk_size = int(chunk_size)

    @staticmethod
    def _action(observation: Mapping[str, Any]) -> dict[str, Any]:
        action: dict[str, Any] = {}
        for side in ("left", "right"):
            arm = observation.get(f"arm_state_{side}")
            hand = observation.get(f"hand_state_{side}")
            if not isinstance(arm, Mapping) or not isinstance(hand, Mapping):
                raise ValueError(f"observation is missing {side} robot state")
            eef = np.asarray(arm.get("eef"), dtype=np.float32).reshape(-1)
            hand_rad = np.asarray(
                hand.get("joint_pos"), dtype=np.float32
            ).reshape(-1)
            if eef.shape != (7,) or hand_rad.shape != (20,):
                raise ValueError(f"invalid {side} robot state dimensions")
            action[f"arm_action_{side}"] = {
                "ee_pos": eef[:3].copy(),
                "ee_quat": normalize_quaternion_xyzw(eef[3:7]),
            }
            action[f"hand_action_{side}"] = np.degrees(hand_rad).tolist()
        return action

    def infer(self, observation: Mapping[str, Any]):
        action = self._action(observation)
        return [action for _ in range(self.chunk_size)]


def _load_policy(args):
    if not args.adapter:
        return HoldPolicy(args.chunk_size)
    factory = import_policy_factory(args.adapter)
    if not callable(factory):
        policy = factory
    else:
        parameters = inspect.signature(factory).parameters.values()
        positional = [
            parameter
            for parameter in parameters
            if parameter.kind
            in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]
        policy = factory(args.adapter_config) if positional else factory()
    if not hasattr(policy, "infer") or not callable(policy.infer):
        raise TypeError("policy adapter factory must return an object with infer()")
    return policy


def run_server(args) -> None:
    policy = _load_policy(args)
    model_id = str(getattr(policy, "model_id", args.adapter or "custom-policy"))
    layout = RobotLayout()
    zmq = _require_zmq()
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(args.bind)
    log.info(
        "Cloud policy listening on %s model=%s action_rate=%.2fHz",
        args.bind,
        model_id,
        args.action_rate_hz,
    )
    requests = 0
    try:
        while True:
            raw = socket.recv()
            request_id = None
            session_id = None
            try:
                received_at = time.time()
                observation = pickle.loads(raw)
                if not isinstance(observation, Mapping):
                    raise ValueError("observation is not a mapping")
                request_id = observation.get("request_id")
                session_id = observation.get("session_id")
                client_protocol = observation.get("protocol_version")
                if (
                    client_protocol is not None
                    and int(client_protocol) != PROTOCOL_VERSION
                ):
                    raise ValueError(
                        f"unsupported protocol_version={client_protocol}; "
                        f"expected {PROTOCOL_VERSION}"
                    )
                if observation.get("message_type") == "hello":
                    socket.send(
                        pickle.dumps(
                            {
                                "protocol_version": PROTOCOL_VERSION,
                                "message_type": "hello_ack",
                                "session_id": session_id,
                                "request_id": request_id,
                                "model_id": model_id,
                                "action_rate_hz": float(args.action_rate_hz),
                            },
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                    )
                    continue
                decoded = decode_observation_images(observation)
                inference_started = time.time()
                output = policy.infer(decoded)
                inference_finished = time.time()
                actions = normalize_policy_actions(output, layout=layout)
                response = {
                    "protocol_version": PROTOCOL_VERSION,
                    "message_type": "action_chunk",
                    "session_id": session_id,
                    "request_id": request_id,
                    "model_id": model_id,
                    "action_rate_hz": float(args.action_rate_hz),
                    "action_chunk": actions,
                    "server_timing": {
                        "received_wall_time": received_at,
                        "inference_started_wall_time": inference_started,
                        "inference_finished_wall_time": inference_finished,
                        "inference_ms": (
                            inference_finished - inference_started
                        ) * 1000.0,
                    },
                }
                socket.send(
                    pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
                )
                requests += 1
                if requests % 100 == 0:
                    log.info(
                        "requests=%d last_chunk=%d inference=%.1fms",
                        requests,
                        len(actions),
                        response["server_timing"]["inference_ms"],
                    )
            except Exception as exc:
                log.exception("Policy request failed")
                socket.send(
                    pickle.dumps(
                        {
                            "protocol_version": PROTOCOL_VERSION,
                            "session_id": session_id,
                            "request_id": request_id,
                            "error": str(exc),
                        },
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                )
    except KeyboardInterrupt:
        log.info("Interrupted")
    finally:
        socket.close()
        context.term()


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generic Tianji/Wuji cloud policy server"
    )
    parser.add_argument("--bind", default="tcp://0.0.0.0:5555")
    parser.add_argument(
        "--adapter",
        default=None,
        help="model adapter factory in module:attribute form",
    )
    parser.add_argument("--adapter-config", default=None)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--action-rate-hz", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.chunk_size <= 0 or args.action_rate_hz <= 0.0:
        parser.error("chunk size and action rate must be positive")
    return args


def main(argv=None):
    run_server(_parse_args(argv))


if __name__ == "__main__":
    main()
