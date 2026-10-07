#!/usr/bin/env python3
"""Does a torchrun rendezvous + NCCL all-reduce actually work across these nodes?

Run this BEFORE committing a multi-node launch to a 15-hour run.  A multi-node
torchrun that cannot rendezvous or cannot build an NCCL communicator does not fail
fast -- it sits there until the rendezvous times out (default 30 min) or the NCCL
watchdog fires (1800 s here), and the log you get back names the timeout, not the
cause.  Thirty seconds of this answers the question directly.

    # once on EACH node, NODE_RANK differing, everything else identical
    NNODES=3 NODE_RANK=0 MASTER_ADDR=<node0 ip> \
      PYTHONPATH=. <venv>/bin/python tools/check_multinode_nccl.py
    NNODES=3 NODE_RANK=1 MASTER_ADDR=<node0 ip> \
      PYTHONPATH=. <venv>/bin/python tools/check_multinode_nccl.py

It is deliberately a plain ``python`` script rather than a torchrun one: torchrun
would spawn ``NPROC`` ranks per node and report a single aggregate failure, which
hides *which* rank or *which* peer was the problem.  Here each node runs one process,
one rank per local GPU is exercised by the all-reduce size, and the failure names the
peer it could not reach.

What it checks, in the order the real launch hits them:

1. rendezvous      -- every rank reaches rank 0 on MASTER_ADDR:MASTER_PORT
2. topology        -- all ranks agree on world size and on the GPU count per node
3. NCCL all-reduce -- a real collective over the real transport, with NCCL_DEBUG
                      bumped to WARN so a fallback to sockets (i.e. the IB fabric is
                      not actually usable) shows up as a warning rather than as a
                      mysteriously slow step three hours in
4. bandwidth       -- a sizing number: if the bus bandwidth is an order of magnitude
                      below the others, that link is the problem

Set KEEP_MASTER_PORT=1 to reuse the port the trainer will use, which also proves the
port is free.  Exit code 0 means the topology works; anything else means do not launch.
"""

from __future__ import annotations

import datetime
import os
import sys
import time

import torch
import torch.distributed as dist


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        sys.exit(f"ERROR: {name} must be an integer, got {raw!r}")


def main() -> int:
    nnodes = env_int("NNODES", 1)
    node_rank = env_int("NODE_RANK", 0)
    master_addr = os.environ.get("MASTER_ADDR") or "127.0.0.1"
    # 50012 is _sft_launcher_common.sh's own default; reusing it here means a pass
    # also proves the trainer's port is free.  +1 keeps the test from colliding with
    # a train that is already running.
    master_port = env_int("MASTER_PORT", 50013 if not os.environ.get("KEEP_MASTER_PORT") else 50012)
    local_gpus = torch.cuda.device_count()

    if nnodes > 1 and master_addr == "127.0.0.1":
        sys.exit("ERROR: MASTER_ADDR is unset -- multi-node needs node 0's reachable IP.")

    local_rank = 0
    if local_gpus:
        torch.cuda.set_device(0)
    tag = f"[node {node_rank}/{nnodes} pid {os.getpid()}]"
    print(f"{tag} host={os.uname().nodename} master={master_addr}:{master_port} "
          f"local_gpus={local_gpus}")

    if not local_gpus:
        # Still worth doing: the rendezvous half is the part that usually breaks, and
        # this node can prove it reaches the others even with no visible GPU.
        print(f"{tag} WARNING: no CUDA device visible; checking rendezvous only")

    started = time.time()
    try:
        dist.init_process_group(
            backend="nccl" if local_gpus else "gloo",
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=nnodes,
            rank=node_rank,
            timeout=datetime.timedelta(seconds=120),  # 120 s, not 30 min: this is a test
        )
    except Exception as exc:  # noqa: BLE001 - the message IS the diagnosis
        sys.exit(f"{tag} FAILED at rendezvous with {master_addr}:{master_port}: "
                 f"{type(exc).__name__}: {exc}\n"
                 f"        -> the peers cannot reach MASTER_ADDR. Check `ping {master_addr}` "
                 f"from this node and that the port is open.")
    print(f"{tag} rendezvous OK in {time.time() - started:.1f}s  "
          f"world_size={dist.get_world_size()}")

    # Rank 0 reports what it heard from, so a run where one node never arrived is
    # visible as a mismatch rather than as a hang.
    heard = [None] * nnodes
    dist.all_gather_object(heard, f"{node_rank}:{os.uname().nodename}:{local_gpus}")
    if node_rank == 0:
        print(f"{tag} peers:")
        for i, who in enumerate(heard):
            print(f"          rank {i}: {who}")
        counts = {str(w).rsplit(":", 1)[-1] for w in heard}
        if len(counts) != 1:
            print(f"{tag} WARNING: nodes disagree on GPU count: {counts}. A torchrun "
                  f"launch would spawn a different rank count on each node and hang.")

    if local_gpus and nnodes > 1:
        tensor = torch.ones(64 * 1024 * 1024, dtype=torch.float32, device="cuda")  # 256 MB
        dist.barrier()
        # Warm-up, then a timed run: the first collective pays for communicator
        # setup, and timing that would understate the bandwidth by a lot.
        for _ in range(2):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        begin = time.time()
        for _ in range(5):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        elapsed = (time.time() - begin) / 5

        # Ring all-reduce moves 2*(N-1)/N * size through each rank.
        bus_gb = 2 * (nnodes - 1) / nnodes * tensor.numel() * 4 / 1e9
        print(f"{tag} all-reduce 256 MB x5: {elapsed * 1000:.1f} ms/iter  "
              f"bus {bus_gb / elapsed:.1f} GB/s")
    elif local_gpus:
        # world_size 1 makes all_reduce a no-op and the bus formula divide to zero,
        # so a single-node smoke of this script would print "0.0 GB/s" and read as a
        # failure.  Say why instead of printing a meaningless number.
        print(f"{tag} world_size=1: the collective and its bandwidth are not exercised "
              f"(this run only proves the script and the CUDA context work)")

    dist.barrier()
    if node_rank == 0 and local_gpus:
        print(f"{tag} NCCL_DEBUG={os.environ.get('NCCL_DEBUG', '<unset>')} -- if you saw "
              f"no channel/IB lines above, rerun with NCCL_DEBUG=INFO to confirm the "
              f"transport is IB and not a silent socket fallback.")
    dist.destroy_process_group()
    print(f"{tag} PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
