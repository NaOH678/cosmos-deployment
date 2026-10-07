"""Sparse modality metrics must not desynchronize distributed loss logging."""

from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cosmos_framework.callbacks.wandb_log import _LossRecord, _reduce_loss_records


def _record(value):
    return _LossRecord(loss=torch.tensor(float(value)), iter_count=1)


def _sparse_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        # Different keys AND insertion orders; missing ranks must not dilute
        # averages. These are actual collectives, not a mocked reduction.
        records = (
            {"loss_shared": _record(2), "fk_loss_sigma_low": _record(4)}
            if rank == 0
            else {"fk_loss_sigma_high": _record(10), "loss_shared": _record(6)}
        )
        assert _reduce_loss_records(records) == {
            "fk_loss_sigma_high": 10.0,
            "fk_loss_sigma_low": 4.0,
            "loss_shared": 4.0,
        }
        # A bin can disappear on every rank after a logging reset. A previously
        # absent rank may be the only contributor at the next logging interval.
        if rank == 0:
            records["fk_loss_sigma_high"] = _record(14)
        else:
            records["loss_shared"] = _record(8)
        assert _reduce_loss_records(records) == {
            "fk_loss_sigma_high": 14.0,
            "fk_loss_sigma_low": 0.0,
            "loss_shared": 8.0,
        }
    finally:
        dist.destroy_process_group()


def test_sparse_losses_across_two_ranks(tmp_path):
    mp.spawn(_sparse_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)
