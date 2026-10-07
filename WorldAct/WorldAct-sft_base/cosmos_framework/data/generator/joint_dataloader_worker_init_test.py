"""A DataLoader worker must not oversubscribe its rank's CPU partition.

Measured on the training node: a ``build_pointflow_batch`` that takes 2.5 ms on
idle CPUs stretched to ~265 ms of wall clock once eight workers each opened
OpenCV's default per-core thread pool, which left the GPU idle for the
difference.  These tests pin the worker configuration that prevents it.
"""

import os

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from cosmos_framework.data.generator.joint_dataloader import limit_dataloader_worker_threads


class _Probe(Dataset):
    """Reports the thread and priority settings seen inside a worker."""

    def __len__(self):
        return 4

    def __getitem__(self, index):
        import cv2

        return np.array([torch.get_num_threads(), cv2.getNumThreads(), os.nice(0)])


def _worker_settings(worker_init_fn, num_workers=2):
    loader = DataLoader(_Probe(), batch_size=1, num_workers=num_workers, worker_init_fn=worker_init_fn)
    return np.concatenate([batch.numpy() for batch in loader]).reshape(-1, 3)


def test_worker_init_caps_opencv_threads_and_lowers_priority():
    settings = _worker_settings(limit_dataloader_worker_threads)
    assert settings[:, 1].max() == 1, "OpenCV workers must decode on one thread"
    assert settings[:, 0].max() == 1, "torch workers must keep one intra-op thread"
    assert settings[:, 2].min() >= 1, "workers must yield to the trainer"
    assert len(set(settings[:, 2])) == 1, "every worker gets the same priority"


def test_worker_init_is_optional_and_default_leaves_opencv_alone():
    """Guards the premise: without the hook OpenCV uses one thread per core."""
    default = _worker_settings(None)
    assert default[:, 2].max() == 0
    if os.cpu_count() and os.cpu_count() > 1:
        assert default[:, 1].max() > 1


@pytest.mark.L0
def test_worker_init_is_picklable():
    """DataLoader sends the hook to workers by pickling, so it must be importable."""
    import pickle

    restored = pickle.loads(pickle.dumps(limit_dataloader_worker_threads))
    assert restored is limit_dataloader_worker_threads
