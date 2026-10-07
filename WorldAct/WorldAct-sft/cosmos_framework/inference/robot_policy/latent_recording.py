"""Opt-in final video latent capture; no VAE decoding on the serving path."""
import json
import logging
import queue
import threading
from pathlib import Path

_LOG = logging.getLogger(__name__)

class LatentRecorder:
    def __init__(self, directory, capacity=2):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.queue = queue.Queue(maxsize=capacity)
        self.counts = dict(enqueued=0, written=0, dropped=0, errors=0)
        self.lock = threading.Lock()
        self.closed = False
        self.thread = threading.Thread(target=self._run, daemon=True, name="video-latent-writer")
        self.thread.start()

    def submit(self, tensor, metadata):
        import torch
        with self.lock:
            if self.closed or self.queue.full():
                self.counts["dropped"] += 1
                return
            # Retain this sample's storage until the worker finishes the transfer.
            # CUDA Graphs are disabled in this service; outputs are not reused buffers.
            tensor = tensor.detach()
            event = None
            if tensor.is_cuda:
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(tensor.device))
            self.counts["enqueued"] += 1
            self.queue.put_nowait((self.counts["enqueued"], tensor, event, dict(metadata)))

    def _run(self):
        import torch
        while True:
            item = self.queue.get()
            if item is None:
                self.queue.task_done()
                break
            seq, tensor, event, metadata = item
            try:
                if tensor.is_cuda:
                    with torch.cuda.device(tensor.device):
                        stream = torch.cuda.Stream(device=tensor.device)
                        with torch.cuda.stream(stream):
                            stream.wait_event(event)
                            cpu = torch.empty_like(tensor, device="cpu", pin_memory=True)
                            cpu.copy_(tensor, non_blocking=True)
                            tensor.record_stream(stream)
                        stream.synchronize()  # Worker only; never on the request thread.
                else:
                    cpu = tensor.clone()
                metadata.update(shape=list(cpu.shape), dtype=str(cpu.dtype), layout="BCTHW",
                                content="final generated vision latent; prediction, not camera recording")
                stem = f"{seq:08d}"
                temp = self.directory / (stem + ".pt.tmp")
                torch.save(cpu, temp)
                temp.replace(self.directory / (stem + ".pt"))
                (self.directory / (stem + ".json")).write_text(json.dumps(metadata, indent=2) + "\n")
                with self.lock:
                    self.counts["written"] += 1
            except Exception:
                with self.lock:
                    self.counts["errors"] += 1
                _LOG.exception("Video latent recording failed")
            finally:
                self.queue.task_done()
                del tensor, item
        (self.directory / "stats.json").write_text(json.dumps(self.counts, indent=2) + "\n")

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
        self.queue.put(None)
        self.thread.join()


def install(adapter, directory):
    """Install only on rank zero, after warmup and before HTTP starts."""
    recorder = LatentRecorder(directory)
    local = threading.local()
    original_infer = adapter._infer_recorded
    original_generate = adapter.model.generate_samples_from_batch

    def infer(observation, *args, **kwargs):
        local.metadata = {k: observation.get(k) for k in
                          ("session_id", "request_id", "timestamp", "client_monotonic")}
        local.metadata.update(model_id=adapter.config.deployment.model_id,
                              model_config=adapter.config.model.config_file,
                              action_rate_hz=adapter.config.deployment.action_rate_hz)
        try:
            return original_infer(observation, *args, **kwargs)
        finally:
            local.metadata = None

    def generate(*args, **kwargs):
        samples = original_generate(*args, **kwargs)
        metadata = getattr(local, "metadata", None)
        if metadata is not None:
            try:
                recorder.submit(samples["vision"][0], metadata)
            except Exception:
                _LOG.exception("Could not enqueue video latent")
        return samples

    adapter._infer_recorded = infer
    adapter.model.generate_samples_from_batch = generate
    return recorder
