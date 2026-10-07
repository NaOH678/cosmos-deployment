"""LMDB + MP4 episode persistence compatible with dexmanip_tool."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import pickle
import shutil
from typing import Any, Mapping, Optional

import numpy as np

from .auxiliary_camera import AuxiliaryCameraWriter
from .schema import RobotLayout
from .teleop_diagnostics import OptionalDatasetSpec


SCALAR_PATHS = {
    "action": "action",
    "action_eef": "action_eef",
    "action_bases": "action_bases",
    "qpos": "/observations/qpos",
    "qvel": "/observations/qvel",
    "effort": "/observations/effort",
    "eef": "/observations/eef",
    "robot_base": "/observations/robot_base",
    "hand_joint_deg": "/observations/hand_joint_deg",
    "commanded_eef": "/diagnostics/commanded_eef",
    "arm_joint_command": "/diagnostics/arm_joint_command",
    "zsp": "/diagnostics/zsp",
}


def _require_lmdb():
    try:
        import lmdb
    except ImportError as exc:
        raise RuntimeError(
            "python lmdb is required; rebuild the project Docker image after "
            "the dependency update"
        ) from exc
    return lmdb


class EpisodeWriter:
    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        layout: RobotLayout,
        camera_names: list[str],
        frame_rate: float = 30.0,
        map_size: int = 1 << 40,
        video_fourcc: str = "mp4v",
        metadata: Optional[Mapping[str, Any]] = None,
        optional_specs: Optional[Mapping[str, OptionalDatasetSpec]] = None,
        auxiliary_camera: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.layout = layout
        self.camera_names = list(camera_names)
        self.frame_rate = float(frame_rate)
        self.map_size = int(map_size)
        self.video_fourcc = str(video_fourcc)
        self.user_metadata = dict(metadata or {})
        self.optional_specs = dict(optional_specs or {})
        self.auxiliary_camera_config = dict(auxiliary_camera or {})
        if self.frame_rate <= 0.0:
            raise ValueError("frame_rate must be positive")
        if len(self.video_fourcc) != 4:
            raise ValueError("video_fourcc must contain four characters")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episode_dir, self._final_dir = self._allocate_episode_path()
        self.episode_dir.mkdir(parents=True, exist_ok=False)
        self.lmdb_dir = self.episode_dir / "lmdb"
        self.lmdb_dir.mkdir()

        lmdb = _require_lmdb()
        self._env = lmdb.open(str(self.lmdb_dir), map_size=self.map_size)
        self._frames: dict[str, list[Any]] = {
            name: [] for name in SCALAR_PATHS.values()
        }
        self._optional_frames: dict[str, list[np.ndarray]] = {
            spec.path: [] for spec in self.optional_specs.values()
        }
        self._sync_timestamps: list[dict[str, Any]] = []
        self._video_writers: dict[str, Any] = {}
        self._video_info: dict[str, dict[str, Any]] = {}
        depth_stream_names = tuple(
            self.auxiliary_camera_config.get("depth_stream_names", ())
        )
        infrared_stream_names = tuple(
            self.auxiliary_camera_config.get(
                "infrared_stream_names", ()
            )
        )
        self._auxiliary_writer: Optional[AuxiliaryCameraWriter] = None
        self._auxiliary_init_error: Optional[str] = None
        self._auxiliary_enqueue_errors = 0
        if depth_stream_names or infrared_stream_names:
            try:
                self._auxiliary_writer = AuxiliaryCameraWriter(
                    self.episode_dir,
                    depth_stream_names=depth_stream_names,
                    infrared_stream_names=infrared_stream_names,
                    frame_rate=self.frame_rate,
                    map_size=int(
                        self.auxiliary_camera_config.get(
                            "map_size", self.map_size
                        )
                    ),
                    video_fourcc=str(
                        self.auxiliary_camera_config.get(
                            "video_fourcc", self.video_fourcc
                        )
                    ),
                    infrared_frame_rate=float(
                        self.auxiliary_camera_config.get(
                            "infrared_frame_rate", self.frame_rate
                        )
                    ),
                    queue_capacity=int(
                        self.auxiliary_camera_config.get(
                            "queue_capacity", 64
                        )
                    ),
                    png_compression=int(
                        self.auxiliary_camera_config.get(
                            "depth_png_compression", 1
                        )
                    ),
                    capture_metadata_path=self.auxiliary_camera_config.get(
                        "capture_metadata_path"
                    ),
                )
            except Exception as exc:
                self._auxiliary_init_error = str(exc)
        self._closed = False
        self._failed_reason: Optional[str] = None

    def _allocate_episode_path(self) -> tuple[Path, Path]:
        existing = sorted(self.output_dir.glob("episode_[0-9][0-9][0-9][0-9]_*"))
        next_index = 0
        for path in existing:
            try:
                next_index = max(next_index, int(path.name.split("_")[1]) + 1)
            except (IndexError, ValueError):
                continue
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_dir = self.output_dir / f"episode_{next_index:04d}_{stamp}"
        return Path(str(final_dir) + ".inprogress"), final_dir

    @property
    def step_count(self) -> int:
        return len(self._frames["action"])

    def auxiliary_status(self) -> dict[str, Any]:
        if self._auxiliary_writer is not None:
            return self._auxiliary_writer.status()
        if self._auxiliary_init_error is not None:
            return {
                "enabled": True,
                "initialization_error": self._auxiliary_init_error,
                "enqueue_errors": self._auxiliary_enqueue_errors,
            }
        return {"enabled": False}

    def append(
        self,
        scalar_frame: Mapping[str, np.ndarray],
        images: Mapping[str, np.ndarray],
        sync_timestamps: Mapping[str, Any],
        optional_frame: Optional[Mapping[str, Any]] = None,
        auxiliary_images: Optional[Mapping[str, np.ndarray]] = None,
        auxiliary_timestamps: Optional[Mapping[str, float]] = None,
        auxiliary_sequences: Optional[Mapping[str, int]] = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("episode is already closed")
        if self._failed_reason is not None:
            raise RuntimeError(
                f"episode writer is quarantined after a write failure: "
                f"{self._failed_reason}"
            )
        missing = [name for name in SCALAR_PATHS if name not in scalar_frame]
        if missing:
            raise ValueError(f"scalar frame missing fields: {missing}")
        if set(images) != set(self.camera_names):
            raise ValueError(
                f"camera set mismatch: expected {self.camera_names}, got {sorted(images)}"
            )

        expected_shapes = {
            "action": (self.layout.action_dim,),
            "action_eef": (self.layout.total_eef_dim,),
            "action_bases": (6,),
            "qpos": (self.layout.state_dim,),
            "qvel": (self.layout.state_dim,),
            "effort": (self.layout.state_dim,),
            "eef": (self.layout.total_eef_dim,),
            "robot_base": (6,),
            "hand_joint_deg": (self.layout.hand_dof * self.layout.n_sides,),
            "commanded_eef": (self.layout.total_eef_dim,),
            "arm_joint_command": (self.layout.arm_dof * self.layout.n_sides,),
            "zsp": (3 * self.layout.n_sides,),
        }
        prepared: dict[str, np.ndarray] = {}
        for field, expected_shape in expected_shapes.items():
            value = np.asarray(scalar_frame[field], dtype=np.float32).reshape(-1)
            if value.shape != expected_shape:
                raise ValueError(
                    f"{field} must have shape {expected_shape}, got {value.shape}"
                )
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{field} contains non-finite values")
            prepared[field] = value.copy()
        for name, frame in images.items():
            self._validate_video_frame(name, frame)
        prepared_optional = self._prepare_optional_frame(optional_frame or {})

        sid = self.step_count
        txn = self._env.begin(write=True)
        try:
            for field, path in SCALAR_PATHS.items():
                value = prepared[field]
                txn.put(f"{path}/{sid:06d}".encode(), pickle.dumps(value))
            sync_value = dict(sync_timestamps)
            txn.put(
                f"/sync_timestamps/{sid:06d}".encode(),
                pickle.dumps(sync_value),
            )
            txn.commit()
        except Exception:
            txn.abort()
            raise
        for field, path in SCALAR_PATHS.items():
            self._frames[path].append(prepared[field])
        for field, spec in self.optional_specs.items():
            self._optional_frames[spec.path].append(prepared_optional[field])
        self._sync_timestamps.append(sync_value)

        try:
            for name, frame in images.items():
                self._write_video_frame(name, frame)
        except Exception as exc:
            # Scalar data may already be committed.  Quarantine the whole
            # episode so it remains .inprogress instead of promoting a
            # scalar/video frame-count mismatch to a valid dataset.
            self._failed_reason = str(exc)
            raise
        if self._auxiliary_writer is not None:
            try:
                self._auxiliary_writer.enqueue(
                    sid,
                    auxiliary_images or {},
                    auxiliary_timestamps or {},
                    auxiliary_sequences,
                )
            except Exception:
                # Auxiliary analysis streams are explicitly best-effort. A
                # programming/runtime fault here must not invalidate a frame
                # whose original scalar/RGB payload was already committed.
                self._auxiliary_enqueue_errors += 1

    def _prepare_optional_frame(
        self, optional_frame: Mapping[str, Any]
    ) -> dict[str, np.ndarray]:
        """Validate diagnostics without ever rejecting a training frame.

        Diagnostic sources are explicitly best-effort.  A missing, malformed,
        or non-finite value is replaced by that field's neutral fill value;
        the corresponding ``available``/``valid`` datasets remain zero.
        """
        prepared: dict[str, np.ndarray] = {}
        for field, spec in self.optional_specs.items():
            fallback = spec.empty()
            if field not in optional_frame:
                prepared[field] = fallback
                continue
            try:
                value = np.asarray(optional_frame[field], dtype=np.dtype(spec.dtype))
                if value.shape != spec.shape:
                    raise ValueError(
                        f"{field} must have shape {spec.shape}, got {value.shape}"
                    )
                if np.issubdtype(value.dtype, np.number) and not np.all(
                    np.isfinite(value)
                ):
                    raise ValueError(f"{field} contains non-finite values")
                prepared[field] = value.copy()
            except (TypeError, ValueError, OverflowError):
                prepared[field] = fallback
        return prepared

    def _validate_video_frame(self, name: str, frame: np.ndarray) -> np.ndarray:
        image = np.asarray(frame)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"{name} image must be uint8 HxWx3 BGR, got {image.shape}")
        height, width = image.shape[:2]
        if name in self._video_info:
            expected = self._video_info[name]["size"]
            if [width, height] != expected:
                raise ValueError(
                    f"{name} image size changed: expected {expected}, "
                    f"got {[width, height]}"
                )
        return image

    def _write_video_frame(self, name: str, frame: np.ndarray) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required when cameras are enabled") from exc
        image = self._validate_video_frame(name, frame)
        height, width = image.shape[:2]
        writer = self._video_writers.get(name)
        if writer is None:
            video_dir = self.episode_dir / "videos"
            video_dir.mkdir(exist_ok=True)
            path = video_dir / f"{name}.mp4"
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*self.video_fourcc),
                self.frame_rate,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"failed to open video writer: {path}")
            self._video_writers[name] = writer
            self._video_info[name] = {
                "path": f"videos/{name}.mp4",
                "num_frames": 0,
                "size": [width, height],
                "fourcc": self.video_fourcc,
                "frame_rate": self.frame_rate,
            }
        writer.write(image)
        self._video_info[name]["num_frames"] += 1

    def finalize(self, extra_metadata: Optional[Mapping[str, Any]] = None) -> Path:
        if self._closed:
            return self._final_dir
        if self._failed_reason is not None:
            raise RuntimeError(
                f"refusing to finalize quarantined episode: {self._failed_reason}"
            )
        for writer in self._video_writers.values():
            writer.release()
        self._video_writers.clear()

        auxiliary_metadata = None
        if self._auxiliary_writer is not None:
            try:
                auxiliary_metadata = self._auxiliary_writer.finalize(
                    self.step_count
                )
            except Exception as exc:
                try:
                    self._auxiliary_writer.close_incomplete()
                except Exception:
                    pass
                auxiliary_metadata = {
                    "schema_version": 1,
                    "best_effort": True,
                    "finalize_error": str(exc),
                }
            auxiliary_metadata["enqueue_errors"] = (
                self._auxiliary_enqueue_errors
            )
        elif self._auxiliary_init_error is not None:
            auxiliary_metadata = {
                "schema_version": 1,
                "best_effort": True,
                "initialization_error": self._auxiliary_init_error,
                "enqueue_errors": self._auxiliary_enqueue_errors,
            }

        scalar_keys = [path.encode() for path in SCALAR_PATHS.values()]
        optional_keys = [
            spec.path.encode() for spec in self.optional_specs.values()
        ]
        image_keys = {
            f"observation/{name}/color_image": list(range(self.step_count))
            for name in self.camera_names
        }
        meta_info: dict[str, Any] = {
            **self.user_metadata,
            **dict(extra_metadata or {}),
            "schema_version": 1,
            "camera_names": self.camera_names,
            "num_steps": self.step_count,
            "frame_rate": self.frame_rate,
            "keys": {
                "scalar_data": scalar_keys,
                "teleop_data": optional_keys,
                "images": image_keys,
            },
            "robot_layout": self.layout.metadata(),
            "videos": self._video_info,
            "video_format": "mp4" if self.camera_names else None,
            "arm_mode": "dual" if self.layout.n_sides == 2 else "single",
        }
        if auxiliary_metadata is not None:
            meta_info["auxiliary_camera"] = auxiliary_metadata
        if self.optional_specs:
            diagnostics_metadata = dict(meta_info.get("teleop_diagnostics", {}))
            diagnostics_metadata.update({
                "schema_version": 1,
                "datasets": {
                    field: spec.metadata()
                    for field, spec in self.optional_specs.items()
                },
            })
            meta_info["teleop_diagnostics"] = diagnostics_metadata

        txn = self._env.begin(write=True)
        try:
            for path, values in self._frames.items():
                txn.put(path.encode(), pickle.dumps(values))
            for path, values in self._optional_frames.items():
                txn.put(path.encode(), pickle.dumps(np.stack(values, axis=0)))
            txn.put(b"/sync_timestamps", pickle.dumps(self._sync_timestamps))
            txn.put(b"__metadata__", pickle.dumps(meta_info))
            txn.put(b"meta_info", pickle.dumps(meta_info))
            txn.commit()
        except Exception:
            txn.abort()
            raise
        self._env.sync()
        self._env.close()

        with (self.episode_dir / "meta_info.pkl").open("wb") as handle:
            pickle.dump(meta_info, handle)
        with (self.episode_dir / "sync_timestamps.json").open("w") as handle:
            json.dump(self._sync_timestamps, handle, indent=2)

        self.episode_dir.rename(self._final_dir)
        self.episode_dir = self._final_dir
        self._closed = True
        return self._final_dir

    def close_incomplete(self) -> None:
        """Close handles while deliberately preserving the .inprogress data."""
        if self._closed:
            return
        if self._auxiliary_writer is not None:
            try:
                self._auxiliary_writer.close_incomplete()
            except Exception:
                pass
        for writer in self._video_writers.values():
            writer.release()
        self._video_writers.clear()
        try:
            self._env.sync()
        except Exception:
            pass
        try:
            self._env.close()
        except Exception:
            pass
        self._closed = True

    def discard(self) -> Path:
        """Close and delete this unfinalized episode.

        Only the writer-owned ``.inprogress`` directory is eligible.  A
        finalized episode is immutable and is never removed through this API.
        """
        path = self.episode_dir
        if self._closed:
            if path.exists() and path.name.endswith(".inprogress"):
                shutil.rmtree(path)
            return path
        if not path.name.endswith(".inprogress"):
            raise RuntimeError(
                f"refusing to discard a non-inprogress episode: {path}"
            )
        self.close_incomplete()
        if path.exists():
            shutil.rmtree(path)
        return path


def load_episode(episode_dir: str | os.PathLike[str]) -> tuple[dict[str, np.ndarray], dict]:
    path = Path(episode_dir).expanduser().resolve()
    meta_path = path / "meta_info.pkl"
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing meta_info.pkl: {path}")
    with meta_path.open("rb") as handle:
        metadata = pickle.load(handle)
    lmdb = _require_lmdb()
    env = lmdb.open(str(path / "lmdb"), readonly=True, lock=False, readahead=False)
    arrays: dict[str, np.ndarray] = {}
    try:
        with env.begin() as txn:
            for field, key in SCALAR_PATHS.items():
                raw = txn.get(key.encode())
                if raw is not None:
                    arrays[field] = np.asarray(pickle.loads(raw), dtype=np.float32)
            diagnostic_metadata = metadata.get("teleop_diagnostics", {})
            for field, info in diagnostic_metadata.get("datasets", {}).items():
                key = str(info["path"])
                raw = txn.get(key.encode())
                if raw is not None:
                    arrays[field] = np.asarray(
                        pickle.loads(raw), dtype=np.dtype(info["dtype"])
                    )
    finally:
        env.close()
    if "action" not in arrays:
        raise RuntimeError(f"episode has no action sequence: {path}")
    return arrays, metadata
