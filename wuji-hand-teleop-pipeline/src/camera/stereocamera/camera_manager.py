"""Direct camera manager for the Stage-D non-ROS image path."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time
from typing import Any, Mapping

import numpy as np
import yaml

from .shared_frames import DEFAULT_DIRECTORY, SharedFrameWriter, ring_path


LOGGER = logging.getLogger("wuji_camera_manager")
CAMERA_ORDER = ("head", "left_wrist", "right_wrist")
DEPTH_RING_NAMES = tuple(f"{name}_depth" for name in CAMERA_ORDER)
HEAD_INFRARED_RING_NAMES = ("head_ir_left", "head_ir_right")
ALL_RING_NAMES = CAMERA_ORDER + DEPTH_RING_NAMES + HEAD_INFRARED_RING_NAMES


def auxiliary_ring_names(
    name: str,
    camera: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return optional direct streams requested for one physical camera."""
    streams = dict(camera.get("streams", {}) or {})
    names = []
    if bool(streams.get("enable_depth", False)):
        names.append(f"{name}_depth")
    if name == "head" and bool(streams.get("enable_infrared", False)):
        names.extend(HEAD_INFRARED_RING_NAMES)
    return tuple(names)


class _RealSenseMultiSensorSession:
    """Keep a 30 Hz RGB sensor independent from lower-rate stereo streams."""

    def __init__(
        self,
        *,
        color_sensor,
        color_profile,
        stereo_sensor,
        auxiliary_profiles: Mapping[str, Any],
    ) -> None:
        self.color_sensor = color_sensor
        self.color_profile = color_profile
        self.stereo_sensor = stereo_sensor
        self.auxiliary_profiles = dict(auxiliary_profiles)
        self._color_queue: queue.Queue[
            tuple[np.ndarray, int, int]
        ] = queue.Queue(maxsize=2)
        self._auxiliary_lock = threading.Lock()
        self._latest_auxiliary: dict[
            str, tuple[np.ndarray, int, int]
        ] = {}
        self._color_started = False
        self._stereo_started = False
        self._open_and_start()

    @staticmethod
    def _replace_latest(target: queue.Queue, value: Any) -> None:
        try:
            target.put_nowait(value)
            return
        except queue.Full:
            pass
        try:
            target.get_nowait()
        except queue.Empty:
            pass
        else:
            target.task_done()
        try:
            target.put_nowait(value)
        except queue.Full:
            pass

    def _callback(self, frame) -> None:
        captured_monotonic_ns = time.monotonic_ns()
        captured_system_ns = time.time_ns()
        try:
            image = np.asanyarray(frame.get_data()).copy()
            stream_type = frame.profile.stream_type()
            stream_index = frame.profile.stream_index()
        except Exception:
            return
        if str(stream_type) == "stream.color":
            self._replace_latest(
                self._color_queue,
                (image, captured_monotonic_ns, captured_system_ns),
            )
            return
        if str(stream_type) == "stream.depth":
            stream_name = next(
                (
                    name
                    for name in self.auxiliary_profiles
                    if name.endswith("_depth")
                ),
                "",
            )
        elif str(stream_type) == "stream.infrared":
            stream_name = (
                "head_ir_left" if stream_index == 1 else "head_ir_right"
            )
        else:
            stream_name = ""
        if stream_name:
            with self._auxiliary_lock:
                self._latest_auxiliary[stream_name] = (
                    image,
                    captured_monotonic_ns,
                    captured_system_ns,
                )

    def _open_and_start(self) -> None:
        try:
            self.color_sensor.open(self.color_profile)
            if self.auxiliary_profiles:
                self.stereo_sensor.open(
                    list(self.auxiliary_profiles.values())
                )
            self.color_sensor.start(self._callback)
            self._color_started = True
            if self.auxiliary_profiles:
                self.stereo_sensor.start(self._callback)
                self._stereo_started = True
        except Exception:
            self.stop()
            raise

    def read(
        self,
        timeout_ms: int,
    ) -> tuple[
        tuple[np.ndarray, int, int],
        dict[str, tuple[np.ndarray, int, int]],
    ]:
        color = self._color_queue.get(
            timeout=max(0.001, float(timeout_ms) / 1000.0)
        )
        self._color_queue.task_done()
        with self._auxiliary_lock:
            auxiliary = self._latest_auxiliary
            self._latest_auxiliary = {}
        return color, auxiliary

    def stop(self) -> None:
        if self._stereo_started:
            try:
                self.stereo_sensor.stop()
            except Exception:
                pass
            self._stereo_started = False
        if self._color_started:
            try:
                self.color_sensor.stop()
            except Exception:
                pass
            self._color_started = False
        if self.auxiliary_profiles:
            try:
                self.stereo_sensor.close()
            except Exception:
                pass
        try:
            self.color_sensor.close()
        except Exception:
            pass


def load_camera_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError("camera configuration must be a mapping")
    return value


def enabled_camera_configs(
    config: Mapping[str, Any],
    active_arm: str = "both",
) -> dict[str, dict[str, Any]]:
    cameras = config.get("cameras", {})
    if not isinstance(cameras, Mapping):
        raise ValueError("camera configuration must contain a cameras mapping")
    normalized_arm = str(active_arm).strip().lower()
    if normalized_arm not in ("both", "left", "right"):
        raise ValueError(
            "active_arm must be one of ('both', 'left', 'right'), "
            f"got {active_arm!r}"
        )
    selected_roles = (
        set(CAMERA_ORDER)
        if normalized_arm == "both"
        else {"head", f"{normalized_arm}_wrist"}
    )
    selected: dict[str, dict[str, Any]] = {}
    for name in CAMERA_ORDER:
        if name not in selected_roles:
            continue
        camera = cameras.get(name)
        if not isinstance(camera, Mapping) or not bool(camera.get("enabled", False)):
            continue
        serial = str(camera.get("serial_number", "")).strip()
        if serial.startswith("YOUR_"):
            LOGGER.warning("%s skipped: serial number is still a placeholder", name)
            continue
        selected[name] = dict(camera)
    return selected


def _udev_properties(device: str) -> dict[str, str]:
    try:
        result = subprocess.run(
            ["udevadm", "info", "--query=property", "--name", device],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}
    properties = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            properties[key] = value
    return properties


def candidate_video_devices(camera: Mapping[str, Any]) -> list[str]:
    """Return stable path first, then V4L nodes matching the configured serial."""
    candidates: list[str] = []
    configured = str(camera.get("video_device", "")).strip()
    if configured and Path(configured).exists():
        candidates.append(str(Path(configured).resolve()))

    serial = str(camera.get("serial_number", "")).strip()
    if serial and not serial.startswith("YOUR_"):
        for path in sorted(glob.glob("/dev/v4l/by-id/*")):
            if serial in Path(path).name and Path(path).exists():
                candidates.append(str(Path(path).resolve()))
        for path in sorted(glob.glob("/dev/video*")):
            properties = _udev_properties(path)
            serial_values = (
                properties.get("ID_SERIAL_SHORT", ""),
                properties.get("ID_SERIAL", ""),
                properties.get("ID_PATH", ""),
            )
            if any(serial in value for value in serial_values):
                candidates.append(path)

    # UVC cameras without a serial are required to have a stable configured
    # symlink.  Falling back to an arbitrary /dev/videoN would silently swap
    # camera roles after replug and is therefore deliberately forbidden.
    return list(dict.fromkeys(candidates))


class CameraCapture:
    def __init__(
        self,
        name: str,
        config: Mapping[str, Any],
        *,
        directory: str | os.PathLike[str],
        capacity: int,
        retry_s: float,
    ) -> None:
        self.name = name
        self.config = dict(config)
        resolution = dict(self.config.get("resolution", {}) or {})
        self.width = int(resolution.get("width", 640))
        self.height = int(resolution.get("height", 480))
        self.fps = float(resolution.get("fps", 30))
        self.camera_type = str(self.config.get("type", "usb")).lower()
        self.side_by_side = bool(self.config.get("stereo_side_by_side", False))
        streams = dict(self.config.get("streams", {}) or {})
        self.enable_depth = bool(streams.get("enable_depth", False))
        self.enable_infrared = (
            self.name == "head"
            and bool(streams.get("enable_infrared", False))
        )
        self.depth_fps = int(
            streams.get("depth_fps", round(self.fps))
        )
        self.infrared_fps = int(
            streams.get("infrared_fps", self.depth_fps)
        )
        if min(self.depth_fps, self.infrared_fps) <= 0:
            raise ValueError("auxiliary RealSense FPS must be positive")
        output_width = self.width // 2 if self.side_by_side else self.width
        self.writer = SharedFrameWriter(
            name,
            width=output_width,
            height=self.height,
            channels=3,
            capacity=capacity,
            directory=directory,
        )
        self.auxiliary_writers: dict[str, SharedFrameWriter] = {}
        if self.enable_depth:
            depth_name = f"{self.name}_depth"
            self.auxiliary_writers[depth_name] = SharedFrameWriter(
                depth_name,
                width=self.width,
                height=self.height,
                channels=1,
                dtype=np.uint16,
                capacity=capacity,
                directory=directory,
            )
        if self.enable_infrared:
            for infrared_name in HEAD_INFRARED_RING_NAMES:
                self.auxiliary_writers[infrared_name] = SharedFrameWriter(
                    infrared_name,
                    width=self.width,
                    height=self.height,
                    channels=1,
                    dtype=np.uint8,
                    capacity=capacity,
                    directory=directory,
                )
        self.retry_s = max(0.1, float(retry_s))
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"camera-{name}",
        )
        self.frames = 0
        self.last_device: str | None = None
        self._active_depth = False
        self._active_infrared = False
        self.stream_metadata: dict[str, Any] = {
            "camera_name": self.name,
            "camera_type": self.camera_type,
            "serial_number": str(
                self.config.get("serial_number", "")
            ).strip(),
            "requested": {
                "color": True,
                "depth": self.enable_depth,
                "infrared_left": self.enable_infrared,
                "infrared_right": self.enable_infrared,
            },
            "active": {
                "color": False,
                "depth": False,
                "infrared_left": False,
                "infrared_right": False,
            },
        }

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=3.0)
        self.writer.close()
        for writer in self.auxiliary_writers.values():
            writer.close()

    def _open(self):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required by the direct camera manager") from exc

        devices = candidate_video_devices(self.config)
        if not devices:
            return None, None
        for device in devices:
            cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2000)
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                cap.release()
                continue
            # A RealSense exposes multiple V4L nodes under one serial.  Only
            # accept a node that actually returns an image at the requested
            # camera role; this avoids binding to metadata/depth-only nodes.
            for _ in range(8):
                ok, frame = cap.read()
                if ok and frame is not None and frame.size:
                    return cap, device
            cap.release()
        return None, None

    def _realsense_configuration(
        self,
        rs,
        serial: str,
        *,
        depth: bool,
        infrared: bool,
    ):
        configuration = rs.config()
        configuration.enable_device(serial)
        configuration.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.bgr8,
            int(round(self.fps)),
        )
        if depth:
            configuration.enable_stream(
                rs.stream.depth,
                self.width,
                self.height,
                rs.format.z16,
                self.depth_fps,
            )
        if infrared:
            for index in (1, 2):
                configuration.enable_stream(
                    rs.stream.infrared,
                    index,
                    self.width,
                    self.height,
                    rs.format.y8,
                    self.infrared_fps,
                )
        return configuration

    @staticmethod
    def _intrinsics(profile) -> dict[str, Any]:
        intrinsics = profile.as_video_stream_profile().get_intrinsics()
        return {
            "width": int(intrinsics.width),
            "height": int(intrinsics.height),
            "fx": float(intrinsics.fx),
            "fy": float(intrinsics.fy),
            "ppx": float(intrinsics.ppx),
            "ppy": float(intrinsics.ppy),
            "model": str(intrinsics.model),
            "coeffs": [float(value) for value in intrinsics.coeffs],
        }

    @staticmethod
    def _extrinsics(source, target) -> dict[str, Any]:
        extrinsics = source.get_extrinsics_to(target)
        return {
            "rotation": [float(value) for value in extrinsics.rotation],
            "translation": [
                float(value) for value in extrinsics.translation
            ],
        }

    def _realsense_metadata_from_profiles(
        self,
        device,
        *,
        color_profile,
        depth_profile=None,
        infrared_profiles: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        streams: dict[str, Any] = {}
        color_profile = color_profile.as_video_stream_profile()
        streams["color"] = {
            "format": "bgr8",
            "fps": int(color_profile.fps()),
            "intrinsics": self._intrinsics(color_profile),
        }
        if depth_profile is not None:
            depth_profile = depth_profile.as_video_stream_profile()
            try:
                depth_scale = float(
                    device.first_depth_sensor().get_depth_scale()
                )
            except Exception:
                depth_scale = None
            streams["depth"] = {
                "format": "z16",
                "dtype": "uint16",
                "fps": int(depth_profile.fps()),
                "depth_scale_m": depth_scale,
                "intrinsics": self._intrinsics(depth_profile),
                "extrinsics_to_color": self._extrinsics(
                    depth_profile, color_profile
                ),
            }
        for side, infrared_profile in (
            infrared_profiles or {}
        ).items():
            infrared_profile = (
                infrared_profile.as_video_stream_profile()
            )
            streams[side] = {
                "format": "y8",
                "dtype": "uint8",
                "fps": int(infrared_profile.fps()),
                "intrinsics": self._intrinsics(infrared_profile),
                "extrinsics_to_color": self._extrinsics(
                    infrared_profile, color_profile
                ),
            }
        infrared_active = bool(infrared_profiles)
        return {
            "camera_name": self.name,
            "camera_type": self.camera_type,
            "serial_number": str(
                self.config.get("serial_number", "")
            ).strip(),
            "requested": {
                "color": True,
                "depth": self.enable_depth,
                "infrared_left": self.enable_infrared,
                "infrared_right": self.enable_infrared,
            },
            "active": {
                "color": True,
                "depth": depth_profile is not None,
                "infrared_left": infrared_active,
                "infrared_right": infrared_active,
            },
            "streams": streams,
        }

    def _realsense_metadata(
        self,
        rs,
        profile,
        *,
        depth: bool,
        infrared: bool,
    ) -> dict[str, Any]:
        infrared_profiles = {}
        if infrared:
            infrared_profiles = {
                side: profile.get_stream(
                    rs.stream.infrared, index
                )
                for index, side in (
                    (1, "infrared_left"),
                    (2, "infrared_right"),
                )
            }
        return self._realsense_metadata_from_profiles(
            profile.get_device(),
            color_profile=profile.get_stream(rs.stream.color),
            depth_profile=(
                profile.get_stream(rs.stream.depth) if depth else None
            ),
            infrared_profiles=infrared_profiles,
        )

    @staticmethod
    def _find_video_profile(
        sensor,
        *,
        stream_type,
        stream_index: int,
        pixel_format,
        width: int,
        height: int,
        fps: int,
    ):
        for profile in sensor.get_stream_profiles():
            if (
                profile.is_video_stream_profile()
                and profile.stream_type() == stream_type
                and profile.stream_index() == stream_index
                and profile.format() == pixel_format
                and profile.as_video_stream_profile().width() == width
                and profile.as_video_stream_profile().height() == height
                and profile.fps() == fps
            ):
                return profile
        raise RuntimeError(
            "RealSense profile unavailable: "
            f"{stream_type}/{stream_index} {width}x{height}@{fps} "
            f"{pixel_format}"
        )

    def _open_realsense_multi_sensor(self, rs, serial: str):
        context = rs.context()
        device = next(
            (
                item
                for item in context.query_devices()
                if item.get_info(rs.camera_info.serial_number) == serial
            ),
            None,
        )
        if device is None:
            return None, None
        color_sensor = None
        stereo_sensor = None
        for sensor in device.query_sensors():
            stream_types = {
                profile.stream_type()
                for profile in sensor.get_stream_profiles()
            }
            if rs.stream.color in stream_types:
                color_sensor = sensor
            if rs.stream.depth in stream_types:
                stereo_sensor = sensor
        if color_sensor is None or stereo_sensor is None:
            return None, None
        color_profile = self._find_video_profile(
            color_sensor,
            stream_type=rs.stream.color,
            stream_index=0,
            pixel_format=rs.format.bgr8,
            width=self.width,
            height=self.height,
            fps=int(round(self.fps)),
        )
        attempts = []
        if self.enable_depth and self.enable_infrared:
            attempts.append((True, True))
        if self.enable_depth:
            attempts.append((True, False))
        attempts.append((False, False))
        for depth, infrared in dict.fromkeys(attempts):
            auxiliary_profiles = {}
            session = None
            try:
                if depth:
                    auxiliary_profiles[f"{self.name}_depth"] = (
                        self._find_video_profile(
                            stereo_sensor,
                            stream_type=rs.stream.depth,
                            stream_index=0,
                            pixel_format=rs.format.z16,
                            width=self.width,
                            height=self.height,
                            fps=self.depth_fps,
                        )
                    )
                if infrared:
                    for index, stream_name in (
                        (1, "head_ir_left"),
                        (2, "head_ir_right"),
                    ):
                        auxiliary_profiles[stream_name] = (
                            self._find_video_profile(
                                stereo_sensor,
                                stream_type=rs.stream.infrared,
                                stream_index=index,
                                pixel_format=rs.format.y8,
                                width=self.width,
                                height=self.height,
                                fps=self.infrared_fps,
                            )
                        )
                session = _RealSenseMultiSensorSession(
                    color_sensor=color_sensor,
                    color_profile=color_profile,
                    stereo_sensor=stereo_sensor,
                    auxiliary_profiles=auxiliary_profiles,
                )
                _color, initial_auxiliary = session.read(3000)
                expected = set(auxiliary_profiles)
                deadline = time.monotonic() + 3.0
                observed = set(initial_auxiliary)
                while expected - observed and time.monotonic() < deadline:
                    _color, auxiliary = session.read(1000)
                    observed.update(auxiliary)
                if expected - observed:
                    raise RuntimeError(
                        "missing initial auxiliary streams: "
                        f"{sorted(expected - observed)}"
                    )
                self._active_depth = depth
                self._active_infrared = infrared
                depth_profile = auxiliary_profiles.get(
                    f"{self.name}_depth"
                )
                infrared_profiles = {
                    "infrared_left": auxiliary_profiles[
                        "head_ir_left"
                    ],
                    "infrared_right": auxiliary_profiles[
                        "head_ir_right"
                    ],
                } if infrared else {}
                try:
                    self.stream_metadata = (
                        self._realsense_metadata_from_profiles(
                            device,
                            color_profile=color_profile,
                            depth_profile=depth_profile,
                            infrared_profiles=infrared_profiles,
                        )
                    )
                    self.stream_metadata["usb_type"] = (
                        device.get_info(
                            rs.camera_info.usb_type_descriptor
                        )
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "%s calibration metadata unavailable: %s",
                        self.name,
                        exc,
                    )
                if (
                    depth != self.enable_depth
                    or infrared != self.enable_infrared
                ):
                    LOGGER.warning(
                        "%s auxiliary profile fallback: depth=%s "
                        "infrared=%s",
                        self.name,
                        depth,
                        infrared,
                    )
                return session, f"RealSense serial {serial}"
            except Exception:
                if session is not None:
                    try:
                        session.stop()
                    except Exception:
                        pass
        self._active_depth = False
        self._active_infrared = False
        return None, None

    def _open_realsense(self):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 is required for serial-number-safe direct "
                "RealSense capture; rebuild the Stage-D Docker image"
            ) from exc
        serial = str(self.config.get("serial_number", "")).strip()
        if not serial or serial.startswith("YOUR_"):
            return None, None
        if self.enable_infrared:
            return self._open_realsense_multi_sensor(rs, serial)
        attempts = []
        if self.enable_depth and self.enable_infrared:
            attempts.append((True, True))
        if self.enable_depth:
            attempts.append((True, False))
        attempts.append((False, False))
        for depth, infrared in dict.fromkeys(attempts):
            pipeline = rs.pipeline()
            configuration = self._realsense_configuration(
                rs,
                serial,
                depth=depth,
                infrared=infrared,
            )
            try:
                profile = pipeline.start(configuration)
                frames = pipeline.wait_for_frames(timeout_ms=3000)
                if not frames.get_color_frame():
                    raise RuntimeError(
                        "device did not return a color frame"
                    )
                if depth and not frames.get_depth_frame():
                    raise RuntimeError(
                        "device did not return a depth frame"
                    )
                if infrared and (
                    not frames.get_infrared_frame(1)
                    or not frames.get_infrared_frame(2)
                ):
                    raise RuntimeError(
                        "device did not return both infrared frames"
                    )
                self._active_depth = depth
                self._active_infrared = infrared
                try:
                    self.stream_metadata = self._realsense_metadata(
                        rs,
                        profile,
                        depth=depth,
                        infrared=infrared,
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "%s calibration metadata unavailable: %s",
                        self.name,
                        exc,
                    )
                if (
                    depth != self.enable_depth
                    or infrared != self.enable_infrared
                ):
                    LOGGER.warning(
                        "%s auxiliary profile fallback: depth=%s "
                        "infrared=%s",
                        self.name,
                        depth,
                        infrared,
                    )
                return pipeline, f"RealSense serial {serial}"
            except Exception:
                try:
                    pipeline.stop()
                except Exception:
                    pass
        self._active_depth = False
        self._active_infrared = False
        return None, None

    @staticmethod
    def _bgr(frame: np.ndarray) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required by the direct camera manager") from exc
        image = np.asarray(frame)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"unsupported camera frame shape: {image.shape}")
        return np.ascontiguousarray(image, dtype=np.uint8)

    def _run(self) -> None:
        cap = None
        use_realsense = self.camera_type in ("d405", "d435", "d435i")
        next_log = 0.0
        while not self.stop_event.is_set():
            if cap is None:
                try:
                    if use_realsense:
                        cap, device = self._open_realsense()
                    else:
                        cap, device = self._open()
                except RuntimeError as exc:
                    cap, device = None, None
                    now = time.monotonic()
                    if now >= next_log:
                        LOGGER.error("%s unavailable: %s", self.name, exc)
                        next_log = now + 5.0
                if cap is None:
                    now = time.monotonic()
                    if now >= next_log:
                        LOGGER.warning(
                            "%s offline: no usable device for serial=%s path=%s",
                            self.name,
                            self.config.get("serial_number", "<none>"),
                            self.config.get("video_device", "<none>"),
                        )
                        next_log = now + 5.0
                    self.stop_event.wait(self.retry_s)
                    continue
                self.last_device = device
                LOGGER.info(
                    "%s online: %s requested=%dx%d@%.1f",
                    self.name,
                    device,
                    self.width,
                    self.height,
                    self.fps,
                )
            auxiliary_frames: dict[
                str, tuple[np.ndarray, int, int]
            ] = {}
            if use_realsense:
                try:
                    if isinstance(
                        cap, _RealSenseMultiSensorSession
                    ):
                        (
                            (
                                frame,
                                captured_monotonic_ns,
                                captured_system_ns,
                            ),
                            auxiliary_frames,
                        ) = cap.read(2000)
                        ok = frame is not None
                    else:
                        frames = cap.wait_for_frames(timeout_ms=2000)
                        color = frames.get_color_frame()
                        ok = bool(color)
                        frame = (
                            np.asanyarray(color.get_data())
                            if ok
                            else None
                        )
                        captured_monotonic_ns = time.monotonic_ns()
                        captured_system_ns = time.time_ns()
                        if self._active_depth:
                            depth = frames.get_depth_frame()
                            if depth:
                                auxiliary_frames[
                                    f"{self.name}_depth"
                                ] = (
                                    np.asanyarray(depth.get_data()),
                                    captured_monotonic_ns,
                                    captured_system_ns,
                                )
                            else:
                                self.auxiliary_writers[
                                    f"{self.name}_depth"
                                ].note_capture_failure()
                except Exception:
                    ok, frame = False, None
            else:
                ok, frame = cap.read()
                captured_monotonic_ns = time.monotonic_ns()
                captured_system_ns = time.time_ns()
            if not ok or frame is None:
                self.writer.note_capture_failure()
                for writer in self.auxiliary_writers.values():
                    writer.note_capture_failure()
                if use_realsense:
                    try:
                        cap.stop()
                    except Exception:
                        pass
                else:
                    cap.release()
                cap = None
                self.stop_event.wait(0.05)
                continue
            try:
                image = self._bgr(frame)
                if self.side_by_side:
                    image = image[:, : image.shape[1] // 2]
                self.writer.write(
                    image,
                    monotonic_ns=captured_monotonic_ns,
                    system_ns=captured_system_ns,
                )
                self.frames += 1
            except Exception as exc:
                self.writer.note_capture_failure()
                now = time.monotonic()
                if now >= next_log:
                    LOGGER.error("%s frame rejected: %s", self.name, exc)
                    next_log = now + 5.0
            for stream_name, auxiliary_sample in auxiliary_frames.items():
                writer = self.auxiliary_writers.get(stream_name)
                if writer is None:
                    continue
                (
                    auxiliary_frame,
                    auxiliary_monotonic_ns,
                    auxiliary_system_ns,
                ) = auxiliary_sample
                try:
                    writer.write(
                        np.ascontiguousarray(
                            auxiliary_frame,
                            dtype=writer.dtype,
                        ),
                        monotonic_ns=auxiliary_monotonic_ns,
                        system_ns=auxiliary_system_ns,
                    )
                except Exception as exc:
                    writer.note_capture_failure()
                    now = time.monotonic()
                    if now >= next_log:
                        LOGGER.error(
                            "%s frame rejected: %s",
                            stream_name,
                            exc,
                        )
                        next_log = now + 5.0
        if cap is not None:
            if use_realsense:
                try:
                    cap.stop()
                except Exception:
                    pass
            else:
                cap.release()


class CameraManager:
    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        directory: str | os.PathLike[str] = DEFAULT_DIRECTORY,
        active_arm: str = "both",
    ) -> None:
        direct = dict(config.get("direct_transport", {}) or {})
        global_config = dict(config.get("global", {}) or {})
        self.directory = Path(directory)
        self.capacity = int(direct.get("ring_capacity", 64))
        self.retry_s = float(direct.get("reconnect_interval_s", 1.0))
        self.startup_delay_s = float(global_config.get("startup_delay", 0.0))
        if self.capacity <= 0:
            raise ValueError("direct_transport.ring_capacity must be positive")
        if self.startup_delay_s < 0.0:
            raise ValueError("global.startup_delay must be non-negative")
        selected = enabled_camera_configs(config, active_arm=active_arm)
        self.directory.mkdir(parents=True, exist_ok=True)

        # Remove stale paths for roles this manager will not produce. Existing
        # mappings remain valid but consumers see the path disappear and mark
        # the role offline instead of displaying a previous session's image.
        active_ring_names = set(selected)
        for name, camera in selected.items():
            active_ring_names.update(auxiliary_ring_names(name, camera))
        for name in ALL_RING_NAMES:
            if name not in active_ring_names:
                try:
                    ring_path(name, self.directory).unlink()
                except FileNotFoundError:
                    pass
        self.captures = [
            CameraCapture(
                name,
                camera,
                directory=self.directory,
                capacity=self.capacity,
                retry_s=self.retry_s,
            )
            for name, camera in selected.items()
        ]
        self.metadata_path = self.directory / "camera_metadata.json"

    def _publish_metadata(self) -> None:
        payload = {
            "schema_version": 1,
            "generated_system_ns": time.time_ns(),
            "cameras": {
                capture.name: capture.stream_metadata
                for capture in self.captures
            },
        }
        temporary = self.metadata_path.with_name(
            f".{self.metadata_path.name}.{os.getpid()}.tmp"
        )
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
            os.replace(temporary, self.metadata_path)
        except Exception as exc:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            LOGGER.warning("camera metadata publish failed: %s", exc)

    def run(self, stop_event: threading.Event) -> None:
        for index, capture in enumerate(self.captures):
            if stop_event.is_set():
                break
            capture.start()
            self._publish_metadata()
            if (
                index + 1 < len(self.captures)
                and stop_event.wait(self.startup_delay_s)
            ):
                break
        self._publish_metadata()
        LOGGER.info(
            "Direct Camera Manager ready: cameras=%s ring=%s capacity=%d",
            [capture.name for capture in self.captures],
            self.directory,
            self.capacity,
        )
        try:
            while not stop_event.wait(1.0):
                self._publish_metadata()
        finally:
            self._publish_metadata()
            for capture in self.captures:
                capture.stop()
            LOGGER.info("Direct Camera Manager stopped")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Direct non-ROS camera manager for Wuji data collection"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--shared-memory-dir", default=str(DEFAULT_DIRECTORY))
    parser.add_argument(
        "--active-arm",
        default="both",
        choices=("both", "left", "right"),
        help="Start head plus only the wrist camera needed by this arm mode",
    )
    args, _unknown = parser.parse_known_args(argv)
    return args


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(levelname)s: %(message)s",
    )
    args = _parse_args(argv)
    manager = CameraManager(
        load_camera_config(args.config),
        directory=args.shared_memory_dir,
        active_arm=args.active_arm,
    )
    stop_event = threading.Event()

    def request_stop(_signal, _frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    manager.run(stop_event)


if __name__ == "__main__":
    main()
