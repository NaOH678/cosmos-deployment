"""Pure helpers for the standalone data-collection GUI."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re


TASK_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
GUI_HANDOFF_RAMP_SEC = 6.0


def repository_root() -> Path:
    override = os.environ.get("WUJI_REPOSITORY_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[4]


def dataset_root() -> Path:
    override = os.environ.get("WUJI_DATASET_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return repository_root() / "datasets" / "tianji_wuji"


def validate_task_name(name: str) -> str:
    normalized = str(name).strip()
    if not TASK_NAME_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Task name must start with an ASCII letter or digit and contain "
            "only ASCII letters, digits, '_' or '-'"
        )
    return normalized


class TaskCatalog:
    def __init__(self, root: Path | None = None):
        self.root = (root or dataset_root()).expanduser().resolve()

    def list_tasks(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(
            path.name
            for path in self.root.iterdir()
            if path.is_dir()
            and not path.is_symlink()
            and not path.name.startswith("episode_")
            and not path.name.endswith(".inprogress")
            and TASK_NAME_PATTERN.fullmatch(path.name)
        )

    def task_path(self, name: str) -> Path:
        return self.root / validate_task_name(name)

    def create_task(self, name: str) -> Path:
        path = self.task_path(name)
        path.mkdir(parents=True, exist_ok=False)
        return path


@dataclass(frozen=True)
class SessionOptions:
    task_name: str
    active_hand: str = "both"
    camera_enabled: bool = True
    camera_transport: str = "direct"
    handoff_ramp_sec: float = GUI_HANDOFF_RAMP_SEC

    def command(self, repo_root: Path | None = None) -> list[str]:
        task = validate_task_name(self.task_name)
        if self.active_hand not in ("both", "left", "right"):
            raise ValueError(
                f"unsupported active hand mode: {self.active_hand}"
            )
        if self.handoff_ramp_sec < 0.0:
            raise ValueError("handoff_ramp_sec cannot be negative")
        if self.camera_transport not in ("direct", "ros"):
            raise ValueError(
                f"unsupported camera transport: {self.camera_transport}"
            )
        root = (repo_root or repository_root()).resolve()
        command = [
            str(root / "src" / "scripts" / "start_record_session.sh"),
            self.active_hand,
            "--task",
            task,
            "--handoff-ramp-sec",
            str(self.handoff_ramp_sec),
            "--camera-transport",
            self.camera_transport,
        ]
        if self.camera_enabled:
            command.append("--with-camera")
        return command


@dataclass(frozen=True)
class RecorderState:
    state: str = "offline"
    recording: bool = False
    pending: bool = False
    capture_has_finished: bool = False

    @classmethod
    def from_status(cls, status: dict | None) -> "RecorderState":
        if not status:
            return cls()
        state = str(status.get("state", "idle"))
        return cls(
            state=state,
            recording=bool(status.get("recording", False)),
            pending=state == "pending_save",
            capture_has_finished=bool(
                status.get("capture_has_finished", state == "pending_save")
            ),
        )

    @property
    def pedal1_action(self) -> str:
        return "finish" if self.recording else "start"

    @property
    def pedal2_allowed(self) -> bool:
        return self.pending

    @property
    def pedal3_allowed(self) -> bool:
        return not self.recording and self.capture_has_finished


def format_arm_snapshot(snapshot: dict | None) -> str:
    """Format the lightweight controller cache for the data GUI."""
    if not snapshot:
        return "硬件缓存：等待状态"
    if "error" in snapshot:
        return "硬件缓存错误：" + str(snapshot["error"])

    arms = snapshot.get("arms")
    if not isinstance(arms, dict):
        return "硬件缓存：状态格式无效"
    parts = []
    for side in ("left", "right"):
        arm = arms.get(side, {})
        if not isinstance(arm, dict):
            arm = {}
        state = arm.get("state")
        error = arm.get("err_code")
        parts.append(
            f"{side}: state={'?' if state is None else state}, "
            f"err={'?' if error is None else error}"
        )

    age = snapshot.get("feedback_age_s")
    if isinstance(age, (int, float)):
        parts.append(f"反馈年龄={max(0.0, float(age)):.3f}s")
    if bool(snapshot.get("sdk_fault_latched", False)):
        parts.append("SDK故障已锁存")
    return "硬件缓存：" + " | ".join(parts)
