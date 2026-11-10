"""Physics-step telemetry and streaming, simulation-time robot videos.

Call ``capture`` after every physics step, then ``finish`` with the evaluator's
summary and action log. Videos use ``sample['time_s']`` rather than the number
of controller calls: a 0.002 s physics step and 25 fps produce one frame per
20 physics steps, so playback time follows simulation time.

The caller selects MUJOCO_GL before importing MuJoCo. Recording does not change
the backend, robot state, camera poses, or success criteria.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


class RobotRecorder:
    """Record one episode; use as a context manager to close video resources.

    Each camera is written to its own ``<camera>.mp4``. Set
    ``cameras=('overview',)`` for only the main view. Rendering happens only
    when a video frame is due; rendered video frames are not accumulated in
    memory. Numerical trajectory samples are retained until ``finish``.

    ``trajectory.npz`` stores every input sample field along its first axis.
    ``phase`` is a Unicode array, so loading never needs ``allow_pickle=True``.
    When video is enabled, ``video_time_s`` contains scheduled frame timestamps
    and ``video_sample_time_s`` the physical sample used for each frame. The
    encoded duration is ``len(video_time_s) / fps``; it differs from elapsed
    simulation time by at most one video frame for continuously sampled runs.
    """

    def __init__(
        self,
        out_dir: str | Path,
        model: Any,
        video: bool = False,
        cameras: Sequence[str] = ("overview", "contact"),
        fps: float = 25,
        width: int = 960,
        height: int = 720,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        if video and (width % 2 or height % 2):
            raise ValueError("H.264 yuv420p video requires even width and height")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.video = video
        self.cameras = tuple(cameras)
        self.fps = float(fps)
        self.width = width
        self.height = height
        self._columns: dict[str, list[Any]] = {}
        self._renderer: Any = None
        self._writers: dict[str, Any] = {}
        self._resources = ExitStack()
        self._last_frames: dict[str, np.ndarray] = {}
        self._backend: Any = None
        self._start_time_s: float | None = None
        self._last_time_s: float | None = None
        self._last_render_time_s: float | None = None
        self._video_times_s: list[float] = []
        self._video_sample_times_s: list[float] = []
        self._closed = False

    @property
    def video_frame_count(self) -> int:
        return len(self._video_times_s)

    def _open_video(self) -> None:
        import imageio.v2 as imageio
        import mujoco

        self.model.vis.global_.offwidth = max(
            self.model.vis.global_.offwidth, self.width
        )
        self.model.vis.global_.offheight = max(
            self.model.vis.global_.offheight, self.height
        )
        self._renderer = mujoco.Renderer(
            self.model, height=self.height, width=self.width
        )
        self._resources.callback(self._renderer.close)
        for camera in self.cameras:
            self._writers[camera] = imageio.get_writer(
                str(self.out_dir / f"{camera}.mp4"),
                format="FFMPEG",
                fps=self.fps,
                codec="libx264",
                quality=8,
                pixelformat="yuv420p",
                macro_block_size=1,
                ffmpeg_params=["-movflags", "+faststart"],
            )
            self._resources.callback(self._writers[camera].close)

    def _render(self) -> dict[str, np.ndarray]:
        frames = {}
        for camera in self.cameras:
            self._renderer.update_scene(self._backend.data, camera=camera)
            frames[camera] = self._renderer.render().copy()
        self._last_frames = frames
        self._last_render_time_s = self._last_time_s
        return frames

    def capture(self, backend: Any, sample: Mapping[str, Any]) -> None:
        """Append telemetry and render due frames from this physical state."""
        if self._closed:
            raise RuntimeError("Cannot capture after recorder.close()")
        time_s = float(sample["time_s"])
        if self._last_time_s is not None and time_s < self._last_time_s:
            raise ValueError("Use a new RobotRecorder after resetting simulation time")
        self._backend = backend
        self._last_time_s = time_s
        for key, value in sample.items():
            copied = str(value) if key == "phase" else np.array(value, copy=True)
            self._columns.setdefault(key, []).append(copied)
        if not self.video:
            return
        if self._start_time_s is None:
            self._start_time_s = time_s
        next_time = self._start_time_s + self.video_frame_count / self.fps
        if time_s + 1e-9 < next_time:
            return

        try:
            if self._renderer is None:
                self._open_video()
            frames = self._render()
            if not self._video_times_s:
                import imageio.v2 as imageio

                for camera, frame in frames.items():
                    imageio.imwrite(self.out_dir / f"{camera}_first.png", frame)
            # Normally exactly one iteration. Repeating a frame also preserves
            # real-time playback if callers deliberately sample more sparsely.
            while next_time <= time_s + 1e-9:
                for camera, frame in frames.items():
                    self._writers[camera].append_data(frame)
                self._video_times_s.append(next_time)
                self._video_sample_times_s.append(time_s)
                next_time = self._start_time_s + self.video_frame_count / self.fps
        except BaseException:
            self.close()
            raise

    def finish(self, summary: Mapping[str, Any], actions: Sequence[Any]) -> None:
        """Save the complete episode and close all rendering/encoding resources."""
        try:
            arrays = {
                key: np.asarray(values, dtype=str) if key == "phase" else np.stack(values)
                for key, values in self._columns.items()
            }
            if self.video:
                arrays["video_time_s"] = np.asarray(self._video_times_s, dtype=float)
                arrays["video_sample_time_s"] = np.asarray(
                    self._video_sample_times_s, dtype=float
                )
            np.savez_compressed(self.out_dir / "trajectory.npz", **arrays)
            for filename, content in (("metrics.json", summary), ("actions.json", actions)):
                (self.out_dir / filename).write_text(
                    json.dumps(content, indent=2, ensure_ascii=False, default=_json_value)
                    + "\n",
                    encoding="utf-8",
                )
            if self._renderer is not None and not self._closed:
                import imageio.v2 as imageio

                if self._last_render_time_s != self._last_time_s:
                    self._render()
                for camera, frame in self._last_frames.items():
                    imageio.imwrite(self.out_dir / f"{camera}_last.png", frame)
        finally:
            self.close()

    def close(self) -> None:
        """Flush writers and release OpenGL resources; safe to call repeatedly."""
        if self._closed:
            return
        self._closed = True
        try:
            self._resources.close()
        finally:
            self._last_frames.clear()

    def __enter__(self) -> RobotRecorder:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
