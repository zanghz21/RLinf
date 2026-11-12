# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataset, image, and action boundaries for a Cosmos3 inference loop."""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image


def _decode_image(value: Any, dataset_root: Path) -> Image.Image:
    from PIL import Image

    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        path = value.get("path")
    else:
        image_bytes = value if isinstance(value, (bytes, bytearray)) else None
        path = value if isinstance(value, str) else None
    if image_bytes:
        source = io.BytesIO(image_bytes)
    elif path:
        source = Path(path)
        if not source.is_absolute():
            source = dataset_root / source
    else:
        raise ValueError("Expected a dataset image with embedded bytes or a file path")
    with Image.open(source) as image:
        return image.convert("RGB")


def load_lerobot_observation(
    dataset_root: Path, episode_index: int, frame_index: int
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    """Read one two-camera observation and its instruction from LeRobot v2.

    Args:
        dataset_root: Dataset directory containing ``meta`` and episode Parquet files.
        episode_index: Dataset episode identifier, not a suite-local reset identifier.
        frame_index: Value of the episode's ``frame_index`` column.

    Returns:
        External image, wrist image, and source metadata. Recorded actions and
        robot state are not read.
    """
    import pyarrow.parquet as pq

    if episode_index < 0 or frame_index < 0:
        raise ValueError(
            f"Expected nonnegative episode/frame indices, got {episode_index}/{frame_index}"
        )
    dataset_root = Path(dataset_root).resolve()
    info = json.loads((dataset_root / "meta/info.json").read_text())
    version = str(info.get("codebase_version", ""))
    if not version.startswith("v2."):
        raise ValueError(f"Expected LeRobot v2 metadata, got {version!r}")
    chunk_size = int(info["chunks_size"])
    if chunk_size <= 0:
        raise ValueError(f"Expected positive chunks_size, got {chunk_size}")
    episode_path = dataset_root / info["data_path"].format(
        episode_chunk=episode_index // chunk_size, episode_index=episode_index
    )
    table = pq.read_table(
        episode_path,
        columns=["image", "wrist_image", "task_index", "episode_index", "frame_index"],
        filters=[("frame_index", "=", frame_index)],
    )
    if table.num_rows != 1:
        raise ValueError(
            f"Expected one row for episode {episode_index}, frame {frame_index}; "
            f"found {table.num_rows} in {episode_path}"
        )
    row = table.to_pylist()[0]
    if int(row["episode_index"]) != episode_index:
        raise ValueError(
            f"Expected episode_index={episode_index}, got {row['episode_index']}"
        )
    task_index = int(row["task_index"])
    tasks = [
        json.loads(line)
        for line in (dataset_root / "meta/tasks.jsonl").read_text().splitlines()
        if line.strip()
    ]
    matches = [task for task in tasks if task["task_index"] == task_index]
    if len(matches) != 1 or not str(matches[0].get("task", "")).strip():
        raise ValueError(
            f"Expected one nonempty instruction for task_index={task_index}"
        )
    source_fps = float(info["fps"])
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"Expected positive source FPS, got {source_fps}")
    metadata = {
        "dataset_root": str(dataset_root),
        "episode_path": str(episode_path),
        "episode_index": episode_index,
        "frame_index": frame_index,
        "task_index": task_index,
        "instruction": matches[0]["task"],
        "source_fps": source_fps,
    }
    return (
        _decode_image(row["image"], dataset_root),
        _decode_image(row["wrist_image"], dataset_root),
        metadata,
    )


def prepare_canvas(
    external: Image.Image, wrist: Image.Image, width: int, height: int
) -> tuple[Image.Image, dict[str, Any]]:
    """Concatenate external/wrist views and resize/pad once like the policy adapter.

    Returns:
        RGB canvas and its source sizes, camera order, resize, and padding metadata.
    """
    import numpy as np
    from PIL import Image

    if width <= 0 or height <= 0:
        raise ValueError(f"Expected positive canvas dimensions, got {width}x{height}")
    if external.size != wrist.size:
        raise ValueError(
            f"Expected matching camera sizes, got {external.size} and {wrist.size}"
        )
    canvas = np.concatenate(
        [np.asarray(external.convert("RGB")), np.asarray(wrist.convert("RGB"))],
        axis=1,
    )
    source_h, source_w = canvas.shape[:2]
    scale = min(width / source_w, height / source_h, 1.0)
    resized_w = max(1, min(int(scale * source_w + 0.5), width))
    resized_h = max(1, min(int(scale * source_h + 0.5), height))
    if (resized_w, resized_h) != (source_w, source_h):
        canvas = np.asarray(
            Image.fromarray(canvas).resize(
                (resized_w, resized_h), resample=Image.Resampling.BICUBIC
            )
        )
    pad_w, pad_h = width - resized_w, height - resized_h
    padding_mode = None
    if pad_w or pad_h:
        padding_mode = "edge" if pad_w >= resized_w or pad_h >= resized_h else "reflect"
        canvas = np.pad(canvas, ((0, pad_h), (0, pad_w), (0, 0)), mode=padding_mode)
    return Image.fromarray(canvas), {
        "camera_order": ["external", "wrist"],
        "source_view_size": list(external.size),
        "source_canvas_size": [source_w, source_h],
        "resized_size": [resized_w, resized_h],
        "canvas_size": [width, height],
        "padding_right": pad_w,
        "padding_bottom": pad_h,
        "padding_mode": padding_mode,
    }


def augment_prompt(task: str) -> str:
    """Describe the same external-left/wrist-right canvas as the policy adapter."""
    prompt = task.strip()
    if not prompt:
        raise ValueError("Expected a nonempty task instruction")
    if not prompt.endswith("."):
        prompt += "."
    return (
        prompt
        + " This video contains concatenated views from multiple camera perspectives."
        + " The left half shows the third-person view; the right half shows"
        + " the wrist-mounted camera."
    )


def image_to_data_uri(image: Image.Image) -> str:
    """Encode an RGB observation as a lossless PNG data URI."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )


def validate_native_actions(
    values: Any, horizon: int, raw_action_dim: int
) -> np.ndarray:
    """Validate a model action chunk without clipping or changing its representation."""
    import numpy as np

    if horizon <= 0 or raw_action_dim <= 0:
        raise ValueError(
            f"Expected positive action horizon/width, got {horizon}/{raw_action_dim}"
        )
    actions = np.asarray(values, dtype=np.float32)
    expected = (horizon, raw_action_dim)
    if actions.shape != expected:
        raise ValueError(f"Expected action shape {expected}, got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("Expected finite model actions, found NaN or infinity")
    return actions


def read_video_frames(path: Path) -> np.ndarray:
    """Decode a generated video to uint8 RGB frames without resizing."""
    import imageio.v2 as imageio
    import numpy as np

    with imageio.get_reader(str(path), format="ffmpeg") as reader:
        frames = list(reader.iter_data())
    if not frames:
        raise ValueError(f"Expected at least one video frame in {path}")
    video = np.stack(frames)
    if video.ndim != 4 or video.shape[-1] != 3 or video.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 RGB video [T,H,W,3], got {video.shape} and {video.dtype}"
        )
    return video


def save_video(path: Path, frames: np.ndarray, fps: float) -> None:
    """Write uint8 RGB frames at their original size to an MP4 artifact."""
    import imageio.v2 as imageio
    import numpy as np

    frames = np.asarray(frames)
    if (
        frames.ndim != 4
        or frames.shape[0] == 0
        or frames.shape[-1] != 3
        or frames.dtype != np.uint8
    ):
        raise ValueError(
            f"Expected nonempty uint8 RGB video [T,H,W,3], got {frames.shape} and {frames.dtype}"
        )
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Expected positive video FPS, got {fps}")
    with imageio.get_writer(
        str(path),
        format="ffmpeg",
        fps=fps,
        codec="libx264",
        macro_block_size=1,
        pixelformat="yuv444p",
        ffmpeg_log_level="error",
    ) as writer:
        for frame in frames:
            writer.append_data(frame)
