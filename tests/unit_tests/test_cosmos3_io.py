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

"""Observation and action boundaries used by the Cosmos3 closed loop."""

import base64
import io
import json

import pytest

from toolkits.cosmos3.closed_loop_io import (
    augment_prompt,
    image_to_data_uri,
    load_lerobot_observation,
    prepare_canvas,
    read_video_frames,
    save_video,
    validate_native_actions,
)

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")


@pytest.fixture
def lerobot_dataset(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    (tmp_path / "meta").mkdir()
    (tmp_path / "data/chunk-001").mkdir(parents=True)
    (tmp_path / "meta/info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.0",
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "fps": 10,
            }
        )
    )
    (tmp_path / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 99, "task": "another task"})
        + "\n"
        + json.dumps({"task_index": 30, "task": "put the bowl on the plate"})
        + "\n"
    )

    def encoded(color):
        buffer = io.BytesIO()
        Image.new("RGB", (16, 16), color).save(buffer, format="PNG")
        return {"bytes": buffer.getvalue(), "path": "unused.png"}

    rows = [
        {
            "image": encoded("blue"),
            "wrist_image": encoded("white"),
            "task_index": 99,
            "episode_index": 1261,
            "frame_index": 2,
            "actions": "recorded actions must not be read",
        },
        {
            "image": encoded("red"),
            "wrist_image": encoded("green"),
            "task_index": 30,
            "episode_index": 1261,
            "frame_index": 6,
            "actions": "recorded actions must not be read",
        },
    ]
    path = tmp_path / "data/chunk-001/episode_001261.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return tmp_path


def test_load_observation_selects_frame_identifier_and_task_mapping(lerobot_dataset):
    external, wrist, metadata = load_lerobot_observation(lerobot_dataset, 1261, 6)

    assert external.getpixel((0, 0)) == (255, 0, 0)
    assert wrist.getpixel((0, 0)) == (0, 128, 0)
    assert metadata["frame_index"] == 6
    assert metadata["episode_index"] == 1261
    assert metadata["task_index"] == 30
    assert metadata["instruction"] == "put the bowl on the plate"
    assert metadata["source_fps"] == 10


def test_load_observation_rejects_missing_frame(lerobot_dataset):
    with pytest.raises(ValueError, match="Expected one row.*found 0"):
        load_lerobot_observation(lerobot_dataset, 1261, 0)


def test_load_observation_decodes_dataset_relative_image_paths(lerobot_dataset):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    Image.new("RGB", (8, 8), "yellow").save(lerobot_dataset / "external.png")
    Image.new("RGB", (8, 8), "blue").save(lerobot_dataset / "wrist.png")
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "image": {"bytes": None, "path": "external.png"},
                    "wrist_image": {"bytes": None, "path": "wrist.png"},
                    "task_index": 30,
                    "episode_index": 1261,
                    "frame_index": 0,
                }
            ]
        ),
        lerobot_dataset / "data/chunk-001/episode_001261.parquet",
    )

    external, wrist, _ = load_lerobot_observation(lerobot_dataset, 1261, 0)

    assert external.getpixel((0, 0)) == (255, 255, 0)
    assert wrist.getpixel((0, 0)) == (0, 0, 255)


def test_canvas_preserves_camera_order_and_reports_resize_padding():
    external = Image.new("RGB", (256, 256), "red")
    wrist = Image.new("RGB", (256, 256), "blue")

    canvas, metadata = prepare_canvas(external, wrist, 320, 192)

    assert canvas.size == (320, 192)
    assert canvas.getpixel((40, 80)) == (255, 0, 0)
    assert canvas.getpixel((240, 80)) == (0, 0, 255)
    assert metadata["resized_size"] == [320, 160]
    assert metadata["padding_bottom"] == 32
    assert metadata["padding_right"] == 0
    assert metadata["padding_mode"] == "reflect"
    assert metadata["camera_order"] == ["external", "wrist"]


def test_canvas_rejects_missing_matching_camera_view():
    with pytest.raises(ValueError, match="Expected matching camera sizes"):
        prepare_canvas(Image.new("RGB", (8, 8)), Image.new("RGB", (4, 4)), 32, 16)


def test_png_data_uri_preserves_canvas_pixels():
    original = Image.fromarray(np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3))
    prefix, encoded = image_to_data_uri(original).split(",", 1)
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as decoded:
        np.testing.assert_array_equal(np.asarray(decoded), np.asarray(original))
    assert prefix == "data:image/png;base64"


def test_prompt_names_both_camera_positions():
    prompt = augment_prompt("put the bowl on the plate")
    assert prompt.startswith("put the bowl on the plate.")
    assert "left half shows the third-person view" in prompt
    assert "right half shows the wrist-mounted camera" in prompt


def test_native_actions_preserve_normalized_values_without_clipping():
    values = np.linspace(-2.0, 2.0, 20, dtype=np.float32).reshape(2, 10)
    np.testing.assert_array_equal(
        validate_native_actions(values.tolist(), 2, 10), values
    )


@pytest.mark.parametrize("shape", [(1, 2, 10), (2, 64), (2, 7), (0, 10)])
def test_native_actions_reject_unexpected_layout(shape):
    with pytest.raises(ValueError, match="Expected action shape"):
        validate_native_actions(np.zeros(shape), 2, 10)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_native_actions_reject_nonfinite_values(invalid):
    values = np.zeros((2, 10), dtype=np.float32)
    values[1, 3] = invalid
    with pytest.raises(ValueError, match="Expected finite model actions"):
        validate_native_actions(values, 2, 10)


def test_video_round_trip_keeps_frame_count_and_dimensions(tmp_path):
    pytest.importorskip("imageio.v2")
    pytest.importorskip("imageio_ffmpeg")
    frames = np.zeros((3, 18, 30, 3), dtype=np.uint8)
    frames[1] = 100
    frames[2] = 200
    path = tmp_path / "rollout.mp4"

    save_video(path, frames, 20)
    decoded = read_video_frames(path)

    assert decoded.shape == frames.shape
    assert decoded.dtype == np.uint8
    assert decoded[2].mean() > decoded[1].mean() > decoded[0].mean()
