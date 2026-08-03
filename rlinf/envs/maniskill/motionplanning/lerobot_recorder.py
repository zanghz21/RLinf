# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LeRobot recorder for single-environment ManiSkill expert rollouts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from mani_skill.utils import common

from rlinf.data.lerobot_writer import LeRobotDatasetWriter


def _single_numpy(value: Any) -> np.ndarray:
    """Convert a single-environment tensor or array to NumPy."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim > 0 and value.shape[0] == 1:
        value = value[0]
    return value


def extract_lerobot_observation(observation: Any) -> dict[str, np.ndarray]:
    """Extract canonical state and camera fields from a ManiSkill observation."""
    if not isinstance(observation, dict):
        return {"observation.state": _single_numpy(observation).astype(np.float32)}

    sensor_data = observation.get("sensor_data", {})
    state_fields = {
        key: value
        for key, value in observation.items()
        if key not in {"sensor_data", "sensor_param"}
    }
    state = common.flatten_state_dict(state_fields, use_torch=True)
    extracted = {
        "observation.state": _single_numpy(state).astype(np.float32),
    }

    camera_names = list(sensor_data)
    if "base_camera" in sensor_data:
        camera_names.remove("base_camera")
        camera_names.insert(0, "base_camera")
    if camera_names:
        extracted["observation.images.base"] = _single_numpy(
            sensor_data[camera_names[0]]["rgb"]
        ).astype(np.uint8)
    if len(camera_names) > 1:
        extracted["observation.images.wrist"] = _single_numpy(
            sensor_data[camera_names[1]]["rgb"]
        ).astype(np.uint8)
    return extracted


class ManiSkillLeRobotExpertRecorder(gym.Wrapper):
    """Record manually flushed motion-planning episodes in LeRobot format."""

    def __init__(
        self,
        env: gym.Env,
        output_dir: Path,
        *,
        task: str,
        fps: int,
    ) -> None:
        super().__init__(env)
        self.output_dir = Path(output_dir)
        self.task = task
        self.fps = fps
        self.writer = LeRobotDatasetWriter()
        self._writer_created = False
        self._last_observation: Any = None
        self._frames: list[dict[str, Any]] = []

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._frames = []
        self._last_observation = observation
        return observation, info

    def step(self, action):
        frame = extract_lerobot_observation(self._last_observation)
        frame["actions"] = _single_numpy(action).astype(np.float32).reshape(-1)
        frame["task"] = self.task
        self._frames.append(frame)
        result = self.env.step(action)
        self._last_observation = result[0]
        return result

    def _create_writer(self, frame: dict[str, Any]) -> None:
        features: dict[str, dict[str, Any]] = {
            "observation.state": {
                "dtype": "float32",
                "shape": tuple(frame["observation.state"].shape),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": tuple(frame["actions"].shape),
                "names": ["actions"],
            },
            "state_id": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["state_id"],
            },
            "is_success": {
                "dtype": "bool",
                "shape": (1,),
                "names": ["is_success"],
            },
            "done": {
                "dtype": "bool",
                "shape": (1,),
                "names": ["done"],
            },
        }
        for key in ("observation.images.base", "observation.images.wrist"):
            if key in frame:
                features[key] = {
                    "dtype": "image",
                    "shape": tuple(frame[key].shape),
                    "names": ["height", "width", "channel"],
                }
        self.writer.create(
            repo_id=str(self.output_dir),
            robot_type="franka_panda",
            fps=self.fps,
            features=features,
        )
        self._writer_created = True

    def flush_episode(self, *, success: bool, state_id: int, save: bool) -> None:
        """Save or discard the currently buffered expert episode."""
        if not self._frames:
            return
        if save:
            for index, frame in enumerate(self._frames):
                frame["state_id"] = np.array([state_id], dtype=np.int64)
                frame["is_success"] = np.array([success], dtype=bool)
                frame["done"] = np.array(
                    [index == len(self._frames) - 1], dtype=bool
                )
            if not self._writer_created:
                self._create_writer(self._frames[0])
            self.writer.add_episode(self._frames)
        self._frames = []
        self._last_observation = None

    def close(self) -> None:
        if self._writer_created:
            self.writer.finalize()
            self._writer_created = False
        super().close()
