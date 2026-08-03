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

"""LeRobot behavior-cloning dataset for ManiSkill MLP and CNN policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from rlinf.data.lerobot_paths import resolve_lerobot_dataset_root


def _uint8_hwc(image: Any) -> torch.Tensor:
    image = torch.as_tensor(image)
    if image.ndim == 3 and image.shape[0] in {1, 3, 4}:
        image = image.permute(1, 2, 0)
    if image.is_floating_point():
        image = image * 255.0 if image.max() <= 1.0 else image
    return image.clamp(0, 255).to(torch.uint8)


class ManiSkillLeRobotBCDataset(Dataset):
    """Adapt canonical expert frames to MLP or CNN SFT batches."""

    def __init__(
        self,
        data_path: str,
        *,
        policy_type: Literal["mlp_policy", "cnn_policy"],
        action_horizon: int = 1,
        video_backend: str = "pyav",
    ) -> None:
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        from lerobot.common.datasets.lerobot_dataset import (
            LeRobotDataset,
            LeRobotDatasetMetadata,
        )

        root = resolve_lerobot_dataset_root(data_path)
        repo_id = Path(root).name
        self.meta = LeRobotDatasetMetadata(repo_id, root=root)
        delta_timestamps = {
            "actions": [step / self.meta.fps for step in range(action_horizon)]
        }
        self.dataset = LeRobotDataset(
            repo_id,
            root=root,
            delta_timestamps=delta_timestamps,
            video_backend=video_backend,
        )
        self.policy_type = policy_type

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.dataset[index]
        states = torch.as_tensor(sample["observation.state"]).float()
        actions = torch.as_tensor(sample["actions"]).float().reshape(-1)
        batch = {"states": states, "action": actions}
        if self.policy_type == "cnn_policy":
            batch["main_images"] = _uint8_hwc(
                sample["observation.images.base"]
            )
            if "observation.images.wrist" in sample:
                batch["extra_view_images"] = _uint8_hwc(
                    sample["observation.images.wrist"]
                ).unsqueeze(0)
        return batch
