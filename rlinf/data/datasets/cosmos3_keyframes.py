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

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from rlinf.models.embodiment.cosmos3.closed_loop_io import (
    load_lerobot_observation,
    prepare_canvas,
)


class Cosmos3KeyframeDataset(Dataset):
    """Expose curated LeRobot observations as WorldModelEnv reset samples."""

    def __init__(self, cfg, image_size):
        self.root = Path(cfg.root)
        self.height, self.width = image_size
        self.keyframes = [(row.episode_index, row.frame_index) for row in cfg.keyframes]
        if not self.keyframes or any(
            type(i) is not int or i < 0 for pair in self.keyframes for i in pair
        ):
            raise ValueError(
                "Expected nonnegative episode and frame indices in keyframes"
            )

    def __len__(self):
        return len(self.keyframes)

    def __getitem__(self, index):
        episode, frame = self.keyframes[index]
        external, wrist, metadata = load_lerobot_observation(self.root, episode, frame)
        canvas, _ = prepare_canvas(external, wrist, self.width, self.height)
        image = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float().div_(255)
        return {"start_items": [{"image": image}], "task": metadata["instruction"]}
