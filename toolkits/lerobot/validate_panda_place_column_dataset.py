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

"""Validate a Panda place-column LeRobot expert dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch


REQUIRED_KEYS = {
    "observation.state",
    "observation.images.base",
    "actions",
    "state_id",
    "is_success",
    "done",
}


def validate_dataset(
    dataset_path: Path,
    *,
    minimum_per_state: int,
    expected_state_count: int,
    require_success: bool,
) -> dict[str, object]:
    """Validate schema and return a compact dataset summary."""
    from lerobot.common.datasets.lerobot_dataset import (
        LeRobotDataset,
        LeRobotDatasetMetadata,
    )

    dataset_path = dataset_path.resolve()
    repo_id = dataset_path.name
    metadata = LeRobotDatasetMetadata(repo_id, root=dataset_path)
    missing = REQUIRED_KEYS - set(metadata.features)
    if missing:
        raise ValueError(f"Dataset is missing required features: {sorted(missing)}")

    dataset = LeRobotDataset(repo_id, root=dataset_path, download_videos=False)
    state_counts: Counter[int] = Counter()
    episode_count = 0
    for start, end in zip(
        dataset.episode_data_index["from"].tolist(),
        dataset.episode_data_index["to"].tolist(),
        strict=True,
    ):
        if end <= start:
            raise ValueError(f"Episode {episode_count} is empty")
        first = dataset[int(start)]
        last = dataset[int(end) - 1]
        state_id = int(torch.as_tensor(first["state_id"]).reshape(-1)[0])
        success = bool(torch.as_tensor(last["is_success"]).reshape(-1)[0])
        done = bool(torch.as_tensor(last["done"]).reshape(-1)[0])
        if require_success and not success:
            raise ValueError(f"Episode {episode_count} is not successful")
        if not done:
            raise ValueError(f"Episode {episode_count} does not end with done=True")
        state_counts[state_id] += int(success)
        episode_count += 1

    expected_ids = set(range(expected_state_count))
    if set(state_counts) - expected_ids:
        unexpected = sorted(set(state_counts) - expected_ids)
        raise ValueError(f"Unexpected state IDs: {unexpected}")
    undercovered = {
        state_id: state_counts[state_id]
        for state_id in expected_ids
        if state_counts[state_id] < minimum_per_state
    }
    if undercovered:
        raise ValueError(
            f"States below minimum success count {minimum_per_state}: {undercovered}"
        )
    return {
        "episodes": episode_count,
        "frames": len(dataset),
        "fps": metadata.fps,
        "state_dim": metadata.features["observation.state"]["shape"],
        "action_dim": metadata.features["actions"]["shape"],
        "successes_by_state": dict(sorted(state_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--minimum-per-state", type=int, default=1)
    parser.add_argument("--expected-state-count", type=int, default=5)
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()
    summary = validate_dataset(
        args.dataset_path,
        minimum_per_state=args.minimum_per_state,
        expected_state_count=args.expected_state_count,
        require_success=not args.allow_failures,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
