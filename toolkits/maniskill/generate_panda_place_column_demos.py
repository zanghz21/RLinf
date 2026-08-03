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

"""Generate ManiSkill demonstrations for ``PandaPlaceColumnInBox-v1``."""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import gymnasium as gym
from mani_skill.utils.wrappers.record import RecordEpisode
from tqdm import tqdm

from rlinf.envs.maniskill.motionplanning.panda_place_column_in_box import (
    MotionPlanningFailure,
    solve,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse demonstration-generation options."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate CPU motion-planning demonstrations for "
            "PandaPlaceColumnInBox-v1."
        )
    )
    parser.add_argument("--num-traj", type=int, default=10)
    parser.add_argument(
        "--min-successes-per-column-xy",
        type=int,
        default=1,
        help=(
            "Require at least this many successful trajectories from every "
            "column_xy_list anchor."
        ),
    )
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--record-dir", type=Path, default=Path("demos"))
    parser.add_argument("--trajectory-name", default="trajectory")
    parser.add_argument("--obs-mode", default="none")
    parser.add_argument("--render-mode", default="rgb_array")
    parser.add_argument("--shader", default="default")
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument(
        "--save-failures",
        action="store_true",
        help="Save failed planning attempts as well as successful trajectories.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help=(
            "Stop after this many attempts; defaults to 10 times the larger "
            "of the total and per-anchor success requirements."
        ),
    )
    return parser.parse_args()


def _scalar_bool(value) -> bool:
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


def _result_success(result) -> bool:
    if result is None or isinstance(result, int):
        return False
    info = result[-1]
    return _scalar_bool(info.get("success", False))


def _collection_complete(
    successes: int,
    successes_by_column_xy: list[int],
    num_traj: int,
    minimum_per_column_xy: int,
) -> bool:
    """Return whether total and per-anchor success requirements are met."""
    return successes >= num_traj and all(
        count >= minimum_per_column_xy for count in successes_by_column_xy
    )


def _coverage_progress(
    successes: int,
    successes_by_column_xy: list[int],
    num_traj: int,
    minimum_per_column_xy: int,
) -> tuple[int, int]:
    """Return completed and required slots for combined collection targets."""
    coverage = sum(
        min(count, minimum_per_column_xy)
        for count in successes_by_column_xy
    )
    coverage_target = minimum_per_column_xy * len(successes_by_column_xy)
    extra_target = max(0, num_traj - coverage_target)
    extra_successes = successes - coverage
    return (
        coverage + min(extra_successes, extra_target),
        coverage_target + extra_target,
    )


def _next_column_xy_index(
    successes_by_column_xy: list[int],
    minimum_per_column_xy: int,
    attempt: int,
) -> int:
    """Cycle through anchors that have not yet reached their quota."""
    candidates = [
        index
        for index, count in enumerate(successes_by_column_xy)
        if count < minimum_per_column_xy
    ]
    if not candidates:
        candidates = list(range(len(successes_by_column_xy)))
    return candidates[attempt % len(candidates)]


def _video_suffix(column_xy_index: int) -> str:
    """Return the video filename annotation for a column anchor state."""
    return f"state_id_{column_xy_index}"


def main(args: argparse.Namespace) -> None:
    """Generate and record expert trajectories."""
    if args.num_traj <= 0:
        raise ValueError("--num-traj must be positive")
    if args.min_successes_per_column_xy <= 0:
        raise ValueError("--min-successes-per-column-xy must be positive")
    if args.max_episode_steps <= 0:
        raise ValueError("--max-episode-steps must be positive")

    env_id = "PandaPlaceColumnInBox-v1"
    env = gym.make(
        env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        render_mode=args.render_mode,
        sensor_configs={"shader_pack": args.shader},
        human_render_camera_configs={"shader_pack": args.shader},
        viewer_camera_configs={"shader_pack": args.shader},
        sim_backend="cpu",
        max_episode_steps=args.max_episode_steps,
    )
    column_xy_list = env.unwrapped.column_xy_list
    required_successes = max(
        args.num_traj,
        args.min_successes_per_column_xy * len(column_xy_list),
    )
    max_attempts = args.max_attempts or required_successes * 10
    if max_attempts < required_successes:
        env.close()
        raise ValueError(
            "--max-attempts must be at least the larger of --num-traj and "
            "len(column_xy_list) * --min-successes-per-column-xy"
        )
    output_dir = args.record_dir / env_id / "motionplanning"
    env = RecordEpisode(
        env,
        output_dir=str(output_dir),
        trajectory_name=args.trajectory_name,
        save_video=args.save_video,
        source_type="motionplanning",
        source_desc="RLinf lateral-grasp motion-planning expert",
        video_fps=30,
        record_reward=False,
        save_on_reset=False,
    )

    failures: Counter[str] = Counter()
    successes = 0
    successes_by_column_xy = [0] * len(column_xy_list)
    attempts = 0
    progress = tqdm(total=required_successes, desc="successful trajectories")
    try:
        while (
            not _collection_complete(
                successes,
                successes_by_column_xy,
                args.num_traj,
                args.min_successes_per_column_xy,
            )
            and attempts < max_attempts
        ):
            seed = args.start_seed + attempts
            column_xy_index = _next_column_xy_index(
                successes_by_column_xy,
                args.min_successes_per_column_xy,
                attempts,
            )
            attempts += 1
            failure_stage = None
            try:
                result = solve(
                    env,
                    seed=seed,
                    column_xy_index=column_xy_index,
                    debug=False,
                    vis=args.vis,
                )
                success = _result_success(result)
                if not success:
                    failure_stage = "success_check"
            except MotionPlanningFailure as exc:
                success = False
                failure_stage = exc.stage
                logger.warning("Seed %d failed: %s", seed, exc)
            except Exception:
                success = False
                failure_stage = "unexpected_error"
                logger.exception("Expert failed for seed %d", seed)

            if success:
                env.flush_trajectory()
                if args.save_video:
                    env.flush_video(suffix=_video_suffix(column_xy_index))
                successes += 1
                successes_by_column_xy[column_xy_index] += 1
            else:
                failures[failure_stage or "unknown"] += 1
                env.flush_trajectory(save=args.save_failures)
                if args.save_video:
                    env.flush_video(
                        suffix=_video_suffix(column_xy_index),
                        save=args.save_failures,
                    )

            completed_slots, _ = _coverage_progress(
                successes,
                successes_by_column_xy,
                args.num_traj,
                args.min_successes_per_column_xy,
            )
            progress.n = completed_slots
            progress.set_postfix(
                attempts=attempts,
                success_rate=f"{successes / attempts:.3f}",
                per_xy=successes_by_column_xy,
            )
            progress.refresh()
    finally:
        progress.close()
        env.close()

    logger.info(
        "Generated %d successful trajectories from %d attempts in %s",
        successes,
        attempts,
        output_dir,
    )
    logger.info("Successes by column_xy_list index: %s", successes_by_column_xy)
    if failures:
        logger.info("Failure stages: %s", dict(failures))
    if not _collection_complete(
        successes,
        successes_by_column_xy,
        args.num_traj,
        args.min_successes_per_column_xy,
    ):
        raise RuntimeError(
            f"Generated {successes}/{args.num_traj} total successes with "
            f"per-column-xy counts {successes_by_column_xy}; required at "
            f"least {args.min_successes_per_column_xy} each before reaching "
            f"--max-attempts={max_attempts}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(parse_args())
