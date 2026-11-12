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

"""Run a dataset-initialized Cosmos3 policy/world-model loop through one server."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from toolkits.cosmos3.closed_loop_client import (
    Cosmos3Client,
    dynamics_request,
    policy_request,
)
from toolkits.cosmos3.closed_loop_io import (
    augment_prompt,
    image_to_data_uri,
    load_lerobot_observation,
    prepare_canvas,
    read_video_frames,
    save_video,
    validate_native_actions,
)

DEFAULT_CONFIG = Path(__file__).parent / "config/closed_loop_nano.yaml"


def write_json(path: Path, value: Any) -> None:
    """Write strict JSON so failed runs cannot hide nonfinite values in artifacts."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_config(config: dict) -> None:
    """Validate the supported single-scene LIBERO inference contract."""
    generation = config["generation"]
    if generation["domain_name"] != "libero" or generation["raw_action_dim"] != 10:
        raise ValueError(
            "This initial loop expects domain_name=libero and raw_action_dim=10"
        )
    allowed = {
        "domain_name",
        "raw_action_dim",
        "num_frames",
        "width",
        "height",
        "fps",
        "action_fps",
        "action_view_point",
        "num_inference_steps",
        "seed",
        "guidance_scale",
    }
    if set(generation) - allowed:
        raise ValueError(
            f"Unsupported generation parameters: {sorted(set(generation) - allowed)}"
        )
    for name in ("width", "height", "fps", "num_inference_steps"):
        if not isinstance(generation[name], int) or generation[name] <= 0:
            raise ValueError(
                f"Expected a positive integer for {name}, got {generation[name]!r}"
            )
    if generation["width"] % 16 or generation["height"] % 16:
        raise ValueError(
            "Expected width and height divisible by 16 for Cosmos3 latent patches"
        )
    frames = generation["num_frames"]
    if not isinstance(frames, int) or frames < 5 or (frames - 1) % 4:
        raise ValueError(f"Expected num_frames=4k+1 with k>=1, got {frames!r}")
    if generation["action_fps"] != generation["fps"]:
        raise ValueError(
            "Expected equal action_fps and fps for one action per new video frame"
        )
    if not isinstance(generation["seed"], int) or generation["seed"] < 0:
        raise ValueError("Expected a nonnegative integer generation seed")
    if not isinstance(config["cycles"], int) or config["cycles"] <= 0:
        raise ValueError("Expected a positive integer cycle count")
    for name in ("startup_timeout_s", "request_timeout_s", "poll_interval_s"):
        value = config["server"][name]
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Expected positive finite {name}, got {value!r}")


def runtime_info(python: str) -> dict:
    """Read installed versions and the SGLang VCS pin without importing model code."""
    script = """
import importlib.metadata as m, json, sys
result = {"python": sys.version, "executable": sys.executable, "packages": {}}
for name in ("sglang", "torch", "diffusers", "transformers", "numpy", "pyarrow"):
    try: result["packages"][name] = m.version(name)
    except m.PackageNotFoundError: result["packages"][name] = None
try:
    direct = m.distribution("sglang").read_text("direct_url.json")
    result["sglang_source"] = json.loads(direct) if direct else None
except m.PackageNotFoundError: result["sglang_source"] = None
print(json.dumps(result))
"""
    result = subprocess.run(
        [python, "-c", script], capture_output=True, text=True, check=True
    )
    return json.loads(result.stdout)


def checkpoint_info(path: Path) -> dict:
    """Record lightweight checkpoint identity and require existing action heads."""
    index_path = path / "model_index.json"
    transformer_path = path / "transformer/config.json"
    model_index = json.loads(index_path.read_text())
    transformer = json.loads(transformer_path.read_text())
    if not transformer.get("action_gen"):
        raise ValueError(f"Expected action_gen=true in {transformer_path}")
    metadata = {
        "path": str(path.resolve()),
        "model_index": model_index,
        "transformer_config": transformer,
        "metadata_sha256": {},
    }
    for name in (
        "model_index.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
    ):
        file = path / name
        if file.is_file():
            metadata["metadata_sha256"][name] = hashlib.sha256(
                file.read_bytes()
            ).hexdigest()
    revisions = set()
    for file in (path / ".cache/huggingface/download").glob("**/*.metadata"):
        lines = file.read_text().splitlines()
        if lines:
            revisions.add(lines[0])
    metadata["download_revisions"] = sorted(revisions)
    return metadata


def code_info() -> dict:
    """Record the exact toolkit sources even when running from an uncommitted checkout."""
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return {
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "files": {
            file.name: hashlib.sha256(file.read_bytes()).hexdigest()
            for file in Path(__file__).parent.glob("*.py")
        },
    }


@contextmanager
def model_server(
    config: dict, output_dir: Path, existing_url: str | None = None
) -> Iterator[tuple[Cosmos3Client, dict]]:
    """Own one local model process group, or connect to a caller-owned server."""
    import requests

    server = config["server"]
    process = None
    log = None
    identity = {"ownership": "external" if existing_url else "launched"}
    url = existing_url or f"http://{server['host']}:{server['port']}"
    client = Cosmos3Client(url, server["request_timeout_s"], server["poll_interval_s"])
    try:
        if not existing_url:
            python = server.get("python") or sys.executable
            installed = runtime_info(python)
            identity["runtime"] = installed
            write_json(output_dir / "runtime.json", installed)
            actual = (
                (installed.get("sglang_source") or {})
                .get("vcs_info", {})
                .get("commit_id")
            )
            expected = config["runtime"]["sglang_revision"]
            if actual != expected:
                raise RuntimeError(
                    f"Expected SGLang commit {expected}, found {actual!r}. "
                    "Install toolkits/cosmos3/requirements-closed-loop.txt in the server environment."
                )
            # Refuse a busy port instead of accidentally attaching to another model.
            with socket.socket() as probe:
                probe.bind((server["host"], server["port"]))
            command = [
                python,
                "-m",
                "sglang.multimodal_gen.runtime.launch_server",
                "--model-path",
                str(Path(config["checkpoint_path"]).resolve()),
                "--pipeline-class-name",
                server["pipeline_class_name"],
                "--host",
                server["host"],
                "--port",
                str(server["port"]),
                "--num-gpus",
                "1",
                "--attention-backend",
                server["attention_backend"],
                "--output-path",
                str(output_dir / "server_outputs"),
                *server.get("extra_args", []),
            ]
            identity["command"] = command
            log = (output_dir / "logs/server.log").open("w")
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            identity["pid"] = process.pid
            deadline = time.monotonic() + server["startup_timeout_s"]
            while True:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"SGLang exited with code {process.returncode}; see {output_dir / 'logs/server.log'}"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"SGLang startup exceeded {server['startup_timeout_s']}s"
                    )
                try:
                    with client.request("GET", "/health", timeout=min(5, remaining)):
                        break
                except (requests.RequestException, RuntimeError):
                    time.sleep(min(1, remaining))
        info = client.server_info()
        expected_path = str(Path(config["checkpoint_path"]).resolve())
        if info.get("model_path") != expected_path:
            raise ValueError(
                f"Expected server model_path={expected_path}, got {info.get('model_path')!r}"
            )
        identity.update({"url": url, "server_info": info})
        write_json(output_dir / "server.json", identity)
        yield client, identity
    finally:
        client.close()
        if process is not None:
            # The new process group belongs to this run, including server workers.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if log is not None:
            log.close()


def run_cycles(
    client: Cosmos3Client,
    initial_image: Any,
    instruction: str,
    generation: dict,
    cycles: int,
    output_dir: Path,
) -> list[dict]:
    """Run real HTTP policy/FD calls and feed only generated observations back."""
    import numpy as np
    from PIL import Image

    current = initial_image
    prompt = augment_prompt(instruction)
    metrics = []
    rollout = [np.asarray(current)]
    for cycle in range(cycles):
        cycle_dir = output_dir / f"cycle_{cycle:03d}"
        cycle_dir.mkdir()
        current.save(cycle_dir / "input.png")
        image_uri = image_to_data_uri(current)
        policy_parameters = {**generation, "seed": generation["seed"] + 2 * cycle}
        request = policy_request(prompt, image_uri, policy_parameters)
        write_json(cycle_dir / "policy_request.json", request)
        started = time.monotonic()
        actions, response = client.generate_actions(request)
        policy_seconds = time.monotonic() - started
        write_json(cycle_dir / "policy_response.json", response)
        np.save(cycle_dir / "actions_native.npy", actions)
        fd_parameters = {**generation, "seed": generation["seed"] + 2 * cycle + 1}
        fd_request = dynamics_request(
            prompt, image_uri, fd_parameters, actions, cycle_dir / "server_video"
        )
        conditioning = validate_native_actions(
            fd_request["action"],
            generation["num_frames"] - 1,
            generation["raw_action_dim"],
        )
        if not np.array_equal(actions, conditioning):
            raise ValueError(
                "Policy actions changed while building forward-dynamics conditioning"
            )
        np.save(cycle_dir / "actions_conditioning.npy", conditioning)
        write_json(cycle_dir / "dynamics_request.json", fd_request)
        started = time.monotonic()
        video_result = client.generate_video(fd_request, cycle_dir / "prediction.mp4")
        dynamics_seconds = time.monotonic() - started
        write_json(cycle_dir / "dynamics_response.json", video_result)
        frames = read_video_frames(cycle_dir / "prediction.mp4")
        expected = (
            generation["num_frames"],
            generation["height"],
            generation["width"],
            3,
        )
        if frames.shape != expected:
            raise ValueError(
                f"Expected generated video shape {expected}, got {frames.shape}"
            )
        current = Image.fromarray(frames[-1])
        current.save(cycle_dir / "next.png")
        # FD returns the conditioning frame first; each chunk contributes H new frames.
        rollout.extend(frames[1:])
        row = {
            "cycle": cycle,
            "policy_seconds": policy_seconds,
            "dynamics_seconds": dynamics_seconds,
            "actions_shape": list(actions.shape),
            "video_shape": list(frames.shape),
            "actions_equal": True,
            "input_sha256": hashlib.sha256(
                (cycle_dir / "input.png").read_bytes()
            ).hexdigest(),
            "next_sha256": hashlib.sha256(
                (cycle_dir / "next.png").read_bytes()
            ).hexdigest(),
            "policy_seed": policy_parameters["seed"],
            "dynamics_seed": fd_parameters["seed"],
        }
        metrics.append(row)
        write_json(output_dir / "metrics.json", metrics)
    save_video(output_dir / "rollout.mp4", np.stack(rollout), generation["fps"])
    return metrics


def run(
    config: dict, output_dir: Path, server_url: str | None, task_id: str | None
) -> None:
    """Create a new artifact directory and retain a failure manifest on errors."""
    import yaml

    validate_config(config)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "logs").mkdir()
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    manifest = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "task_id": task_id,
        "code": code_info(),
        "cycles_requested": config["cycles"],
        "model_fps": config["generation"]["fps"],
        "normalization": "model_native",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    write_json(output_dir / "manifest.json", manifest)
    try:
        manifest["checkpoint"] = checkpoint_info(Path(config["checkpoint_path"]))
        dataset = config["dataset"]
        external, wrist, metadata = load_lerobot_observation(
            Path(dataset["root"]), dataset["episode_index"], dataset["frame_index"]
        )
        initial, layout = prepare_canvas(
            external,
            wrist,
            config["generation"]["width"],
            config["generation"]["height"],
        )
        external.save(output_dir / "initial_external.png")
        wrist.save(output_dir / "initial_wrist.png")
        initial.save(output_dir / "initial_canvas.png")
        manifest.update({"dataset": metadata, "canvas": layout})
        write_json(output_dir / "manifest.json", manifest)
        with model_server(config, output_dir, server_url) as (client, identity):
            manifest["server"] = identity
            write_json(output_dir / "manifest.json", manifest)
            metrics = run_cycles(
                client,
                initial,
                metadata["instruction"],
                config["generation"],
                config["cycles"],
                output_dir,
            )
            final_info = client.server_info()
            if final_info != identity["server_info"]:
                raise RuntimeError("Server identity changed during the closed loop")
            manifest["cycles_completed"] = len(metrics)
        manifest["status"] = "completed"
    except BaseException as error:
        manifest.update(
            {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        )
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output_dir / "manifest.json", manifest)


def main() -> None:
    """Parse the standalone toolkit CLI and run one reproducible experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New unique run directory on shared storage",
    )
    parser.add_argument("--cycles", type=int)
    parser.add_argument(
        "--server-url", help="Use an existing server; this runner will not stop it"
    )
    parser.add_argument(
        "--task-id", help="Compute platform job identifier, if submitted as a task"
    )
    args = parser.parse_args()
    import yaml

    def terminate(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    config = yaml.safe_load(args.config.read_text())
    if args.cycles is not None:
        config = {**config, "cycles": args.cycles}
    run(config, args.output_dir, args.server_url, args.task_id)


if __name__ == "__main__":
    main()
