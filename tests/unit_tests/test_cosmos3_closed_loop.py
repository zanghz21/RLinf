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

"""Exercise Cosmos3 worker boundaries with raw frames and no model weights."""

import base64
import io
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from rlinf.models.embodiment.cosmos3.closed_loop_client import (
    Cosmos3Client,
    dynamics_request,
)
from rlinf.models.embodiment.cosmos3.closed_loop_io import (
    read_video_frames,
)
from rlinf.models.embodiment.cosmos3.sglang_adapter import Cosmos3SGLangAdapter
from rlinf.utils.http_client import InferenceHTTPClient
from rlinf.workers.rollout.sglang.sglang_embodied_worker import SGLangEmbodiedWorker

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")
pytest.importorskip("requests")


class FakeSGLang:
    """Serve policy and raw rollout responses at the real HTTP boundary."""

    def __init__(self, frames=None):
        self.frames = frames or []
        self.generated_output = None
        self.response_count = 1
        self.error_status = None
        self.delay_s = 0
        self.rollout_requests = []
        self.model_path = "/unused/checkpoint"
        self.requests = []
        self.policy_values = None
        self.policy_values_by_image = None
        self.reverse_policy_response = False
        self.policy_outputs = []
        self.policy_barrier = None

    def handler(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def respond(self, value, status=200):
                if status < 400 and self.headers.get("Accept") == "application/msgpack":
                    from sglang.multimodal_gen.runtime.entrypoints.action.protocol import (
                        pack_msgpack,
                    )

                    payload = pack_msgpack(value)
                    self.send_response(status)
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("Content-Type", "application/msgpack")
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                payload = (
                    value if isinstance(value, bytes) else json.dumps(value).encode()
                )
                self.send_response(status)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header(
                    "Content-Type",
                    "video/mp4" if isinstance(value, bytes) else "application/json",
                )
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                if self.headers.get("Content-Type") == "application/msgpack":
                    from sglang.multimodal_gen.runtime.entrypoints.action.protocol import (
                        unpack_msgpack,
                    )

                    payload = unpack_msgpack(raw)
                else:
                    payload = json.loads(raw)
                owner.requests.append(("POST", self.path, payload))
                if self.path == "/v1/actions/generations":
                    barrier, owner.policy_barrier = owner.policy_barrier, None
                    if barrier is not None:
                        barrier.wait(timeout=20)
                    parameters = payload["parameters"]
                    horizon, width = (
                        parameters["num_frames"] - 1,
                        parameters["raw_action_dim"],
                    )

                    data = []
                    for index in range(len(payload["input"]["prompt"])):
                        values = owner.policy_values
                        if owner.policy_values_by_image is not None:
                            reference = payload["input"]["input_reference"][index]
                            marker = int(decode_image(reference)[0, 0, 0])
                            values = owner.policy_values_by_image[marker]
                        if values is None:
                            values = (
                                np.linspace(
                                    -2, 2, horizon * width, dtype=np.float32
                                ).reshape(horizon, width)
                                + len(owner.policy_outputs)
                            ).tolist()
                        owner.policy_outputs.append(values)
                        data.append(
                            {
                                "input_index": index,
                                "action": {
                                    "values": values,
                                    "shape": [horizon, width],
                                    "raw_action_dim": width,
                                    "domain_id": 5,
                                    "action_mode": "policy",
                                },
                            }
                        )
                    if owner.reverse_policy_response:
                        data.reverse()
                    self.respond({"data": data})
                elif self.path == "/rollout/generate":
                    import msgspec
                    import torch
                    from safetensors.torch import save

                    owner.rollout_requests.append(payload)
                    assert self.headers.get("Content-Type") == "application/json"
                    assert self.headers.get("Accept") == "application/msgpack"
                    if owner.delay_s:
                        time.sleep(owner.delay_s)
                    if owner.error_status:
                        self.respond(
                            {"detail": "test generation failure"}, owner.error_status
                        )
                        return
                    output = owner.generated_output
                    if output is None:
                        frames = owner.frames[
                            (len(owner.rollout_requests) - 1) % len(owner.frames)
                        ]
                        output = (
                            torch.from_numpy(frames.copy()).permute(3, 0, 1, 2).float()
                            / 255
                        )
                    output = output.contiguous()
                    response = {
                        "request_id": f"test-request-{len(owner.rollout_requests)}",
                        "prompt": payload["prompt"],
                        "seed": payload["seed"],
                        "generated_output": {
                            "__tensor__": True,
                            "data": save({"t": output}),
                            "shape": list(output.shape),
                            "dtype": str(output.dtype),
                        },
                        "inference_time_s": 0.01,
                    }
                    body = msgspec.msgpack.encode([response] * owner.response_count)
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Content-Type", "application/msgpack")
                    self.end_headers()
                    try:
                        self.wfile.write(body)
                    except BrokenPipeError:
                        pass
                else:
                    self.respond({"error": "unknown route"}, 404)

            def do_GET(self):
                owner.requests.append(("GET", self.path, None))
                if self.path == "/server_info":
                    from sglang.version import __version__

                    self.respond(
                        {
                            "model_path": owner.model_path,
                            "pipeline": "RLinfCosmos3Pipeline",
                            "version": __version__,
                        }
                    )
                    return
                if self.path == "/v1/actions/metadata":
                    self.respond(
                        {
                            "policy_family": "cosmos3",
                            "capabilities": {
                                "max_batch_size": 3,
                                "batch_inputs": True,
                                "batched_action_modes": ["policy"],
                            },
                        }
                    )
                    return
                self.respond({"error": "unknown route"}, 404)

        return Handler


class BarrierSGLang(FakeSGLang):
    """Require concurrent HTTP arrivals and return tagged frames out of order."""

    def __init__(self, frames_by_round, failure_round=None, hold_success=False):
        super().__init__()
        self.frames_by_round = frames_by_round
        self.num_envs = len(frames_by_round[0])
        self.barrier = threading.Barrier(self.num_envs, timeout=5)
        self.lock = threading.Lock()
        self.counts = [0] * self.num_envs
        self.finished = [
            [threading.Event() for _ in range(self.num_envs)] for _ in frames_by_round
        ]
        self.failure_round = failure_round
        self.hold_success = hold_success
        self.release_success = threading.Event()
        self.failed_response = threading.Event()
        self.return_order = []
        self.failures = []
        self.tagged_requests = []

    def handler(self):
        owner = self
        parent = super().handler()

        class Handler(parent):
            def do_POST(self):
                if self.path != "/rollout/generate":
                    return super().do_POST()
                import msgspec
                from safetensors.numpy import save

                payload = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                index = int(payload["extra_sampling_params"]["action"][0][0])
                with owner.lock:
                    cycle = owner.counts[index]
                    owner.counts[index] += 1
                    owner.requests.append(("POST", self.path, payload))
                    owner.rollout_requests.append(payload)
                    owner.tagged_requests.append((cycle, index, payload))
                try:
                    owner.barrier.wait()
                    if cycle == owner.failure_round and index == 0:
                        self.respond({"detail": "one environment failed"}, 503)
                        owner.failed_response.set()
                    else:
                        if owner.hold_success:
                            assert owner.release_success.wait(timeout=10)
                        if cycle != owner.failure_round and index + 1 < owner.num_envs:
                            assert owner.finished[cycle][index + 1].wait(timeout=5)
                        frames = owner.frames_by_round[cycle][index]
                        raw = np.ascontiguousarray(
                            frames.transpose(3, 0, 1, 2).astype(np.float32) / 255
                        )
                        body = msgspec.msgpack.encode(
                            [
                                {
                                    "request_id": f"round-{cycle}-env-{index}",
                                    "generated_output": {
                                        "__tensor__": True,
                                        "data": save({"t": raw}),
                                    },
                                    "inference_time_s": 0.01,
                                }
                            ]
                        )
                        self.send_response(200)
                        self.send_header("Content-Type", "application/msgpack")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        self.wfile.flush()
                    with owner.lock:
                        owner.return_order.append((cycle, index))
                except Exception as exc:
                    owner.failures.append(str(exc))
                    self.respond({"detail": str(exc)}, 500)
                finally:
                    owner.finished[cycle][index].set()

        return Handler


@pytest.fixture
def serve():
    servers = []

    def start(service):
        server = ThreadingHTTPServer(("127.0.0.1", 0), service.handler())
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        servers.append((server, thread))
        return f"http://127.0.0.1:{server.server_port}"

    yield start
    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def generation():
    return {
        "domain_name": "libero",
        "raw_action_dim": 10,
        "num_frames": 5,
        "width": 16,
        "height": 16,
        "fps": 20,
        "action_fps": 20.0,
        "action_view_point": "ego_view",
        "num_inference_steps": 1,
        "seed": 17,
    }


@pytest.fixture
def raw_frames():
    rng = np.random.default_rng(718)
    return [rng.integers(0, 256, (5, 16, 16, 3), dtype=np.uint8) for _ in range(3)]


def test_raw_dynamics_returns_frames_without_video_jobs(generation, serve, raw_frames):
    service = FakeSGLang(raw_frames)
    client = Cosmos3Client(serve(service), timeout_s=5)
    source = np.full((16, 16, 3), 17, dtype=np.uint8)
    actions = np.arange(40, dtype=np.float32).reshape(4, 10)
    payload = dynamics_request("move bowl", source, generation, actions)
    try:
        frames, metadata = client.generate_frames(payload)
    finally:
        client.close()
        client.close()
    np.testing.assert_array_equal(frames, raw_frames[0])
    request = service.rollout_requests[0]
    np.testing.assert_array_equal(decode_image(request["image_path"][0]), source)
    params = request["extra_sampling_params"]
    np.testing.assert_array_equal(params["action"], actions)
    assert request["rollout"] is False
    assert request["rollout_debug_mode"] is False
    assert request["seed"] == generation["seed"]
    assert params["save_output"] is False
    assert params["return_frames"] is False
    assert params["return_file_paths_only"] is False
    assert "return_raw_frames" not in params
    assert "output_path" not in params
    assert "seed" not in params
    assert metadata["request_id"] == "test-request-1"
    assert metadata["generation_time_s"] == 0.01
    assert metadata["raw_shape"] == [3, 5, 16, 16]
    assert all(path == "/rollout/generate" for _, path, _ in service.requests)


def decode_image(uri):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))))


@pytest.mark.parametrize("failure", ["http_error", "timeout"])
def test_raw_dynamics_failure_is_not_retried(generation, serve, raw_frames, failure):
    service = FakeSGLang(raw_frames)
    if failure == "http_error":
        service.error_status = 500
    else:
        service.delay_s = 0.2
    client = Cosmos3Client(
        serve(service), timeout_s=0.05 if failure == "timeout" else 5
    )
    try:
        with pytest.raises(RuntimeError, match="failed"):
            client.generate_frames(
                dynamics_request(
                    "move bowl",
                    np.zeros((16, 16, 3), dtype=np.uint8),
                    generation,
                    np.zeros((4, 10), dtype=np.float32),
                )
            )
    finally:
        client.close()
    assert len(service.rollout_requests) == 1
    assert all(path == "/rollout/generate" for _, path, _ in service.requests)


@pytest.mark.parametrize(
    "layout,dtype", [("cthw", "float32"), ("thwc", "float32"), ("thwc", "uint8")]
)
def test_raw_dynamics_decodes_supported_tensor_layouts(
    generation, serve, raw_frames, layout, dtype
):
    import torch

    service = FakeSGLang(raw_frames)
    output = torch.from_numpy(raw_frames[0].copy())
    if dtype == "float32":
        output = output.float() / 255
    if layout == "cthw":
        output = output.permute(3, 0, 1, 2)
    service.generated_output = output
    client = Cosmos3Client(serve(service), timeout_s=5)
    try:
        frames, _ = client.generate_frames(
            dynamics_request(
                "move bowl", raw_frames[0][0], generation, np.zeros((4, 10))
            )
        )
    finally:
        client.close()
    np.testing.assert_array_equal(frames, raw_frames[0])
    assert frames.dtype == np.uint8


def test_raw_dynamics_matches_official_pixel_quantization(
    generation, serve, raw_frames
):
    import torch
    from sglang.multimodal_gen.runtime.entrypoints.utils import _sample_to_uint8_frames

    service = FakeSGLang(raw_frames)
    raw = torch.from_numpy(raw_frames[0].copy()).permute(3, 0, 1, 2).float() / 255
    raw[0, 0, 0, :3] = torch.tensor([-0.5, 0.5, 1.5])
    service.generated_output = raw
    expected = np.stack(_sample_to_uint8_frames(raw))
    client = Cosmos3Client(serve(service), timeout_s=5)
    try:
        frames, _ = client.generate_frames(
            dynamics_request(
                "move bowl", raw_frames[0][0], generation, np.zeros((4, 10))
            )
        )
    finally:
        client.close()
    np.testing.assert_array_equal(frames, expected)
    np.testing.assert_array_equal(frames[0, 0, :3, 0], [0, 127, 255])


@pytest.mark.parametrize(
    "shape,response_count",
    [((3, 5, 16, 16), 0), ((3, 5, 16, 16), 2), ((1, 3, 5, 16, 16), 1)],
)
def test_raw_dynamics_rejects_batch_or_tensor_shape(
    generation, serve, raw_frames, shape, response_count
):
    import torch

    service = FakeSGLang(raw_frames)
    service.generated_output = torch.zeros(shape)
    service.response_count = response_count
    client = Cosmos3Client(serve(service), timeout_s=5)
    try:
        with pytest.raises(ValueError):
            client.generate_frames(
                dynamics_request(
                    "move bowl", raw_frames[0][0], generation, np.zeros((4, 10))
                )
            )
    finally:
        client.close()
    assert len(service.rollout_requests) == 1


@pytest.fixture
def run_config(tmp_path, generation):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("yaml")
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.0",
                "chunks_size": 1000,
                "data_path": "data/episode_{episode_index:06d}.parquet",
                "fps": 10,
            }
        )
    )
    (root / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 30, "task": "move bowl"}) + "\n"
    )
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(buffer, format="PNG")
    image = {"bytes": buffer.getvalue(), "path": None}
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "image": image,
                    "wrist_image": image,
                    "episode_index": 0,
                    "frame_index": 0,
                    "task_index": 30,
                }
            ]
        ),
        root / "data/episode_000000.parquet",
    )
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "transformer").mkdir(parents=True)
    (checkpoint / "model_index.json").write_text("{}")
    (checkpoint / "transformer/config.json").write_text('{"action_gen": true}')
    return {
        "checkpoint_path": str(checkpoint),
        "cycles": 1,
        "generation": generation,
        "dataset": {"root": str(root), "episode_index": 0, "frame_index": 0},
        "server": {
            "startup_timeout_s": 5,
            "request_timeout_s": 5,
        },
    }


@pytest.fixture
def worker_config(run_config, tmp_path):
    cfg = OmegaConf.load(
        Path(__file__).resolve().parents[2]
        / "examples/embodiment/config/cosmos3_closed_loop.yaml"
    )
    cfg.env.eval.dataset = run_config["dataset"]
    cfg.env.eval.generation = run_config["generation"]
    cfg.env.eval.server = run_config["server"]
    cfg.env.eval.artifact_dir = str(tmp_path / "artifacts")
    cfg.env.eval.max_episode_steps = 12
    cfg.env.eval.max_steps_per_rollout_epoch = 12
    cfg.rollout.model.model_path = run_config["checkpoint_path"]
    cfg.rollout.sglang.server.model_path = run_config["checkpoint_path"]
    cfg.rollout.model.num_action_chunks = 4
    cfg.rollout.model.num_frames = 5
    cfg.rollout.model.size = "16x16"
    cfg.rollout.model.num_inference_steps = 1
    cfg.rollout.sglang.seed = 17
    return cfg


def rollout_worker(cfg, url):
    """Exercise the real inference method without starting a scheduler cluster."""
    pytest.importorskip("sglang.multimodal_gen.runtime.entrypoints.action.protocol")
    worker = object.__new__(SGLangEmbodiedWorker)
    worker.sglang_adapter = Cosmos3SGLangAdapter(cfg, rank=0)
    worker.http_client = InferenceHTTPClient(url)
    worker._http_timeout_s = 5
    worker._http_max_retries = 0
    worker._http_retry_backoff_s = 0
    return worker


@pytest.mark.parametrize("debug", [False, True])
def test_workers_preserve_actions_and_feed_back_raw_images(
    worker_config, serve, raw_frames, debug
):
    import torch

    from rlinf.envs import get_env_cls
    from rlinf.workers.env.env_worker import EnvWorker

    worker_config.env.eval.debug.save_artifacts = debug
    service = FakeSGLang(raw_frames)
    url = serve(service)
    rollout = rollout_worker(worker_config, url)
    env_cfg = worker_config.env.eval
    env_worker = object.__new__(EnvWorker)
    env_worker.cfg = worker_config
    env_worker.model_cfg = worker_config.rollout.model
    env_worker._rank = 0
    env_worker._world_size = 1
    env_worker._worker_info = None
    env_worker.stage_num = 1
    env_worker.set_sglang_server_urls([url])
    env_worker.eval_env_list = env_worker._setup_env_and_wrappers(
        get_env_cls(env_cfg.env_type, env_cfg), env_cfg, num_envs_per_stage=1
    )
    env = env_worker.eval_env_list[0]
    env_worker.eval_prev_done = [torch.zeros(1, dtype=torch.bool)]
    env_worker.use_external_reward_model = False
    observations = []
    try:
        obs, _ = env.reset()
        for cycle in range(3):
            observations.append(obs["main_images"].clone())
            actions, _ = rollout.predict(obs)
            output, metrics = env_worker.env_evaluate_step(actions.numpy(), 0)
            obs = output.obs
            np.testing.assert_array_equal(
                obs["main_images"][0].numpy(), raw_frames[cycle][-1]
            )
            if not debug:
                assert env._frames == []
        assert metrics["episode_len"].item() == 12
        assert "success" not in metrics
        assert "return" not in metrics
        if debug:
            preview = Path(env_cfg.artifact_dir) / "episode_000/closed_loop.mp4"
            written_at = preview.stat().st_mtime_ns
            env.flush_video()
            env.flush_video()
            assert preview.stat().st_mtime_ns == written_at
        else:
            assert not Path(env_cfg.artifact_dir).exists()
    finally:
        env_worker.close_cosmos3_envs()
        rollout.http_client.close()

    policies = [
        body
        for method, path, body in service.requests
        if method == "POST" and path == "/v1/actions/generations"
    ]
    assert len(policies) == 3
    assert not any(path.startswith("/v1/videos") for _, path, _ in service.requests)
    dynamics = service.rollout_requests
    assert len(dynamics) == 3
    for cycle, (policy, request) in enumerate(zip(policies, dynamics, strict=True)):
        np.testing.assert_array_equal(
            request["extra_sampling_params"]["action"], service.policy_outputs[cycle]
        )
        encoded = policy["input"]["input_reference"][0].split(",", 1)[1]
        policy_image = np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))))
        np.testing.assert_array_equal(
            policy_image, decode_image(request["image_path"][0])
        )
        np.testing.assert_array_equal(policy_image, observations[cycle][0].numpy())
        assert request["prompt"] == policy["input"]["prompt"][0]
        assert "concatenated views" in request["prompt"]
        if cycle:
            np.testing.assert_array_equal(policy_image, raw_frames[cycle - 1][-1])
        if debug:
            directory = Path(env_cfg.artifact_dir) / f"episode_000/cycle_{cycle:03d}"
            np.testing.assert_array_equal(
                np.load(directory / "frames.npy"), raw_frames[cycle]
            )
            np.testing.assert_array_equal(
                np.asarray(Image.open(directory / "next_observation.png")),
                raw_frames[cycle][-1],
            )
            assert not np.array_equal(
                read_video_frames(directory / "prediction.mp4")[-1],
                raw_frames[cycle][-1],
            )
    if debug:
        assert preview.stat().st_mtime_ns == written_at
    else:
        assert not Path(env_cfg.artifact_dir).exists()


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_native_adapter_rejects_nonfinite_actions(worker_config, serve, invalid):
    service = FakeSGLang()
    values = np.zeros((4, 10), dtype=np.float32)
    values[0, 0] = invalid
    service.policy_values = values.tolist()
    worker = rollout_worker(worker_config, serve(service))
    with pytest.raises(ValueError, match="finite"):
        worker.predict(
            {
                "main_images": np.zeros((1, 16, 16, 3), dtype=np.uint8),
                "task_descriptions": ["move bowl"],
            }
        )


@pytest.mark.parametrize("shape", [(3, 10), (4, 7), (4, 64)])
def test_native_adapter_rejects_wrong_horizon_or_width(worker_config, shape):
    adapter = Cosmos3SGLangAdapter(worker_config, rank=0)
    with pytest.raises(ValueError, match="shape"):
        adapter.parse_response({"data": [{"action": {"values": np.zeros(shape)}}]}, {})


def test_native_adapter_rejects_normalization_stats(worker_config):
    worker_config.rollout.model.action_stats_path = "/unused/stats.json"
    with pytest.raises(ValueError, match="action_stats_path unset"):
        Cosmos3SGLangAdapter(worker_config, rank=0)


def test_physical_libero_conversion_remains_default(worker_config):
    worker_config.rollout.model.pop("action_output")
    worker_config.rollout.model.action_dim = 7
    adapter = Cosmos3SGLangAdapter(worker_config, rank=0)
    native = np.tile([0.2, -0.3, 0.4, 1, 0, 0, 0, 1, 0, 0.6], (4, 1))
    actions, _ = adapter.parse_response({"data": [{"action": {"values": native}}]}, {})
    expected = np.tile([0.2, -0.3, 0.4, 0, 0, 0, 0.6], (4, 1))
    np.testing.assert_allclose(actions[0].numpy(), expected)


def test_example_config_validates_without_starting_ray(worker_config):
    from examples.embodiment.eval_cosmos3_closed_loop import validate_closed_loop_cfg

    validate_closed_loop_cfg(worker_config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("runner.only_eval", False),
        ("rollout.model.action_output", "libero"),
        ("env.eval.generation.num_frames", 17),
        ("env.eval.generation.seed", 99),
        ("rollout.model.size", "32x16"),
        ("env.eval.max_episode_steps", 16),
        ("rollout.sglang.launch_server", False),
        ("env.eval.video_cfg.save_video", True),
    ],
)
def test_example_rejects_incompatible_configuration(worker_config, field, value):
    from examples.embodiment.eval_cosmos3_closed_loop import validate_closed_loop_cfg

    OmegaConf.update(worker_config, field, value)
    with pytest.raises(ValueError):
        validate_closed_loop_cfg(worker_config)


@pytest.mark.parametrize(
    "url", ["http://model-server:30000", "https://model-server:30000"]
)
def test_example_accepts_external_http_servers(worker_config, url):
    from examples.embodiment.eval_cosmos3_closed_loop import validate_closed_loop_cfg

    worker_config.rollout.sglang.launch_server = False
    worker_config.rollout.sglang.server_urls = [url]
    validate_closed_loop_cfg(worker_config)


def test_native_policy_preserves_seed_zero(worker_config):
    worker_config.rollout.sglang.seed = 0
    adapter = Cosmos3SGLangAdapter(worker_config, rank=0)
    payload, _ = adapter.build_request(
        {
            "main_images": np.zeros((1, 16, 16, 3), dtype=np.uint8),
            "task_descriptions": ["move bowl"],
        }
    )
    assert payload["parameters"]["seed"] == 0


def test_env_failure_retains_debug_actions_and_error(worker_config, serve, raw_frames):
    from rlinf.envs import get_env_cls

    service = FakeSGLang(raw_frames)
    service.error_status = 500
    cfg = worker_config.env.eval
    cfg.debug.save_artifacts = True
    env = get_env_cls(cfg.env_type, cfg)(cfg=cfg, num_envs=1)
    env.set_sglang_server_urls([serve(service)])
    try:
        env.reset()
        with pytest.raises(RuntimeError, match="test generation failure"):
            env.chunk_step(np.zeros((1, 4, 10), dtype=np.float32))
    finally:
        env.close()
    cycle = Path(cfg.artifact_dir) / "episode_000/cycle_000"
    assert (cycle / "actions_native.npy").exists()
    status = json.loads((cycle / "status.json").read_text())
    assert status["status"] == "failed"
    assert "test generation failure" in status["error"]
    assert not (cycle / "next_observation.png").exists()


def test_env_rejects_reusing_artifact_directory(worker_config):
    from rlinf.envs import get_env_cls

    cfg = worker_config.env.eval
    cfg.debug.save_artifacts = True
    root = Path(cfg.artifact_dir)
    root.mkdir()
    sentinel = root / "existing.json"
    sentinel.write_text("previous run")
    with pytest.raises(FileExistsError):
        get_env_cls(cfg.env_type, cfg)(cfg=cfg, num_envs=1)
    assert sentinel.read_text() == "previous run"


def test_embodied_import_does_not_load_text_engine():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "from rlinf.workers.rollout.sglang.sglang_embodied_worker "
            "import SGLangEmbodiedWorker; "
            "assert 'rlinf.hybrid_engines.sglang.common.sgl_engine' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("guidance", [1.0, 1.5, 3.0])
def test_policy_request_transmits_configured_guidance(worker_config, guidance):
    worker_config.rollout.model.guidance_scale = guidance
    adapter = Cosmos3SGLangAdapter(worker_config, rank=0)
    payload, _ = adapter.build_request(
        {
            "main_images": np.zeros((1, 16, 16, 3), dtype=np.uint8),
            "task_descriptions": ["move bowl"],
        }
    )
    assert payload["parameters"]["guidance_scale"] == guidance


@pytest.mark.parametrize("guidance", [None, "missing"])
def test_policy_request_omits_unconfigured_guidance(worker_config, guidance):
    if guidance == "missing":
        worker_config.rollout.model.pop("guidance_scale", None)
    else:
        worker_config.rollout.model.guidance_scale = None
    adapter = Cosmos3SGLangAdapter(worker_config, rank=0)
    payload, _ = adapter.build_request(
        {
            "main_images": np.zeros((1, 16, 16, 3), dtype=np.uint8),
            "task_descriptions": ["move bowl"],
        }
    )
    assert "guidance_scale" not in payload["parameters"]


@pytest.mark.parametrize("guidance", [0.5, -1.0, float("nan"), float("inf")])
def test_policy_request_rejects_invalid_guidance(worker_config, guidance):
    worker_config.rollout.model.guidance_scale = guidance
    with pytest.raises(ValueError, match="guidance"):
        adapter = Cosmos3SGLangAdapter(worker_config, rank=0)
        adapter.build_request(
            {
                "main_images": np.zeros((1, 16, 16, 3), dtype=np.uint8),
                "task_descriptions": ["move bowl"],
            }
        )


def test_cosmos3_extension_preserves_policy_guidance_and_video_defaults():
    pytest.importorskip("sglang.multimodal_gen")
    env = {
        **os.environ,
        "SGLANG_EXTERNAL_MODEL_PACKAGE": (
            "rlinf.models.embodiment.cosmos3.sglang_extension"
        ),
        "CUDA_VISIBLE_DEVICES": "",
        "SGLANG_DIFFUSION_PLATFORM_OVERRIDE": "cpu",
        "HF_HUB_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    script = r"""
from types import SimpleNamespace

import numpy as np

from sglang.multimodal_gen.configs.pipeline_configs.cosmos3 import Cosmos3Config
from sglang.multimodal_gen.configs.sample.cosmos3 import Cosmos3SamplingParams
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.registry import get_pipeline_config_classes
from sglang.multimodal_gen.runtime.entrypoints.action.cosmos3 import (
    build_cosmos3_action_sampling_params,
)
from sglang.multimodal_gen.runtime.entrypoints.action.protocol import (
    build_action_sampling_params,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import build_sampling_params
from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import RolloutRequest
from sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api import _build_sampling_kwargs
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.server_args import set_global_server_args

from rlinf.models.embodiment.cosmos3.closed_loop_client import dynamics_request

extension = "RLinfCosmos3Pipeline"
config_cls, sampling_cls = get_pipeline_config_classes(extension)
assert config_cls is Cosmos3Config
assert sampling_cls.__module__ == (
    "rlinf.models.embodiment.cosmos3.sglang_extension"
)
args = SimpleNamespace(
    pipeline_config=Cosmos3Config(is_edge=True, distilled_sigmas=None),
    pipeline_class_name=extension,
    model_path="/unused/checkpoint",
    backend="sglang",
    model_id=None,
    batching_max_size=1,
    num_gpus=1,
    output_path=None,
    comfyui_mode=False,
    attention_backend_config=SimpleNamespace(VSA_sparsity=0.0),
    enable_trace=False,
)
parameters = {
    "action_mode": "policy",
    "domain_name": "libero",
    "raw_action_dim": 10,
    "num_frames": 17,
    "height": 480,
    "width": 832,
    "num_inference_steps": 30,
    "seed": 17,
}

def policy(guidance=None, stock=False):
    values = dict(parameters)
    if guidance is not None:
        values["guidance_scale"] = guidance
    payload = {
        "input": {
            "prompt": "move bowl",
            "input_reference": np.zeros((1, 16, 16, 3), dtype=np.uint8),
        },
        "parameters": values,
    }
    if stock:
        return build_cosmos3_action_sampling_params(
            payload, payload["input"], args, Cosmos3SamplingParams
        )
    return build_action_sampling_params(payload, args)

assert policy().guidance_scale == 1.0
for guidance in (1.0, 1.5, 3.0):
    result = policy(guidance)
    assert result.guidance_scale == guidance, result.guidance_scale
    assert result.num_frames == 17
    assert result.num_inference_steps == 30

assert policy(3.0, stock=True).guidance_scale == 5.0

for guidance, expected in ((None, 5.0), (3.0, 3.0)):
    kwargs = {
        **parameters,
        "action_mode": "forward_dynamics",
        "action": [[0.0] * 10] * 16,
    }
    if guidance is not None:
        kwargs["guidance_scale"] = guidance
    result = SamplingParams.from_user_sampling_params_args(
        args.model_path, args, **kwargs
    )
    stock_result = Cosmos3SamplingParams(**kwargs)
    stock_result._explicit_fields = set(kwargs)
    stock_result._adjust(args)
    for item in (result, stock_result):
        assert item.guidance_scale == expected
        assert item.num_frames == 17
        assert item.action_mode == "forward_dynamics"

set_global_server_args(args)
http_parameters = {key: value for key, value in parameters.items() if key != "action_mode"}
http_parameters.update(fps=20, action_fps=20.0, action_view_point="ego_view")
native_actions = np.zeros((16, 10), dtype=np.float32)
payload = dynamics_request(
    "move bowl", np.zeros((480, 832, 3), dtype=np.uint8),
    http_parameters, native_actions,
)
request = RolloutRequest(**payload)
kwargs = _build_sampling_kwargs(request)
sampling = build_sampling_params("test-http-dynamics", **kwargs)
assert isinstance(sampling, sampling_cls)
assert sampling.guidance_scale == 5.0
assert sampling.action_mode == "forward_dynamics"
assert sampling.num_frames == 17
assert sampling.width == 832 and sampling.height == 480
prepared = prepare_request(server_args=args, sampling_params=sampling)
assert prepared.rollout is False
assert prepared.save_output is False
assert prepared.return_frames is False
assert prepared.return_raw_frames is False
assert prepared.return_file_paths_only is False
np.testing.assert_array_equal(prepared.action, native_actions)

args.pipeline_class_name = extension
args.pipeline_config.distilled_sigmas = [1.0, 0.5, 0.1, 0.0]
result = policy(3.0)
assert result.guidance_scale == 1.0
assert result.num_inference_steps == 4
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_comparison_replay_keeps_single_environment_actions(
    generation, serve, raw_frames
):
    from examples.embodiment.compare_cosmos3_libero import (
        CosmosSimulator,
        Trajectory,
        run_batched_trajectories,
    )

    initial = np.full((16, 16, 3), 25, dtype=np.uint8)
    native = np.arange(120, dtype=np.float32).reshape(12, 10)
    physical = np.arange(84, dtype=np.float32).reshape(12, 7)
    source = Trajectory(
        "source", np.repeat(initial[None], 13, axis=0), native, physical, []
    )
    service = FakeSGLang(raw_frames)
    client = Cosmos3Client(serve(service), timeout_s=5)
    try:
        (replay,) = run_batched_trajectories(
            "replay",
            CosmosSimulator(client, initial[None], ["move bowl"], generation),
            None,
            12,
            4,
            replay=[source],
        )
    finally:
        client.close()

    expected_frames = np.concatenate([initial[None], *(f[1:] for f in raw_frames)])
    np.testing.assert_array_equal(replay.frames, expected_frames)
    np.testing.assert_array_equal(replay.native_actions, native)
    np.testing.assert_array_equal(replay.libero_actions, physical)
    assert len(service.requests) == 3
    assert len(service.rollout_requests) == 3
    for cycle, request in enumerate(service.rollout_requests):
        np.testing.assert_array_equal(
            request["extra_sampling_params"]["action"],
            native[cycle * 4 : (cycle + 1) * 4],
        )
        expected_input = initial if cycle == 0 else raw_frames[cycle - 1][-1]
        np.testing.assert_array_equal(
            decode_image(request["image_path"][0]), expected_input
        )
    assert all(chunk["policy_s"] == 0 for chunk in replay.chunks)


@pytest.mark.parametrize("missing", ["native", "libero"])
def test_comparison_replay_rejects_incomplete_action_sequences(
    generation, serve, raw_frames, missing
):
    from examples.embodiment.compare_cosmos3_libero import (
        CosmosSimulator,
        Trajectory,
        run_batched_trajectories,
    )

    service = FakeSGLang(raw_frames)
    client = Cosmos3Client(serve(service), timeout_s=5)
    initial = raw_frames[0][0]
    native = np.zeros((12, 10), dtype=np.float32)
    libero = np.zeros((12, 7), dtype=np.float32)
    source = Trajectory(
        "source",
        np.repeat(initial[None], 13, axis=0),
        native[:-1] if missing == "native" else native,
        libero[:-1] if missing == "libero" else libero,
        [],
    )
    try:
        with pytest.raises(ValueError, match="cover every environment and step"):
            run_batched_trajectories(
                "replay",
                CosmosSimulator(client, initial[None], ["move bowl"], generation),
                None,
                12,
                4,
                replay=[source],
            )
    finally:
        client.close()
    assert service.requests == []


def test_comparison_pairs_preserve_frame_alignment_and_pixels(raw_frames):
    from examples.embodiment.compare_cosmos3_libero import Trajectory, paired_frames

    left = Trajectory("left", raw_frames[0].copy(), None, None, [])
    right = Trajectory("right", raw_frames[1].copy(), None, None, [])
    paired = paired_frames(left, right, "Cosmos", "LIBERO")
    assert paired.shape == (5, 48, 32, 3)
    np.testing.assert_array_equal(paired[:, 32:, :16], raw_frames[0])
    np.testing.assert_array_equal(paired[:, 32:, 16:], raw_frames[1])
    assert np.count_nonzero(paired[0, :32]) > 0
    np.testing.assert_array_equal(paired[0, :32], paired[-1, :32])
    np.testing.assert_array_equal(left.frames, raw_frames[0])
    np.testing.assert_array_equal(right.frames, raw_frames[1])


def test_comparison_pairs_reject_misaligned_trajectories(raw_frames):
    from examples.embodiment.compare_cosmos3_libero import Trajectory, paired_frames

    left = Trajectory("left", raw_frames[0], None, None, [])
    right = Trajectory("right", raw_frames[1][:-1], None, None, [])
    with pytest.raises(ValueError, match="identical frame counts"):
        paired_frames(left, right, "Cosmos", "LIBERO")


def test_comparison_cases_expand_hydra_lists_without_changing_config():
    from hydra import compose, initialize_config_dir

    from examples.embodiment.compare_cosmos3_libero import comparison_cases

    config_dir = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = compose(
            config_name="cosmos3_libero_compare",
            overrides=["comparison.task_ids=[6,0]", "comparison.reset_ids=[1,0]"],
        )
    original = OmegaConf.to_container(cfg.comparison)
    OmegaConf.set_readonly(cfg, True)
    assert comparison_cases(cfg.comparison) == [(6, 1), (6, 0), (0, 1), (0, 0)]
    assert OmegaConf.to_container(cfg.comparison) == original


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("task_ids", 6),
        ("task_ids", []),
        ("reset_ids", []),
        ("task_ids", [6, 6]),
        ("reset_ids", [0, 0]),
        ("task_ids", [True]),
        ("reset_ids", [0.5]),
        ("task_ids", [-1]),
    ],
)
def test_comparison_cases_reject_ambiguous_or_overlapping_selection(field, invalid):
    from examples.embodiment.compare_cosmos3_libero import comparison_cases

    cfg = OmegaConf.create({"task_ids": [6], "reset_ids": [0]})
    cfg[field] = invalid
    with pytest.raises(ValueError, match=field):
        comparison_cases(cfg)


def test_concurrent_dynamics_preserve_input_order(generation, serve, raw_frames):
    service = BarrierSGLang([raw_frames])
    client = Cosmos3Client(serve(service), timeout_s=10)
    requests = []
    for index in range(3):
        actions = np.zeros((4, 10), dtype=np.float32)
        actions[:, 0] = index
        requests.append(
            dynamics_request(
                f"environment {index}", raw_frames[index][0], generation, actions
            )
        )
    try:
        results = client.generate_frames_batch(requests)
    finally:
        client.close()
    assert not service.failures
    assert service.counts == [1, 1, 1]
    assert service.return_order == [(0, 2), (0, 1), (0, 0)]
    for index, (frames, metadata) in enumerate(results):
        np.testing.assert_array_equal(frames, raw_frames[index])
        assert metadata["request_id"] == f"round-0-env-{index}"


def test_concurrent_dynamics_drain_failed_batch_without_retry(
    generation, serve, raw_frames
):
    from concurrent.futures import ThreadPoolExecutor, TimeoutError

    service = BarrierSGLang([raw_frames[:2]], failure_round=0, hold_success=True)
    client = Cosmos3Client(serve(service), timeout_s=10)
    requests = []
    for index in range(2):
        actions = np.full((4, 10), index, dtype=np.float32)
        requests.append(
            dynamics_request("move bowl", raw_frames[index][0], generation, actions)
        )
    try:
        with ThreadPoolExecutor(max_workers=1) as caller:
            result = caller.submit(client.generate_frames_batch, requests)
            try:
                assert service.failed_response.wait(timeout=5)
                # The successful request cannot finish until the test releases it.
                with pytest.raises(TimeoutError):
                    result.result(timeout=0.5)
            finally:
                service.release_success.set()
            with pytest.raises(RuntimeError, match="environment 0"):
                result.result(timeout=5)
        assert all(event.is_set() for event in service.finished[0])
    finally:
        service.release_success.set()
        client.close()
    assert not service.failures
    assert service.counts == [1, 1]


@pytest.fixture
def parallel_env_config(worker_config):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root = Path(worker_config.env.eval.dataset.root)
    for index, color in enumerate(("red", "green")):
        buffer = io.BytesIO()
        Image.new("RGB", (16, 16), color).save(buffer, format="PNG")
        image = {"bytes": buffer.getvalue(), "path": None}
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "image": image,
                        "wrist_image": image,
                        "episode_index": index,
                        "frame_index": index,
                        "task_index": 30 + index,
                    }
                ]
            ),
            root / f"data/episode_{index:06d}.parquet",
        )
    (root / "meta/tasks.jsonl").write_text(
        "".join(
            json.dumps({"task_index": 30 + i, "task": f"move bowl {i}"}) + chr(10)
            for i in range(2)
        )
    )
    cfg = worker_config.env.eval
    cfg.total_num_envs = 2
    cfg.dataset.episode_indices = [0, 1]
    cfg.dataset.frame_indices = [0, 1]
    return cfg


@pytest.mark.parametrize("debug", [False, True])
def test_parallel_env_observations_remain_isolated(
    parallel_env_config, serve, raw_frames, debug
):
    from rlinf.envs.sim.cosmos3.env import Cosmos3Env

    cfg = parallel_env_config
    cfg.debug.save_artifacts = debug
    frames = [[value, np.roll(value, 3, axis=2)] for value in raw_frames]
    service = BarrierSGLang(frames)
    env = Cosmos3Env(cfg, num_envs=2)
    env.set_sglang_server_urls([serve(service)])
    actions = np.zeros((2, 4, 10), dtype=np.float32)
    actions[1, :, 0] = 1
    try:
        initial, _ = env.reset()
        initial_images = initial["main_images"].numpy().copy()
        assert initial_images.shape == (2, 16, 16, 3)
        assert not np.array_equal(initial_images[0], initial_images[1])
        assert len(set(initial["task_descriptions"])) == 2
        for cycle in range(3):
            observations, rewards, terminated, truncated, info = env.chunk_step(actions)
            assert rewards.shape == terminated.shape == truncated.shape == (2, 4)
            assert not rewards.any() and not terminated.any()
            assert bool(truncated[:, -1].all()) == (cycle == 2)
            for index in range(2):
                np.testing.assert_array_equal(
                    observations[0]["main_images"][index].numpy(),
                    frames[cycle][index][-1],
                )
                if debug:
                    directory = (
                        Path(cfg.artifact_dir)
                        / f"episode_000/env_{index:03d}/cycle_{cycle:03d}"
                    )
                    np.testing.assert_array_equal(
                        np.load(directory / "frames.npy"), frames[cycle][index]
                    )
        assert info[0]["episode"]["episode_len"].tolist() == [12, 12]
        assert info[0]["episode"]["generated_chunks"].tolist() == [3, 3]
    finally:
        env.close()
    assert not service.failures
    assert service.counts == [3, 3]
    assert service.return_order == [
        (cycle, index) for cycle in range(3) for index in (1, 0)
    ]
    for cycle, index, payload in service.tagged_requests:
        expected = initial_images[index] if cycle == 0 else frames[cycle - 1][index][-1]
        np.testing.assert_array_equal(decode_image(payload["image_path"][0]), expected)
        np.testing.assert_array_equal(
            payload["extra_sampling_params"]["action"], actions[index]
        )
    if debug:
        for index in range(2):
            assert (
                Path(cfg.artifact_dir) / f"episode_000/env_{index:03d}/closed_loop.mp4"
            ).is_file()
    else:
        assert not Path(cfg.artifact_dir).exists()


def test_parallel_env_failed_batch_does_not_advance_any_state(
    parallel_env_config, serve, raw_frames
):
    from rlinf.envs.sim.cosmos3.env import Cosmos3Env

    cfg = parallel_env_config
    cfg.max_episode_steps = 8
    frames = [[value, np.roll(value, 3, axis=2)] for value in raw_frames]
    service = BarrierSGLang(frames, failure_round=0)
    env = Cosmos3Env(cfg, num_envs=2)
    env.set_sglang_server_urls([serve(service)])
    actions = np.zeros((2, 4, 10), dtype=np.float32)
    actions[1, :, 0] = 1
    try:
        initial, _ = env.reset()
        initial_images = initial["main_images"].numpy().copy()
        with pytest.raises(RuntimeError, match="environment 0"):
            env.chunk_step(actions)
        assert all(event.is_set() for event in service.finished[0])
        service.failure_round = None
        _, _, _, truncated, info = env.chunk_step(actions)
        assert not truncated.any()
        assert "episode" not in info[0]
        observations, _, _, truncated, info = env.chunk_step(actions)
        assert truncated[:, -1].all()
        assert info[0]["episode"]["episode_len"].tolist() == [8, 8]
        for index in range(2):
            np.testing.assert_array_equal(
                observations[0]["main_images"][index].numpy(), frames[2][index][-1]
            )
    finally:
        env.close()
    assert not service.failures
    assert service.counts == [3, 3]
    for cycle, index, payload in service.tagged_requests:
        expected = initial_images[index] if cycle <= 1 else frames[1][index][-1]
        np.testing.assert_array_equal(decode_image(payload["image_path"][0]), expected)


def test_concurrent_comparison_replay_keeps_each_saved_sequence(
    generation, serve, raw_frames
):
    from examples.embodiment.compare_cosmos3_libero import (
        CosmosSimulator,
        Trajectory,
        run_batched_trajectories,
    )

    frames = [[value, np.roll(value, 3, axis=2)] for value in raw_frames]
    service = BarrierSGLang(frames)
    client = Cosmos3Client(serve(service), timeout_s=10)
    initial = np.stack([raw_frames[0][0], raw_frames[1][0]])
    sources = []
    for index in range(2):
        native = np.arange(120, dtype=np.float32).reshape(12, 10)
        native[:, 0] = index
        physical = np.arange(84, dtype=np.float32).reshape(12, 7) + index
        sources.append(
            Trajectory(
                f"source-{index}",
                np.repeat(initial[index][None], 13, axis=0),
                native,
                physical,
                [],
            )
        )
    try:
        results = run_batched_trajectories(
            "replay",
            CosmosSimulator(client, initial, ["task zero", "task one"], generation),
            None,
            12,
            4,
            replay=sources,
        )
    finally:
        client.close()
    assert service.counts == [3, 3]
    assert not service.failures
    assert len(service.requests) == 6
    assert all(path == "/rollout/generate" for _, path, _ in service.requests)
    for index, result in enumerate(results):
        expected_frames = np.concatenate(
            [
                initial[index][None],
                *(round_frames[index][1:] for round_frames in frames),
            ]
        )
        np.testing.assert_array_equal(result.frames, expected_frames)
        np.testing.assert_array_equal(
            result.native_actions, sources[index].native_actions
        )
        np.testing.assert_array_equal(
            result.libero_actions, sources[index].libero_actions
        )
        assert all(chunk["policy_s"] == 0 for chunk in result.chunks)
    for cycle, index, payload in service.tagged_requests:
        expected_input = initial[index] if cycle == 0 else frames[cycle - 1][index][-1]
        np.testing.assert_array_equal(
            decode_image(payload["image_path"][0]), expected_input
        )
        np.testing.assert_array_equal(
            payload["extra_sampling_params"]["action"],
            sources[index].native_actions[cycle * 4 : (cycle + 1) * 4],
        )


def test_concurrent_comparison_failure_keeps_all_initial_images(
    generation, serve, raw_frames
):
    from examples.embodiment.compare_cosmos3_libero import CosmosSimulator

    frames = [[value, np.roll(value, 3, axis=2)] for value in raw_frames[:2]]
    service = BarrierSGLang(frames, failure_round=0)
    client = Cosmos3Client(serve(service), timeout_s=10)
    initial = np.stack([raw_frames[0][0], raw_frames[1][0]])
    simulator = CosmosSimulator(client, initial, ["task zero", "task one"], generation)
    native = np.zeros((2, 4, 10), dtype=np.float32)
    native[1, :, 0] = 1
    physical = np.zeros((2, 4, 7), dtype=np.float32)
    try:
        np.testing.assert_array_equal(simulator.reset(), initial)
        with pytest.raises(RuntimeError, match="environment 0"):
            simulator.step(native, physical)
        service.failure_round = None
        next_frames, _ = simulator.step(native, physical)
        for index in range(2):
            np.testing.assert_array_equal(next_frames[index], frames[1][index][1:])
    finally:
        client.close()
    assert service.counts == [2, 2]
    assert not service.failures
    for _, index, payload in service.tagged_requests:
        np.testing.assert_array_equal(
            decode_image(payload["image_path"][0]), initial[index]
        )


def test_parallel_env_broadcasts_scalar_reset_fixture(worker_config, serve):
    from rlinf.envs.sim.cosmos3.env import Cosmos3Env

    cfg = worker_config.env.eval
    cfg.dataset.episode_indices = None
    cfg.dataset.frame_indices = None
    env = Cosmos3Env(cfg, num_envs=2)
    service = FakeSGLang()
    env.set_sglang_server_urls([serve(service)])
    try:
        observation, _ = env.reset()
        np.testing.assert_array_equal(
            observation["main_images"][0].numpy(),
            observation["main_images"][1].numpy(),
        )
        assert (
            observation["task_descriptions"][0] == observation["task_descriptions"][1]
        )
    finally:
        env.close()
    assert service.requests == []


@pytest.mark.parametrize("field", ["episode_indices", "frame_indices"])
def test_parallel_env_rejects_incomplete_reset_fixture_list(worker_config, field):
    from rlinf.envs.sim.cosmos3.env import Cosmos3Env

    cfg = worker_config.env.eval
    cfg.dataset[field] = [0]
    with pytest.raises(ValueError, match=field):
        Cosmos3Env(cfg, num_envs=2)


@pytest.mark.parametrize("as_tensor", [False, True])
def test_native_action_conversion_preserves_input_and_padded_channels(
    worker_config, tmp_path, as_tensor
):
    import torch

    stats = tmp_path / "action_stats.json"
    stats.write_text(
        json.dumps(
            {
                "global_raw": {
                    "q01": [-0.5] * 3 + [-1.0] * 6 + [0.0],
                    "q99": [0.5] * 3 + [1.0] * 6 + [1.0],
                }
            }
        )
    )
    worker_config.rollout.model.action_output = "libero"
    worker_config.rollout.model.action_dim = 7
    worker_config.rollout.model.action_stats_path = str(stats)
    worker_config.rollout.model.action_normalization = "quantile_rot"
    adapter = Cosmos3SGLangAdapter(worker_config, rank=0)
    native = np.array(
        [
            [0.2, -0.4, 0.6, 1, 0, 0, 0, 1, 0, -1, 999],
            [0.4, -0.6, 0.8, 0, 1, 0, -1, 0, 0, -0.2, 999],
            [0.6, -0.8, 0.2, 1, 0, 0, 0, 0, 1, 0.2, 999],
            [0.8, -0.2, 0.4, 1, 0, 0, 0, 1, 0, 1, 999],
        ],
        dtype=np.float32,
    )[None]
    original = native.copy()
    physical = adapter.native_actions_to_libero(
        torch.from_numpy(native) if as_tensor else native
    )
    np.testing.assert_array_equal(native, original)
    assert physical.shape == (1, 4, 7) and physical.dtype == np.float32
    np.testing.assert_allclose(physical[..., :3], native[..., :3] * 0.5, atol=1e-7)
    np.testing.assert_allclose(
        physical[0, :, 3:6],
        [[0, 0, 0], [0, 0, np.pi / 2], [np.pi / 2, 0, 0], [0, 0, 0]],
        atol=1e-6,
    )
    np.testing.assert_allclose(physical[0, :, -1], [0, 0.4, 0.6, 1], atol=1e-7)


@pytest.fixture
def comparison_vendor(tmp_path):
    """Fake only the vendor SDK used by real LIBERO and scheduler processes."""
    pytest.importorskip("gym")
    package = tmp_path / "vendor"
    arrivals = tmp_path / "arrivals"
    arrivals.mkdir()
    files = {
        "libero/__init__.py": "",
        "libero/libero/__init__.py": """
            from pathlib import Path
            def get_libero_path(name):
                return str(Path(__file__).parent)
            def set_libero_default_path(path):
                pass
        """,
        "libero/libero/benchmark.py": """
            from types import SimpleNamespace
            import numpy as np
            class Benchmark:
                n_tasks = 2
                def __init__(self, task_order_index=0):
                    assert task_order_index == 0
                def get_num_tasks(self):
                    return self.n_tasks
                def get_task(self, index):
                    return SimpleNamespace(
                        problem_folder="", bddl_file=f"task_{index}.bddl",
                        language=f"move object {index}",
                    )
                def get_task_init_states(self, index):
                    return np.array([[index, reset] for reset in range(5)], dtype=np.float64)
            def get_benchmark_dict():
                return {"fake": Benchmark}
            def get_benchmark(name):
                return get_benchmark_dict()[name]
        """,
        "libero/libero/envs.py": """
            import json
            import os
            import time
            from pathlib import Path
            import numpy as np

            class OffScreenRenderEnv:
                def __init__(self, bddl_file_name, camera_heights, camera_widths, **kwargs):
                    self.task = int(Path(bddl_file_name).stem.split("_")[-1])
                    self.shape = (camera_heights, camera_widths, 3)
                    self.state = np.array([self.task, 0], dtype=np.float64)
                    self.tick = 0
                    self.round = -1
                    self.seed_value = None
                def seed(self, value):
                    self.seed_value = value
                def observation(self):
                    marker = 40 + 40 * self.task + 10 * int(self.state[1]) + self.tick
                    image = np.full(self.shape, marker, dtype=np.uint8)
                    return {
                        "agentview_image": image.copy(),
                        "robot0_eye_in_hand_image": image.copy(),
                        "robot0_eef_pos": np.zeros(3),
                        "robot0_eef_quat": np.array([0., 0., 0., 1.]),
                        "robot0_gripper_qpos": np.zeros(2),
                    }
                def reset(self, **kwargs):
                    self.round += 1
                    self.tick = 0
                    return self.observation()
                def set_init_state(self, state):
                    self.state = np.asarray(state).copy()
                    return self.observation()
                def get_sim_state(self):
                    return np.concatenate([self.state, [self.tick, self.seed_value]])
                def check_success(self):
                    return self.task == 0 and self.tick >= 2
                def step(self, action):
                    self.tick += 1
                    root = Path(os.environ["FAKE_LIBERO_ARRIVALS"])
                    prefix = f"round-{self.round}-step-{self.tick}"
                    (root / f"{prefix}-task-{self.task}.json").write_text(json.dumps({
                        "pid": os.getpid(), "action": np.asarray(action).tolist(),
                    }))
                    if (root / "exit").exists() and self.task == 1 and self.tick == 1:
                        os._exit(23)
                    parties = int(os.environ.get("FAKE_LIBERO_PARTIES", "1"))
                    deadline = time.monotonic() + 8
                    while len(list(root.glob(f"{prefix}-task-*.json"))) < parties:
                        if time.monotonic() >= deadline:
                            raise RuntimeError("LIBERO workers did not step concurrently")
                        time.sleep(0.002)
                    if (root / "fail").exists() and self.task == 1 and self.tick == 4:
                        raise RuntimeError("controlled physical step failure")
                    return self.observation(), 2.5, self.check_success(), {}
                def close(self):
                    pass
        """,
    }
    for relative, source in files.items():
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textwrap.dedent(source))
    return package, arrivals


def comparison_config(cfg, tmp_path):
    stats = tmp_path / "comparison_stats.json"
    stats.write_text(
        json.dumps(
            {
                "global_raw": {
                    "q01": [-0.5] * 3 + [-1.0] * 6 + [0.0],
                    "q99": [0.5] * 3 + [1.0] * 6 + [1.0],
                }
            }
        )
    )
    cfg.comparison = {
        "suite": "fake",
        "task_order_index": 0,
        "reset_seed": 17,
        "settle_steps": 0,
        "camera_size": 16,
        "steps": 8,
        "num_envs": 3,
        "action_stats_path": str(stats),
        "save_raw_frames": True,
    }
    cfg.rollout.sglang.server.batching_max_size = 3
    cfg.rollout.sglang.http_timeout_s = 30
    cfg.rollout.sglang.http_max_retries = 0
    config_path = tmp_path / "comparison.yaml"
    OmegaConf.save(cfg, config_path)
    return config_path


def run_comparison_process(source, config_path, tmp_path, vendor, *args, parties=1):
    package, arrivals = vendor
    repo = Path(__file__).resolve().parents[2]
    script = tmp_path / "comparison_process.py"
    script.write_text(textwrap.dedent(source))
    result = subprocess.run(
        [sys.executable, str(script), str(config_path), *map(str, args)],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(package), str(repo)]),
            "FAKE_LIBERO_ARRIVALS": str(arrivals),
            "FAKE_LIBERO_PARTIES": str(parties),
            "LIBERO_TYPE": "standard",
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "RAY_ADDRESS": "local",
        },
        text=True,
        capture_output=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_comparison_libero_reset_ids_and_failure_recovery(
    worker_config, tmp_path, comparison_vendor
):
    config = comparison_config(worker_config, tmp_path)
    run_comparison_process(
        """
        import multiprocessing
        import os
        import sys
        import time
        from pathlib import Path
        import numpy as np
        from omegaconf import OmegaConf
        from examples.embodiment.cosmos3_libero_env import LiberoComparisonEnv

        def main():
            cfg = OmegaConf.load(sys.argv[1])
            arrivals = Path(os.environ["FAKE_LIBERO_ARRIVALS"])
            native = np.zeros((2, 4, 10), dtype=np.float32)
            physical = np.arange(56, dtype=np.float32).reshape(2, 4, 7) / 100
            simulator = LiberoComparisonEnv(cfg, [(1, 1), (0, 1)])
            pids = []
            try:
                initial = simulator.reset()
                np.testing.assert_array_equal(initial[:, 0, 0, 0], [90, 50])
                np.testing.assert_array_equal(simulator.init_states, [[1, 1], [0, 1]])
                np.testing.assert_array_equal(simulator.reference_states, [[1, 1, 0, 17], [0, 1, 0, 17]])
                frames, details = simulator.step(native, physical)
                pids = [item["worker_pid"] for item in details]
                assert len(set(pids)) == 2 and os.getpid() not in pids
                np.testing.assert_array_equal(frames[:, :, 0, 0, 0], [[91, 92, 93, 94], [51, 52, 53, 54]])
                assert all(item["rewards"] == [2.5] * 4 for item in details)
                for index, task in enumerate([1, 0]):
                    for step in range(4):
                        import json
                        record = json.loads((arrivals / f"round-0-step-{step+1}-task-{task}.json").read_text())
                        assert record["pid"] == pids[index]
                        np.testing.assert_allclose(record["action"], physical[index, step])
                np.testing.assert_array_equal(simulator.reset(), initial)
                assert not set(pids) & {child.pid for child in multiprocessing.active_children()}
                (arrivals / "fail").touch()
                try:
                    simulator.step(native, physical)
                except (EOFError, BrokenPipeError, OSError):
                    pass
                else:
                    raise AssertionError("Failed physical step was accepted")
                try:
                    simulator.step(native, physical)
                except RuntimeError as exc:
                    assert "Reset" in str(exc)
                else:
                    raise AssertionError("Failed physical environment advanced")
                (arrivals / "fail").unlink()
                np.testing.assert_array_equal(simulator.reset(), initial)
                recovered, _ = simulator.step(native, physical)
                np.testing.assert_array_equal(recovered[:, -1, 0, 0, 0], [94, 54])
                np.testing.assert_array_equal(simulator.reset(), initial)
                (arrivals / "exit").touch()
                pids = [child.pid for child in multiprocessing.active_children()]
                assert len(pids) == 2
                try:
                    simulator.step(native, physical)
                except (EOFError, BrokenPipeError, OSError):
                    pass
                else:
                    raise AssertionError("Terminated worker was accepted")
            finally:
                start = time.monotonic()
                simulator.close()
                assert time.monotonic() - start < 15
            simulator.close()
            for pid in pids:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise AssertionError(f"Owned worker {pid} survived close")

        if __name__ == "__main__":
            main()
        """,
        config,
        tmp_path,
        comparison_vendor,
        parties=2,
    )


@pytest.mark.placement
def test_comparison_workers_use_rank_endpoints_and_preserve_batch_replays(
    worker_config, tmp_path, comparison_vendor, serve, raw_frames
):
    pytest.importorskip("sglang.multimodal_gen.runtime.entrypoints.action.protocol")
    config = comparison_config(worker_config, tmp_path)
    values = np.zeros((4, 10), dtype=np.float32)
    values[:, 3:9] = [1, 0, 0, 0, 1, 0]
    values[:, -1] = [-1, -0.2, 0.2, 1]
    services = [FakeSGLang(raw_frames), FakeSGLang(raw_frames)]
    barrier = threading.Barrier(2)
    expected = []
    for index in range(3):
        chunk = values.copy()
        chunk[:, :3] = [0.2 * (index + 1), -0.4, 0.6]
        expected.append(chunk)
    expected = np.stack(expected)
    for service in services:
        service.model_path = worker_config.rollout.model.model_path
        service.policy_values_by_image = {i: values.tolist() for i in range(256)}
        service.policy_values_by_image.update(
            {10 * (i + 1): expected[i].tolist() for i in range(3)}
        )
        service.reverse_policy_response = True
        service.policy_barrier = barrier
    urls = [serve(service) for service in services]
    output = tmp_path / "results"
    output.mkdir()
    run_comparison_process(
        """
        import json
        import os
        import sys
        from pathlib import Path
        import numpy as np
        import ray
        from omegaconf import OmegaConf
        from rlinf.scheduler import Cluster, NodePlacementStrategy
        from examples.embodiment.cosmos3_comparison_worker import Cosmos3ComparisonWorker

        def main():
            cfg = OmegaConf.load(sys.argv[1])
            output = Path(sys.argv[2])
            cluster_cfg = OmegaConf.create({
                "num_nodes": 1, "component_placement": {},
                "node_groups": [{
                    "label": "comparison_cpu", "node_ranks": "0",
                    "env_configs": [{
                        "node_ranks": "0", "python_interpreter_path": sys.executable,
                        "env_vars": [{"PYTHONPATH": os.environ["PYTHONPATH"]},
                                     {"FAKE_LIBERO_ARRIVALS": os.environ["FAKE_LIBERO_ARRIVALS"]},
                                     {"FAKE_LIBERO_PARTIES": "1"}, {"LIBERO_TYPE": "standard"}],
                    }],
                }],
            })
            group = None
            try:
                cluster = Cluster(cluster_cfg=cluster_cfg)
                group = Cosmos3ComparisonWorker.create_group(cfg, None).launch(
                    cluster, name="ComparisonCPU",
                    placement_strategy=NodePlacementStrategy([0, 0], node_group_label="comparison_cpu"),
                )
                group.set_sglang_server_urls(sys.argv[3:5]).wait()
                metadata = group.init_worker().wait()
                assert [row["rank"] for row in metadata] == [0, 1]
                assert [row["server_url"] for row in metadata] == sys.argv[3:5]
                assert len({row["pid"] for row in metadata}) == 2
                images = np.stack([np.full((16, 16, 3), 10*(i+1), dtype=np.uint8) for i in range(3)])
                batches = group.predict_batch(images, ["move bowl"]*3).wait()
                for rank, (native, physical) in enumerate(batches):
                    np.savez(output / f"policy_{rank}.npz", native=native, physical=physical)
                mixed = group.execute_on(0).predict_batch(images, ["move bowl", "open drawer", "move bowl"]).wait()[0]
                np.savez(output / "mixed.npz", native=mixed[0], physical=mixed[1])
                left = group.execute_on(0).run_cases([(1, 1), (0, 1)], str(output))
                right = group.execute_on(1).run_cases([(0, 0)], str(output))
                results = left.wait()[0] + right.wait()[0]
                assert len(results) == 3
                assert [result["instruction"] for result in results] == [
                    "move object 1", "move object 0", "move object 0"
                ]
                (output / "workers.json").write_text(json.dumps(metadata))
            finally:
                if group is not None:
                    group.close().wait()
                    group._close()
                ray.shutdown()

        if __name__ == "__main__":
            main()
        """,
        config,
        tmp_path,
        comparison_vendor,
        output,
        *urls,
    )
    for name in ["policy_0.npz", "policy_1.npz", "mixed.npz"]:
        result = np.load(output / name)
        np.testing.assert_array_equal(result["native"], expected)
        np.testing.assert_allclose(
            result["physical"][..., :3], expected[..., :3] * 0.5, atol=1e-7
        )
        np.testing.assert_allclose(result["physical"][..., 3:6], 0, atol=1e-7)
        np.testing.assert_array_equal(
            result["physical"][..., -1], np.tile([1, 1, -1, -1], (3, 1))
        )
    requests = [
        [
            body
            for method, path, body in service.requests
            if path == "/v1/actions/generations"
        ]
        for service in services
    ]
    assert [len(row) for row in requests] == [11, 5]
    assert [len(body["input"]["prompt"]) for body in requests[0][:3]] == [3, 2, 1]
    assert [len(body["input"]["prompt"]) for body in requests[1]] == [3, 1, 1, 1, 1]
    assert [len(service.rollout_requests) for service in services] == [8, 4]
    for task, reset, rank in [(1, 1, 0), (0, 1, 0), (0, 0, 1)]:
        case = output / f"task_{task:03d}" / f"reset_{reset:03d}"
        manifest = json.loads((case / "manifest.json").read_text())
        assert manifest["same_initial_pixels"] and manifest["same_replay_actions"]
        assert manifest["first_policy_chunk_equal"]
        assert manifest["worker"]["rank"] == rank
        assert manifest["worker"]["server_url"] == urls[rank]
        assert all((case / video).is_file() for video in manifest["videos"])
        for source, replay in [
            ("cosmos_rollout", "libero_replay_cosmos"),
            ("libero_rollout", "cosmos_replay_libero"),
        ]:
            for array in ["actions_native.npy", "actions_libero.npy"]:
                np.testing.assert_array_equal(
                    np.load(case / source / array), np.load(case / replay / array)
                )
            frames = np.load(case / source / "frames.npy")
            assert frames.shape == (9, 16, 16, 3)
