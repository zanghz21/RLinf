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
        self.policy_outputs = []

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
                    parameters = payload["parameters"]
                    horizon, width = (
                        parameters["num_frames"] - 1,
                        parameters["raw_action_dim"],
                    )
                    values = owner.policy_values
                    if values is None:
                        values = (
                            np.linspace(
                                -2, 2, horizon * width, dtype=np.float32
                            ).reshape(horizon, width)
                            + len(owner.policy_outputs)
                        ).tolist()
                    owner.policy_outputs.append(values)
                    self.respond(
                        {
                            "data": [
                                {
                                    "input_index": 0,
                                    "action": {
                                        "values": values,
                                        "shape": [horizon, width],
                                        "raw_action_dim": width,
                                        "domain_id": 5,
                                        "action_mode": "policy",
                                    },
                                }
                            ]
                        }
                    )
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
                self.respond({"error": "unknown route"}, 404)

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
