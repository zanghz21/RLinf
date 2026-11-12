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

"""Exercise Cosmos3 worker boundaries against an HTTP service without weights."""

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from rlinf.models.embodiment.cosmos3.closed_loop_client import Cosmos3Client
from rlinf.models.embodiment.cosmos3.closed_loop_io import (
    read_video_frames,
    save_video,
)
from rlinf.models.embodiment.cosmos3.sglang_adapter import Cosmos3SGLangAdapter
from rlinf.utils.http_client import InferenceHTTPClient
from rlinf.workers.rollout.sglang.sglang_embodied_worker import SGLangEmbodiedWorker

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")
pytest.importorskip("requests")


class FakeSGLang:
    """Serve policy envelopes and asynchronous video jobs at the process boundary."""

    def __init__(self, videos=(), mode="completed", cancel_status=200):
        self.videos = videos
        self.mode = mode
        self.cancel_status = cancel_status
        self.model_path = "/unused/checkpoint"
        self.requests = []
        self.policy_values = None
        self.policy_outputs = []
        self.jobs = {}

    def handler(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def respond(self, value, status=200):
                if self.headers.get("Accept") == "application/msgpack":
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
                elif self.path == "/v1/videos":
                    if owner.mode == "submission_failed":
                        self.respond({"error": "test submission failure"}, 503)
                        return
                    job_id = f"video-{len(owner.jobs)}"
                    owner.jobs[job_id] = {"polls": 0, "index": len(owner.jobs)}
                    self.respond({"id": job_id, "status": "queued"})
                else:
                    self.respond({"error": "unknown route"}, 404)

            def do_GET(self):
                owner.requests.append(("GET", self.path, None))
                if self.path == "/server_info":
                    self.respond(
                        {"model_path": owner.model_path, "pipeline": "Cosmos3Pipeline"}
                    )
                    return
                pieces = self.path.strip("/").split("/")
                if len(pieces) < 3 or pieces[:2] != ["v1", "videos"]:
                    self.respond({"error": "unknown route"}, 404)
                    return
                job_id = pieces[2]
                job = owner.jobs[job_id]
                if pieces[-1] == "content":
                    self.respond(owner.videos[job["index"] % len(owner.videos)])
                    return
                job["polls"] += 1
                status = owner.mode
                if status == "completed" and job["polls"] == 1:
                    status = "in_progress"
                self.respond(
                    {"id": job_id, "status": status, "error": "test generation failure"}
                )

            def do_DELETE(self):
                owner.requests.append(("DELETE", self.path, None))
                self.respond({"status": "cancelled"}, owner.cancel_status)

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
def videos(tmp_path):
    pytest.importorskip("imageio.v2")
    pytest.importorskip("imageio_ffmpeg")
    result = []
    for index in range(3):
        frames = np.full((5, 16, 16, 3), 40 + index * 50, dtype=np.uint8)
        frames[-1, :, :, 0] += 20
        path = tmp_path / f"source_{index}.mp4"
        save_video(path, frames, 20)
        result.append(path.read_bytes())
    return result


@pytest.mark.parametrize("cancel_status", [200, 500])
def test_failed_video_is_cancelled_and_keeps_original_error(
    tmp_path, serve, cancel_status
):
    service = FakeSGLang(mode="failed", cancel_status=cancel_status)
    client = Cosmos3Client(serve(service), timeout_s=5, poll_interval_s=0.001)
    try:
        with pytest.raises(RuntimeError, match="test generation failure"):
            client.generate_video(
                {"action_mode": "forward_dynamics"}, tmp_path / "prediction.mp4"
            )
    finally:
        client.close()
    assert ("DELETE", "/v1/videos/video-0", None) in service.requests
    assert not any(path.endswith("/content") for _, path, _ in service.requests)


@pytest.mark.parametrize("cancel_status", [200, 500])
def test_video_timeout_cancels_job_without_masking_timeout(
    tmp_path, serve, cancel_status
):
    service = FakeSGLang(mode="in_progress", cancel_status=cancel_status)
    client = Cosmos3Client(serve(service), timeout_s=0.15, poll_interval_s=0.01)
    try:
        with pytest.raises(TimeoutError, match="exceeded"):
            client.generate_video(
                {"action_mode": "forward_dynamics"}, tmp_path / "prediction.mp4"
            )
    finally:
        client.close()
    assert ("DELETE", "/v1/videos/video-0", None) in service.requests


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
            "poll_interval_s": 0.001,
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


def test_workers_preserve_actions_and_feed_back_generated_images(
    worker_config, serve, videos
):
    import torch

    from rlinf.envs import get_env_cls
    from rlinf.workers.env.env_worker import EnvWorker

    service = FakeSGLang(videos)
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
        for _ in range(3):
            observations.append(obs["main_images"].clone())
            actions, _ = rollout.predict(obs)
            output, metrics = env_worker.env_evaluate_step(actions.numpy(), 0)
            obs = output.obs
        assert metrics["episode_len"].item() == 12
        assert "success" not in metrics
        assert "return" not in metrics
    finally:
        env_worker.close_cosmos3_envs()
        rollout.http_client.close()

    posts = [
        (path, body) for method, path, body in service.requests if method == "POST"
    ]
    assert [path for path, _ in posts] == ["/v1/actions/generations", "/v1/videos"] * 3
    for cycle in range(3):
        policy = posts[2 * cycle][1]
        dynamics = posts[2 * cycle + 1][1]
        np.testing.assert_array_equal(dynamics["action"], service.policy_outputs[cycle])
        assert dynamics["reference_url"] == policy["input"]["input_reference"][0]
        assert dynamics["prompt"] == policy["input"]["prompt"][0]
        assert "concatenated views" in dynamics["prompt"]
        assert policy["parameters"]["domain_name"] == "libero"
        if cycle:
            previous = Path(env_cfg.artifact_dir)
            # The downloaded clip is also the exact source of the next observation.
            decoded = read_video_frames(
                next(previous.rglob(f"cycle_{cycle - 1:03d}/prediction.mp4"))
            )
            np.testing.assert_array_equal(observations[cycle][0].numpy(), decoded[-1])
    assert len(list(Path(env_cfg.artifact_dir).rglob("actions_native.npy"))) == 3


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
    assert not service.jobs


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
    ],
)
def test_example_rejects_incompatible_configuration(worker_config, field, value):
    from examples.embodiment.eval_cosmos3_closed_loop import validate_closed_loop_cfg

    OmegaConf.update(worker_config, field, value)
    with pytest.raises(ValueError):
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


def test_env_failure_retains_actions_and_error(worker_config, serve):
    from rlinf.envs import get_env_cls

    service = FakeSGLang(mode="failed")
    cfg = worker_config.env.eval
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
    root = Path(cfg.artifact_dir)
    root.mkdir()
    sentinel = root / "existing.json"
    sentinel.write_text("previous run")
    with pytest.raises(FileExistsError):
        get_env_cls(cfg.env_type, cfg)(cfg=cfg, num_envs=1)
    assert sentinel.read_text() == "previous run"


def test_video_submission_failure_is_not_retried(tmp_path, serve):
    service = FakeSGLang(mode="submission_failed")
    client = Cosmos3Client(serve(service), timeout_s=5, poll_interval_s=0.001)
    try:
        with pytest.raises(RuntimeError, match="test submission failure"):
            client.generate_video(
                {"action_mode": "forward_dynamics"}, tmp_path / "prediction.mp4"
            )
    finally:
        client.close()
    assert [(method, path) for method, path, _ in service.requests] == [
        ("POST", "/v1/videos")
    ]
    assert not service.jobs
