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

"""Exercise the Cosmos3 runner against an HTTP service without model weights."""

import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from toolkits.cosmos3.closed_loop_client import Cosmos3Client, policy_request
from toolkits.cosmos3.closed_loop_io import (
    image_to_data_uri,
    read_video_frames,
    save_video,
)
from toolkits.cosmos3.run_closed_loop import run, run_cycles

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
                payload = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
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


def test_three_cycles_preserve_actions_and_feed_generated_pngs(
    tmp_path, serve, generation, videos
):
    service = FakeSGLang(videos)
    client = Cosmos3Client(serve(service), timeout_s=5, poll_interval_s=0.001)
    output = tmp_path / "run"
    output.mkdir()
    initial = Image.new("RGB", (16, 16), "blue")
    try:
        metrics = run_cycles(
            client, initial, "put the bowl on the plate", generation, 3, output
        )
    finally:
        client.close()

    posts = [
        (path, body) for method, path, body in service.requests if method == "POST"
    ]
    assert [path for path, _ in posts] == ["/v1/actions/generations", "/v1/videos"] * 3
    assert len(metrics) == 3
    for cycle in range(3):
        directory = output / f"cycle_{cycle:03d}"
        policy, dynamics = posts[2 * cycle][1], posts[2 * cycle + 1][1]
        assert policy["parameters"]["action_mode"] == "policy"
        assert policy["parameters"]["raw_action_dim"] == 10
        assert policy["input"]["prompt"].startswith("put the bowl on the plate.")
        assert "right half shows the wrist-mounted camera" in policy["input"]["prompt"]
        assert dynamics["action_mode"] == "forward_dynamics"
        assert dynamics["domain_name"] == "libero"
        assert dynamics["reference_url"] == policy["input"]["input_reference"]
        assert dynamics["action"] == service.policy_outputs[cycle]
        native = np.load(directory / "actions_native.npy")
        np.testing.assert_array_equal(
            native, np.asarray(dynamics["action"], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            native, np.load(directory / "actions_conditioning.npy")
        )
        uri_bytes = base64.b64decode(dynamics["reference_url"].split(",", 1)[1])
        assert uri_bytes == (directory / "input.png").read_bytes()
        if cycle:
            previous = output / f"cycle_{cycle - 1:03d}" / "next.png"
            assert (directory / "input.png").read_bytes() == previous.read_bytes()
            assert metrics[cycle]["input_sha256"] == metrics[cycle - 1]["next_sha256"]
    assert (output / "cycle_000/input.png").read_bytes() != (
        output / "cycle_001/input.png"
    ).read_bytes()
    assert read_video_frames(output / "rollout.mp4").shape == (13, 16, 16, 3)
    assert not [request for request in service.requests if request[0] == "DELETE"]


@pytest.mark.parametrize("values", [[[0.0] * 7] * 4, [["nan"] * 10] * 4])
def test_policy_rejects_malformed_or_nonfinite_actions(serve, generation, values):
    service = FakeSGLang()
    service.policy_values = values
    client = Cosmos3Client(serve(service), timeout_s=5, poll_interval_s=0.001)
    payload = policy_request(
        "move bowl", image_to_data_uri(Image.new("RGB", (16, 16))), generation
    )
    try:
        with pytest.raises(
            ValueError, match="Expected (action shape|finite model actions)"
        ):
            client.generate_actions(payload)
    finally:
        client.close()
    assert not service.jobs


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


def test_runner_rejects_video_with_wrong_shape(tmp_path, serve, generation, videos):
    service = FakeSGLang(videos)
    client = Cosmos3Client(serve(service), timeout_s=5, poll_interval_s=0.001)
    output = tmp_path / "wrong-shape"
    output.mkdir()
    try:
        with pytest.raises(ValueError, match="Expected generated video shape"):
            run_cycles(
                client,
                Image.new("RGB", (32, 16)),
                "move bowl",
                {**generation, "width": 32},
                1,
                output,
            )
    finally:
        client.close()
    assert (output / "cycle_000/prediction.mp4").exists()
    assert not (output / "cycle_000/next.png").exists()


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


def test_run_retains_failure_manifest_and_external_server(tmp_path, serve, run_config):
    service = FakeSGLang(mode="failed")
    service.model_path = str(Path(run_config["checkpoint_path"]).resolve())
    url = serve(service)
    output = tmp_path / "failed-run"
    with pytest.raises(RuntimeError, match="test generation failure"):
        run(run_config, output, url, "test-task-42")

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "test generation failure" in manifest["error"]
    assert manifest["task_id"] == "test-task-42"
    assert manifest["dataset"]["source_fps"] == 10
    assert manifest["server"]["ownership"] == "external"
    assert manifest["finished_at"]
    assert (output / "cycle_000/actions_native.npy").exists()
    client = Cosmos3Client(url)
    try:
        assert client.server_info()["model_path"] == service.model_path
    finally:
        client.close()


def test_run_refuses_to_overwrite_existing_output_directory(tmp_path, run_config):
    output = tmp_path / "existing-run"
    output.mkdir()
    sentinel = output / "manifest.json"
    sentinel.write_text("existing results")
    with pytest.raises(FileExistsError):
        run(run_config, output, None, None)
    assert sentinel.read_text() == "existing results"
    assert list(output.iterdir()) == [sentinel]
