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

"""Exercise the common inference HTTP transport against a local HTTP service."""

import asyncio
import gc
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

from rlinf.utils.http_client import InferenceHTTPClient


@pytest.fixture
def service():
    class Service:
        calls = []
        failures = 0
        health_status = 200
        content = b"video-content" * 64

    state = Service()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            return

        def handle_request(self):
            size = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(size)) if size else None
            state.calls.append((self.command, self.path, body))
            status = 200
            content_type = "application/json"
            payload = {"method": self.command, "body": body}
            if self.path == "/health":
                status = state.health_status
            elif self.path == "/retry" and state.failures:
                state.failures -= 1
                status = 503
                payload = {"error": "temporarily unavailable"}
            elif self.path == "/stream":
                payload = state.content
                content_type = "application/octet-stream"
            data = (
                payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            )
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = handle_request
        do_POST = handle_request
        do_DELETE = handle_request

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_json_get_delete_and_streaming_use_common_transport(service):
    with InferenceHTTPClient(service.url) as client:
        assert client.generate(
            prompt="hello", sampling_params={"max_new_tokens": 1}
        ) == {
            "method": "POST",
            "body": {
                "text": "hello",
                "return_logprob": False,
                "sampling_params": {"max_new_tokens": 1},
            },
        }
        assert (
            client.chat_completion([{"role": "user", "content": "hi"}], model="test")[
                "body"
            ]["model"]
            == "test"
        )
        with client.request("GET", "/info") as response:
            assert response.json()["method"] == "GET"
        with client.request("GET", "/stream", stream=True, timeout_s=2) as response:
            assert b"".join(response.iter_content(chunk_size=17)) == service.content
        with client.request("DELETE", "/job") as response:
            assert response.json()["method"] == "DELETE"


def test_generation_is_not_retried_by_default(service):
    service.failures = 3
    with InferenceHTTPClient(service.url) as client:
        with pytest.raises(RuntimeError, match="temporarily unavailable"):
            client.post("/retry", {"prompt": "generate"})
    assert len(service.calls) == 1


def test_explicit_retries_close_responses_and_preserve_policy(service, monkeypatch):
    original = requests.Session.request
    closed = []

    def request(session, *args, **kwargs):
        response = original(session, *args, **kwargs)
        close = response.close

        def tracked_close():
            closed.append(response.status_code)
            close()

        response.close = tracked_close
        return response

    monkeypatch.setattr(requests.Session, "request", request)
    service.failures = 2
    with InferenceHTTPClient(service.url) as client:
        assert client.post("/retry", {}, max_retries=2)["method"] == "POST"
    assert len(service.calls) == 3
    assert closed == [503, 503, 200]


@pytest.mark.parametrize("status", [200, 201, 404, 503])
def test_health_boolean_contract(service, status):
    service.health_status = status
    with InferenceHTTPClient(service.url) as client:
        assert client.health() is (status == 200)
    assert len(service.calls) == 1


def test_connection_errors_are_reported_and_health_is_false(service, monkeypatch):
    attempts = []

    def fail_request(*args, **kwargs):
        attempts.append(args)
        raise requests.ConnectionError("test connection failure")

    monkeypatch.setattr(requests.Session, "request", fail_request)
    with InferenceHTTPClient(service.url) as client:
        assert client.health() is False
        with pytest.raises(RuntimeError, match="after 1 attempt"):
            client.post("/generate", {})
    assert len(attempts) == 2


@pytest.mark.parametrize(
    "timeout,expected", [(0.5, (0.5, 0.5)), (20, (10.0, 20)), (None, (10.0, None))]
)
def test_connect_wait_is_bounded_by_remaining_timeout(
    service, monkeypatch, timeout, expected
):
    original = requests.Session.request
    observed = []

    def request(session, *args, **kwargs):
        observed.append(kwargs["timeout"])
        return original(session, *args, **kwargs)

    monkeypatch.setattr(requests.Session, "request", request)
    with InferenceHTTPClient(service.url) as client:
        with client.request("GET", "/info", timeout_s=timeout):
            pass
    assert observed == [expected]


def test_environment_proxies_are_bypassed(service, monkeypatch):
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.setenv(key, "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    with InferenceHTTPClient(service.url) as client:
        assert client.post("/generate", {}, timeout_s=2)["method"] == "POST"
        with client.request("GET", "/stream", timeout_s=2, stream=True) as response:
            assert response.content == service.content


def test_close_and_aclose_release_connections_and_allow_reopening(service, monkeypatch):
    original = requests.Session.close
    closed = []

    def close(session):
        closed.append(session)
        original(session)

    monkeypatch.setattr(requests.Session, "close", close)
    client = InferenceHTTPClient(service.url)
    client.post("/generate", {})
    client.close()
    client.close()
    assert len(closed) == 1
    client.post("/generate", {})

    async def check_async():
        async with client:
            assert await client.async_generate(prompt="hello") == {
                "method": "POST",
                "body": {"text": "hello", "return_logprob": False},
            }
            assert await client.async_health() is True
        await client.aclose()

    asyncio.run(check_async())
    assert len(closed) == 2
    assert closed[0] is not closed[1]


def test_concurrent_callers_have_separate_reusable_sessions(service, monkeypatch):
    original = requests.Session.request
    barrier = threading.Barrier(2, timeout=5)
    sessions = {}

    def request(session, *args, **kwargs):
        sessions.setdefault(threading.get_ident(), []).append(session)
        barrier.wait()
        return original(session, *args, **kwargs)

    monkeypatch.setattr(requests.Session, "request", request)
    with InferenceHTTPClient(service.url) as client:

        def generate(index):
            return [client.post("/generate", {"index": index}) for _ in range(2)]

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(generate, [0, 1]))
    assert len(sessions) == 2
    assert len({id(calls[0]) for calls in sessions.values()}) == 2
    assert all(calls[0] is calls[1] for calls in sessions.values())
    assert [[response["body"]["index"] for response in batch] for batch in results] == [
        [0, 0],
        [1, 1],
    ]


def test_finished_threads_release_their_connections(service, monkeypatch):
    original = requests.Session.close
    closed = []
    results = []

    def close(session):
        closed.append(session)
        original(session)

    monkeypatch.setattr(requests.Session, "close", close)
    client = InferenceHTTPClient(service.url)
    thread = threading.Thread(
        target=lambda: results.append(client.post("/generate", {}))
    )
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert len(results) == 1
    del thread
    gc.collect()
    assert len(closed) == 1
    client.close()
    assert len(closed) == 1


def test_discarded_client_releases_connections_on_a_live_thread(service, monkeypatch):
    original = requests.Session.close
    closed = []

    def close(session):
        closed.append(session)
        original(session)

    monkeypatch.setattr(requests.Session, "close", close)
    client = InferenceHTTPClient(service.url)
    client.post("/generate", {})
    del client
    gc.collect()
    assert len(closed) == 1
