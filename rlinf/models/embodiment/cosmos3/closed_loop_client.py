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

"""Cosmos3 video jobs using RLinf's shared inference HTTP transport."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

from rlinf.utils.http_client import InferenceHTTPClient


def dynamics_request(
    prompt: str, image_uri: str, parameters: dict, actions: Any, output_path: Path
) -> dict:
    """Build a visual request with the supplied native actions held as conditions."""
    return {
        **parameters,
        "prompt": prompt,
        "reference_url": image_uri,
        "action_mode": "forward_dynamics",
        "action": actions.tolist(),
        "output_path": str(output_path),
    }


class Cosmos3Client:
    """Manage Cosmos3 video jobs through the common inference HTTP client.

    Generation POSTs are never automatically retried: retrying an ambiguous
    timeout could enqueue a second generation with no record of the first.
    """

    def __init__(
        self, base_url: str, timeout_s: float = 1800, poll_interval_s: float = 1
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"Expected an HTTP server URL, got {base_url!r}")
        if timeout_s <= 0 or poll_interval_s <= 0:
            raise ValueError("Expected positive request timeout and polling interval")
        self.timeout_s = timeout_s
        self.poll_interval_s = poll_interval_s
        self.http_client = InferenceHTTPClient(base_url)

    def close(self) -> None:
        """Release client connections without stopping the shared model server."""
        self.http_client.close()

    def server_info(self) -> dict:
        """Return the model path and runtime identity advertised by the server."""
        with self.http_client.request(
            "GET", "/server_info", timeout_s=self.timeout_s
        ) as response:
            return response.json()

    def generate_video(self, payload: dict, destination: Path) -> dict:
        """Submit forward dynamics, await completion, and download its video.

        Returns:
            The submitted and completed job metadata, including any timing fields.
        """
        deadline = time.monotonic() + self.timeout_s
        with self.http_client.request(
            "POST", "/v1/videos", json=payload, timeout_s=self.timeout_s, max_retries=0
        ) as response:
            submitted = response.json()
        job_id = submitted.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError(f"Expected a video job ID, got {submitted!r}")
        path = "/v1/videos/" + quote(job_id, safe="")
        job = submitted
        try:
            while job.get("status") != "completed":
                status = job.get("status")
                if status in ("failed", "cancelled", "canceled"):
                    raise RuntimeError(f"Forward dynamics job {job_id} {status}: {job}")
                if status not in ("queued", "in_progress", "processing", "running"):
                    raise ValueError(f"Unexpected video job status: {job!r}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Forward dynamics job {job_id} exceeded {self.timeout_s}s"
                    )
                time.sleep(min(self.poll_interval_s, remaining))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Forward dynamics job {job_id} exceeded {self.timeout_s}s"
                    )
                with self.http_client.request(
                    "GET", path, timeout_s=remaining
                ) as response:
                    job = response.json()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Video download for {job_id} exceeded {self.timeout_s}s"
                )
            with self.http_client.request(
                "GET", path + "/content", stream=True, timeout_s=remaining
            ) as response:
                with destination.open("xb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                f"Video download for {job_id} exceeded {self.timeout_s}s"
                            )
                        output.write(chunk)
            if destination.stat().st_size == 0:
                raise ValueError(f"SGLang returned an empty video for {job_id}")
        except BaseException:
            # Delete this client's job; aborting GPU work depends on the server version.
            if job.get("status") != "completed":
                try:
                    self.http_client.request("DELETE", path, timeout_s=5).close()
                except Exception:
                    pass
            raise
        return {"submitted": submitted, "completed": job}
