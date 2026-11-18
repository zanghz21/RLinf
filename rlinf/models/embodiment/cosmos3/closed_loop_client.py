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

"""Raw Cosmos3 frames through SGLang's rollout HTTP API."""

from __future__ import annotations

import math
import time
from typing import Any
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from rlinf.models.embodiment.cosmos3.closed_loop_io import image_to_data_uri
from rlinf.utils.http_client import InferenceHTTPClient


def dynamics_request(
    prompt: str, image: np.ndarray | Image.Image, parameters: dict, actions: Any
) -> dict:
    """Build an inference-only request with lossless in-memory image input."""
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                f"Expected an RGB uint8 frame [H,W,3], got {image.shape}/{image.dtype}"
            )
        image = Image.fromarray(image)
    options = dict(parameters)
    seed = options.pop("seed")
    return {
        "prompt": prompt,
        "seed": seed,
        "image_path": [image_to_data_uri(image)],
        "rollout": False,
        "rollout_debug_mode": False,
        "extra_sampling_params": {
            **options,
            "action_mode": "forward_dynamics",
            "action": np.asarray(actions).tolist(),
            "save_output": False,
            "return_frames": False,
            "return_file_paths_only": False,
        },
    }


class Cosmos3Client:
    """Request raw frames from the same HTTP server used by the policy.

    Generation POSTs are never retried: an ambiguous timeout could otherwise
    enqueue duplicate work. Closing this client leaves the server running.
    """

    def __init__(self, base_url: str, timeout_s: float = 1800) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"Expected an HTTP server URL, got {base_url!r}")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Expected a finite positive request timeout")
        self.timeout_s = timeout_s
        self.http_client = InferenceHTTPClient(base_url)
        self._closed = False

    def close(self) -> None:
        """Release client connections without stopping the shared server."""
        self._closed = True
        self.http_client.close()

    def server_info(self) -> dict:
        """Return the model path and runtime identity advertised by the server."""
        if self._closed:
            raise RuntimeError("The Cosmos3 client is closed")
        with self.http_client.request(
            "GET", "/server_info", timeout_s=self.timeout_s
        ) as response:
            return response.json()

    def generate_frames(self, payload: dict) -> tuple[np.ndarray, dict]:
        """Decode raw HTTP tensors to uint8 RGB [T,H,W,3], without video codecs."""
        import msgpack
        from safetensors.numpy import load

        if self._closed:
            raise RuntimeError("The Cosmos3 client is closed")
        options = payload["extra_sampling_params"]
        if (
            payload.get("rollout") is not False
            or options.get("action_mode") != "forward_dynamics"
            or options.get("save_output") is not False
            or options.get("return_frames") is not False
            or options.get("return_file_paths_only") is not False
        ):
            raise ValueError(
                "Expected inference-only forward dynamics with file output disabled"
            )
        started = time.monotonic()
        with self.http_client.request(
            "POST",
            "/rollout/generate",
            json=payload,
            headers={"Accept": "application/msgpack"},
            timeout_s=self.timeout_s,
            max_retries=0,
        ) as response:
            results = msgpack.unpackb(response.content, raw=False)
        if not isinstance(results, list) or len(results) != 1:
            raise ValueError("Expected exactly one Cosmos3 rollout response")
        result = results[0]
        encoded = result.get("generated_output")
        if not isinstance(encoded, dict) or encoded.get("__tensor__") is not True:
            raise ValueError("Expected a serialized tensor in generated_output")
        raw = load(encoded["data"])["t"]
        expected = (
            int(options["num_frames"]),
            int(options["height"]),
            int(options["width"]),
            3,
        )
        if raw.shape == (3, *expected[:3]):
            frames = raw.transpose(1, 2, 3, 0)
        elif raw.shape == expected:
            frames = raw
        else:
            raise ValueError(
                f"Expected raw RGB tensor in CTHW or THWC layout for {expected}, got {raw.shape}"
            )
        if np.issubdtype(frames.dtype, np.floating) and np.isfinite(frames).all():
            # Match SGLang's decoded-frame conversion: clamp and truncate, not round.
            frames = np.clip(frames * 255, 0, 255).astype(np.uint8)
        elif frames.dtype != np.uint8:
            raise ValueError(
                f"Expected finite floating-point or uint8 RGB frames, got {frames.dtype}"
            )
        frames = np.ascontiguousarray(frames)
        return frames, {
            "request_id": result["request_id"],
            "elapsed_s": time.monotonic() - started,
            "generation_time_s": result.get("inference_time_s"),
            "raw_shape": list(raw.shape),
            "raw_dtype": str(raw.dtype),
            "frames_shape": list(frames.shape),
            "frames_dtype": str(frames.dtype),
        }
