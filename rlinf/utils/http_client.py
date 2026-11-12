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


from __future__ import annotations

import os
import threading
import time
import weakref
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import aiohttp
import requests

_PROXY_ENV_VARS = (
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
)


@contextmanager
def no_proxy_env() -> Iterator[None]:
    """Temporarily unset HTTP proxy env vars.

    Removes ``http_proxy`` / ``https_proxy`` / ``all_proxy`` (both cases)
    from ``os.environ`` for the duration of the block, and restores the
    original values on exit. Any subprocess spawned inside the block
    inherits the stripped env, which is useful when the child's HTTP
    client (e.g. sglang-router's Rust reqwest, or sglang server's
    tokenizer-manager IPC) would otherwise tunnel intra-cluster traffic
    through a user-configured proxy.
    """
    saved = {var: os.environ.pop(var, None) for var in _PROXY_ENV_VARS}
    try:
        yield
    finally:
        for var, val in saved.items():
            if val is not None:
                os.environ[var] = val


class InferenceHTTPClient:
    """Thin HTTP client for an sglang router or server base URL.

    Sync methods reuse one requests session per calling thread; call :meth:`close`
    or use ``with`` after all requests finish. Async methods reuse an aiohttp
    session; :meth:`aclose` or ``async with`` releases both kinds of session.
    Keep each client in its worker process.


    Example::

        # sync
        with InferenceHTTPClient("http://router:30000") as client:
            out = client.generate(
                prompt="Hello", sampling_params={"max_new_tokens": 16}
            )

        # async
        async with InferenceHTTPClient("http://router:30000") as c:
            pending = {asyncio.create_task(c.async_generate(prompt=p)) for p in prompts}
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    handle(task.result())
    """

    def __init__(
        self,
        base_url: str,
        connect_timeout: float = 10.0,
        max_connections: int = 1024 * 16,
    ):
        self.base_url = base_url.rstrip("/")
        self.connect_timeout = connect_timeout
        self.max_connections = max_connections
        self._session: Optional[aiohttp.ClientSession] = None
        self._sync_sessions: weakref.WeakKeyDictionary[
            threading.Thread, tuple[requests.Session, weakref.finalize]
        ] = weakref.WeakKeyDictionary()
        self._sync_session_lock = threading.Lock()
        weakref.finalize(
            self,
            InferenceHTTPClient._close_sync_sessions,
            self._sync_sessions,
            self._sync_session_lock,
        )

    # ------------------------------------------------------------------
    # Sync API
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: Optional[str] = None,
        input_ids: Optional[list[int]] = None,
        sampling_params: Optional[dict] = None,
        return_logprob: bool = False,
    ) -> dict:
        return self.post(
            "/generate",
            self._generate_body(prompt, input_ids, sampling_params, return_logprob),
        )

    def chat_completion(
        self,
        messages: list[dict],
        model: str = "sglang-model",
        **kwargs: Any,
    ) -> dict:
        body = {"model": model, "messages": messages, **kwargs}
        return self.post("/v1/chat/completions", body)

    def health(self) -> bool:
        try:
            with self.request("GET", "/health", timeout_s=5) as response:
                return response.status_code == 200
        except RuntimeError:
            return False

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------
    async def async_generate(
        self,
        prompt: Optional[str] = None,
        input_ids: Optional[list[int]] = None,
        sampling_params: Optional[dict] = None,
        return_logprob: bool = False,
    ) -> dict:
        return await self._apost(
            "/generate",
            self._generate_body(prompt, input_ids, sampling_params, return_logprob),
        )

    async def async_chat_completion(
        self,
        messages: list[dict],
        model: str,
        **kwargs: Any,
    ) -> dict:
        body = {"model": model, "messages": messages, **kwargs}
        return await self._apost("/v1/chat/completions", body)

    async def async_health(self) -> bool:
        session = self._get_or_create_session()
        try:
            async with session.get(
                f"{self.base_url}/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                return resp.status == 200
        except (aiohttp.ClientError, TimeoutError):
            return False

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------
    def _get_or_create_sync_session(self) -> requests.Session:
        # Reward workers share a client across ephemeral thread pools. Weak thread
        # ownership releases those pools without sharing mutable Session state.
        thread = threading.current_thread()
        with self._sync_session_lock:
            if thread not in self._sync_sessions:
                session = requests.Session()
                cleanup = weakref.finalize(thread, session.close)
                self._sync_sessions[thread] = (session, cleanup)
            return self._sync_sessions[thread][0]

    @staticmethod
    def _close_sync_sessions(registry, lock) -> None:
        # The GC callback must not retain the client whose lifetime it watches.
        with lock:
            sessions = list(registry.values())
            registry.clear()
        for _, cleanup in sessions:
            cleanup()

    def close(self) -> None:
        """Close idle synchronous sessions; later requests can open new sessions."""
        self._close_sync_sessions(self._sync_sessions, self._sync_session_lock)

    def __enter__(self) -> "InferenceHTTPClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _get_or_create_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # limit         = total in-flight conns across all hosts
            # limit_per_host=0  → no separate per-host ceiling, so `limit` rules
            # enable_cleanup_closed helps reclaim sockets when peers hang up.
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=0,
                enable_cleanup_closed=True,
            )
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def aclose(self) -> None:
        """Close both HTTP sessions, if created."""
        self.close()
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def __aenter__(self) -> "InferenceHTTPClient":
        self._get_or_create_session()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_body(
        prompt: Optional[str],
        input_ids: Optional[list[int]],
        sampling_params: Optional[dict],
        return_logprob: bool,
    ) -> dict:
        body: dict = {"return_logprob": return_logprob}
        if prompt is not None:
            body["text"] = prompt
        if input_ids is not None:
            body["input_ids"] = input_ids
        if sampling_params is not None:
            body["sampling_params"] = sampling_params
        return body

    def request(
        self,
        method: str,
        path: str,
        *,
        timeout_s: Optional[float] = None,
        max_retries: int = 0,
        retry_backoff_s: float = 0.0,
        **kwargs: Any,
    ) -> requests.Response:
        """Send a synchronous request; the caller must close the returned response.

        ``kwargs`` accepts requests options such as ``json``, ``data``, ``headers``
        and ``stream``. ``timeout_s`` bounds each connect/read wait, not the full
        transfer; connect waits are also bounded by ``connect_timeout``. Consume
        streaming responses in the requesting thread before it exits. HTTP errors
        and connection failures raise RuntimeError. Retries are opt-in, including
        for POSTs whose repetition may create duplicate jobs. Environment proxies
        are bypassed without changing environment CA or netrc settings.
        """
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("timeout_s must be positive or None")
        method = method.upper()
        url = f"{self.base_url}{path}"
        retries = max(0, int(max_retries))
        connect_timeout = self.connect_timeout
        if timeout_s is not None:
            connect_timeout = min(connect_timeout, timeout_s)
        session = self._get_or_create_sync_session()
        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            is_last = attempt >= retries
            try:
                response = session.request(
                    method,
                    url,
                    timeout=(connect_timeout, timeout_s),
                    proxies={"http": "", "https": "", "all": ""},
                    **kwargs,
                )
            except requests.exceptions.RequestException as exc:
                last_error = exc
                if is_last:
                    raise RuntimeError(
                        f"{method} {url} failed after {retries + 1} attempt(s): {exc}"
                    ) from exc
                self._sleep_before_retry(attempt, retry_backoff_s)
                continue
            if response.status_code in {500, 502, 503, 504} and not is_last:
                try:
                    last_error = RuntimeError(
                        f"status={response.status_code}, body={response.text[:500]}"
                    )
                finally:
                    response.close()
                self._sleep_before_retry(attempt, retry_backoff_s)
                continue
            if not response.ok:
                try:
                    detail = (
                        f"status={response.status_code}, body={response.text[:500]}"
                    )
                    if last_error is not None:
                        detail += f" (after {retries + 1} attempt(s); prior error: {last_error})"
                    raise RuntimeError(f"{method} {url} failed: {detail}")
                finally:
                    response.close()
            return response

    def post(
        self,
        path: str,
        body: dict,
        *,
        msgpack: bool = False,
        timeout_s: Optional[float] = None,
        max_retries: int = 0,
        retry_backoff_s: float = 0.0,
    ) -> dict:
        """POST a body and decode JSON or the SGLang msgpack envelope.

        Request retries and timeouts follow :meth:`request`. Model-specific codec
        dependencies stay lazy so JSON callers do not need SGLang installed.
        """
        if msgpack:
            # SGLang renamed vla/ to action/; retain older fork compatibility.
            try:
                from sglang.multimodal_gen.runtime.entrypoints.action.protocol import (
                    pack_msgpack,
                    unpack_msgpack,
                )
            except ImportError:
                from sglang.multimodal_gen.runtime.entrypoints.vla.protocol import (
                    pack_msgpack,
                    unpack_msgpack,
                )
            ct = "application/msgpack"
            request_kwargs = {
                "data": pack_msgpack(self._to_msgpackable(body)),
                "headers": {"Content-Type": ct, "Accept": ct},
            }
        else:
            request_kwargs = {"json": body}
        with self.request(
            "POST",
            path,
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
            **request_kwargs,
        ) as response:
            if msgpack:
                if "msgpack" not in response.headers.get("content-type", "").lower():
                    raise RuntimeError(
                        "expected a msgpack response, got content-type="
                        f"{response.headers.get('content-type')!r}"
                    )
                return unpack_msgpack(response.content)
            return response.json()

    @staticmethod
    def _sleep_before_retry(attempt: int, retry_backoff_s: float) -> None:
        if retry_backoff_s > 0:
            time.sleep(retry_backoff_s * float(attempt + 1))

    @staticmethod
    def _to_msgpackable(value: Any) -> Any:
        """Recursively convert torch tensors / tianshou Batch to plain types."""
        import torch

        try:
            from tianshou.data import Batch

            if isinstance(value, Batch):
                value = value.__getstate__()
        except ImportError:
            pass
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        if isinstance(value, Mapping):
            return {
                str(k): InferenceHTTPClient._to_msgpackable(v) for k, v in value.items()
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [InferenceHTTPClient._to_msgpackable(v) for v in value]
        return value

    async def _apost(self, path: str, body: dict) -> dict:
        session = self._get_or_create_session()
        async with session.post(
            f"{self.base_url}{path}",
            json=body,
            timeout=aiohttp.ClientTimeout(
                total=None,
                sock_connect=self.connect_timeout,
                sock_read=None,
            ),
        ) as resp:
            resp.raise_for_status()
            return await resp.json()
