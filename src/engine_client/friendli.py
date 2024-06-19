# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

# pylint: disable=duplicate-code

"""Friendli engine client."""

import asyncio
import json
import time
from multiprocessing import Queue
from typing import Any, Dict, List

import numpy as np

from engine_client.base import (
    CompletionResult,
    EngineClient,
    RequestConfig,
    RequestData,
)
from friendli import AsyncFriendli

# Request config for the Friendli Engine.
FriendliRequestConfig = RequestConfig


class FriendliHttpBaseClient(EngineClient):
    """Base Http Client for the Friendli Engine."""

    @property
    def request_url(self) -> str:
        return f"{self.base_uri}/v1/completions"

    @property
    def health_url(self) -> str:
        return f"{self.base_uri}/health"

    def get_completion_result(  # pylint: disable=too-many-arguments
        self,
        start,
        first_token_end_time,
        end,
        prompt_length,
        response_length,
        response_text,
    ) -> CompletionResult:
        if self.request_config.stream:
            for line in response_text.split("\n"):  # SSE is a Line-based protocol
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    data = json.loads(line[len("data:") :])
                    if data["event"] == "complete":
                        response_length = data["usage"][
                            "completion_tokens"
                        ]  # Add first token
        else:
            response_json = json.loads(response_text)
            response_length = len(response_json["choices"][0]["tokens"])
        return CompletionResult(
            start_time=start,
            first_token_end_time=first_token_end_time,
            end_time=end,
            prompt_length=prompt_length,
            response_length=response_length,
        )

    def build_http_request(self, data: RequestData) -> Dict[str, Any]:
        assert isinstance(self.request_config, FriendliRequestConfig)
        request_body: Dict[str, Any] = {
            "max_tokens": len(data.response_tokens),
            "min_tokens": len(data.response_tokens),
            "top_k": 1,
            "stream": self.request_config.stream,
        }
        prompt_or_tokens_key = "tokens" if self.use_token else "prompt"
        request_body[prompt_or_tokens_key] = (
            data.prompt_tokens if self.use_token else data.prompt
        )
        return request_body


class FriendliClient(FriendliHttpBaseClient):
    """Friendli Client for the Friendli Engine."""

    async def _send_request(  # pylint: disable=too-many-locals
        self, data: RequestData, queue: Queue, use_grpc: bool
    ) -> CompletionResult:
        """Send request."""
        first_token_end_time = None
        client = AsyncFriendli(base_url=self.base_uri, use_grpc=use_grpc)

        start = time.time()
        while True:
            try:
                request_body = self.build_http_request(data)
                response = await client.completions.create(**request_body)
                if self.request_config.stream:
                    response_text = ""
                    response_length = 0
                    async for chunk in response:
                        if not first_token_end_time:
                            first_token_end_time = time.time()  # Get first token time
                        response_text += chunk.text
                        response_length += 1
                else:
                    assert not use_grpc  # gRPC only support streaming mode
                    response_text = response.choices[0].text
                    response_length = len(response.choices[0].tokens)
                break  # succeed; break the loop
            except Exception as e:  # pylint: disable=broad-exception-caught
                # failed; will retry
                print(f"Failed, retry: {str(e)}")
        end = time.time()

        prompt_length = len(data.prompt_tokens)

        result = (
            self.get_completion_result(
                start,
                first_token_end_time,
                end,
                prompt_length,
                response_length,
                response_text,
            )
            if self.request_config.stream
            else CompletionResult(
                start_time=start,
                first_token_end_time=first_token_end_time,
                end_time=end,
                prompt_length=prompt_length,
                response_length=response_length,
            )
        )

        queue.put_nowait(result)
        return result

    async def _async_warm_up(self, use_grpc: bool) -> None:
        tasks = []
        send_request_fn = self.send_grpc_request if use_grpc else self.send_http_request

        for data in self.dataset[: self.warmup_size]:
            task = asyncio.create_task(send_request_fn(None, data, Queue()))
            tasks.append((task))
        await asyncio.gather(*tasks)

    async def _async_run(
        self, queue: Queue, request_rate_per_client: float, use_grpc: bool
    ) -> List[CompletionResult]:
        """Async request run."""
        tasks = []
        send_request_fn = self.send_grpc_request if use_grpc else self.send_http_request

        for data in self.dataset[self.warmup_size + self.index :: self.num_exp_process]:
            await asyncio.sleep(np.random.exponential(1.0 / request_rate_per_client))
            task = asyncio.create_task(send_request_fn(None, data, queue))
            tasks.append(task)
        results = await asyncio.gather(*tasks)
        return results

    async def send_http_request(
        self, session, data: RequestData, queue: Queue
    ) -> CompletionResult:
        return await self._send_request(data, queue, False)

    async def send_grpc_request(
        self, stub, data: RequestData, queue: Queue
    ) -> CompletionResult:
        return await self._send_request(data, queue, True)

    async def async_http_run(
        self, queue: Queue, request_rate_per_client: float
    ) -> List[CompletionResult]:
        return await self._async_run(queue, request_rate_per_client, False)

    async def async_grpc_run(
        self, queue: Queue, request_rate_per_client: float
    ) -> List[CompletionResult]:
        return await self._async_run(queue, request_rate_per_client, True)

    async def async_http_warm_up(self) -> None:
        await self._async_warm_up(False)

    async def async_grpc_warm_up(self) -> None:
        await self._async_warm_up(True)
