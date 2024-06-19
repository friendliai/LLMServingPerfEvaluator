# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Engine client base model."""

import asyncio
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Queue
from typing import Any, Dict, List, Optional

import aiohttp
import grpc
import numpy as np
import requests
from grpc_health.v1 import health_pb2, health_pb2_grpc
from pydantic import BaseModel
from transformers import PreTrainedTokenizer
from workload.base import RequestData


class Engine(str, Enum):
    """Engine enum class."""

    FRIENDLI = "friendli"
    VLLM = "vllm"


@dataclass
class CompletionResult:
    """Completion result."""

    start_time: float
    first_token_end_time: Optional[float]
    end_time: float
    prompt_length: int
    response_length: int


class RequestConfig(BaseModel):  # pylint: disable=too-few-public-methods
    """Request config."""

    stream: bool = False


class ClientType(str, Enum):
    """Client type enum class."""

    HTTP = "http"
    GRPC = "grpc"


class EngineClient(ABC):  # pylint: disable=too-many-instance-attributes
    """Engine client base class."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        pid: int,
        host: int,
        port: int,
        num_exp_process: int,
        use_token: bool,
        warmup_size: int,
        tokenizer: PreTrainedTokenizer,
        dataset: List[RequestData],
        request_config: RequestConfig,
        client_type: ClientType = ClientType.HTTP,
    ):
        """Initialize EngineClient."""
        self.base_uri = (
            f"http://{host}:{port}"
            if client_type == ClientType.HTTP
            else f"{host}:{port}"
        )
        self.index = pid
        self.dataset = dataset
        self.use_token = use_token
        self.tokenizer = tokenizer
        self.request_config = request_config
        self.num_exp_process = num_exp_process
        self.warmup_size = warmup_size
        self.client_type = client_type

    @property
    @abstractmethod
    def request_url(self) -> str:
        """Request url."""

    @property
    @abstractmethod
    def health_url(self) -> str:
        """Health url."""

    @abstractmethod
    def build_http_request(self, data: RequestData) -> Dict[str, Any]:
        """Build http request body."""

    def build_grpc_request(self, data: RequestData):
        """Build grpc request."""
        raise NotImplementedError()

    @abstractmethod
    def get_completion_result(  # pylint: disable=too-many-arguments
        self,
        start,
        first_token_end_time,
        end,
        prompt_length,
        response_length,
        response_text,
    ) -> CompletionResult:
        """Get completion result."""

    def check_health(self) -> None:
        """Health check."""
        if self.client_type == ClientType.HTTP:
            self.check_http_health()
        else:
            self.check_grpc_health()

    def check_http_health(self) -> None:
        """Check http server health."""
        refused_cnt = 0
        while True:
            try:
                response = requests.get(  # pylint: disable=missing-timeout
                    self.health_url
                )
                if response.status_code == 200:
                    break
                raise Exception(  # pylint: disable=broad-exception-raised
                    "Health check failed"
                )
            except Exception:  # pylint: disable=broad-exception-caught
                refused_cnt += 1
                if refused_cnt < 5:
                    print(f"Connection Refused: Retry {refused_cnt} times.")
                else:
                    print(f"Connection Refused: Falied {refused_cnt} times .")
                    sys.exit(1)
                time.sleep(30)

    def check_grpc_health(self) -> None:
        """Check grpc server health."""
        refused_cnt = 0
        while True:
            try:
                with grpc.insecure_channel(self.base_uri) as channel:
                    health_stub = health_pb2_grpc.HealthStub(channel)
                    health_request = health_pb2.HealthCheckRequest(service="")
                    health_resp = health_stub.Check(health_request)
                    if health_resp.status == health_pb2.HealthCheckResponse.SERVING:
                        break
                    raise Exception(  # pylint: disable=broad-exception-raised
                        "Health check failed"
                    )
            except Exception:  # pylint: disable=broad-exception-caught
                refused_cnt += 1
                if refused_cnt < 5:
                    print(f"Connection Refused: Retry {refused_cnt} times.")
                else:
                    print(f"Connection Refused: Falied {refused_cnt} times .")
                    sys.exit(1)
                time.sleep(10)

    async def send_http_request(
        self, session, data: RequestData, queue: Queue
    ) -> CompletionResult:
        """Send http request."""
        start = time.time()
        first_token_end_time = None
        while True:
            request_body = self.build_http_request(data)
            async with session.post(self.request_url, json=request_body) as response:
                if self.request_config.stream:
                    async for _ in response.content:
                        first_token_end_time = time.time()  # Get first token time
                        break
                response_text = await response.text()
                if response.status == 200:
                    # succeed; break the loop
                    break
                # failed; will retry
                print("Failed:", response.status, response_text, request_body)
        end = time.time()

        prompt_length = len(data.prompt_tokens)
        response_length = len(data.response_tokens)

        result = self.get_completion_result(
            start,
            first_token_end_time,
            end,
            prompt_length,
            response_length,
            response_text,
        )
        queue.put_nowait(result)
        return result

    async def send_grpc_request(
        self, stub, data: RequestData, queue: Queue
    ) -> CompletionResult:
        """Send grpc request."""
        raise NotImplementedError()

    async def async_http_run(
        self, queue: Queue, request_rate_per_client: float
    ) -> List[CompletionResult]:
        """Async http request run."""
        tasks = []
        connector = aiohttp.TCPConnector(
            limit_per_host=None, limit=None, keepalive_timeout=60  # type: ignore
        )
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=1800), connector=connector
        ) as session:
            for data in self.dataset[
                self.warmup_size + self.index :: self.num_exp_process
            ]:
                await asyncio.sleep(
                    np.random.exponential(1.0 / request_rate_per_client)
                )
                task = asyncio.create_task(self.send_http_request(session, data, queue))
                tasks.append(task)
            results = await asyncio.gather(*tasks)
        return results

    async def async_grpc_run(
        self, queue: Queue, request_rate_per_client: float
    ) -> List[CompletionResult]:
        """Async grpc run."""
        raise NotImplementedError()

    def wrapper_async_run(
        self, return_dict, queue, request_rate_per_client: float
    ) -> None:
        """Wrapper of async run."""
        return_dict[self.index] = asyncio.run(
            self.async_http_run(queue, request_rate_per_client)
            if self.client_type == ClientType.HTTP
            else self.async_grpc_run(queue, request_rate_per_client)
        )

    async def async_http_warm_up(self) -> None:
        """Async warm up."""
        tasks = []
        connector = aiohttp.TCPConnector(
            limit_per_host=None, limit=None, keepalive_timeout=60  # type: ignore
        )
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=1800), connector=connector
        ) as session:
            for data in self.dataset[: self.warmup_size]:
                task = asyncio.create_task(
                    self.send_http_request(session, data, Queue())
                )
                tasks.append(task)
            await asyncio.gather(*tasks)

    async def async_grpc_warm_up(self) -> None:
        """Async grpc warm up."""
        raise NotImplementedError()

    def warm_up(self) -> None:
        """Wrapper of Warm up."""
        asyncio.run(
            self.async_http_warm_up()
            if self.client_type == ClientType.HTTP
            else self.async_grpc_warm_up()
        )
