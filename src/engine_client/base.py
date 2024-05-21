import aiohttp
import asyncio
import time
import requests
import sys
import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from enum import Enum
from transformers import PreTrainedTokenizer
from multiprocessing import Queue

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

class RequestConfig(BaseModel):
    """Request config."""

    stream: bool = False

"""
Engine client base class.
"""
class EngineClient(ABC):

    def __init__(
            self,
            pid: int,
            host: int,
            port: int,
            num_exp_process:int,
            use_token: bool,
            warmup_size: int,
            tokenizer: PreTrainedTokenizer,
            dataset: List[RequestData],
            request_config: RequestConfig
        ):
        self.base_uri = f"http://{host}:{port}"
        self.index = pid
        self.dataset = dataset
        self.use_token = use_token
        self.tokenizer = tokenizer
        self.request_config = request_config
        self.num_exp_process = num_exp_process
        self.warmup_size = warmup_size

    @property
    @abstractmethod
    def request_url(self) -> str:
        pass

    @property
    @abstractmethod
    def health_url(self) -> str:
        pass

    @abstractmethod
    def build_request(self, data: RequestData) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_completion_result(self, start, first_token_end_time, end, prompt_length, response_length, response_text) -> CompletionResult:
        pass

    def check_health(self) -> bool:
        refused_cnt = 0
        while True:
            try:
                response = requests.get(self.health_url)
                if response.status_code == 200:
                    break
                else:
                    raise Exception("Health check failed")
            except Exception:
                refused_cnt+=1
                if refused_cnt < 5:
                    print(f"Connection Refused: Retry {refused_cnt} times.")
                else:
                    print(f"Connection Refused: Falied {refused_cnt} times .")
                    sys.exit(1)
                time.sleep(30)

    async def send_request(self, session, data: RequestData, queue: Queue) -> CompletionResult:
        start = time.time()
        first_token_end_time = None
        while True:
            request_body = self.build_request(data)
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

        result = self.get_completion_result(start, first_token_end_time, end, prompt_length, response_length, response_text)
        queue.put_nowait(result)
        return result

    async def async_run(self, queue: Queue, request_rate_per_client: float) -> List[CompletionResult]:
        tasks = []
        connector = aiohttp.TCPConnector(
            limit_per_host=None, limit=None, keepalive_timeout=60  # type: ignore
        )
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=1800), connector=connector
        ) as session:
            for data in self.dataset[self.warmup_size + self.index::self.num_exp_process]:
                await asyncio.sleep(np.random.exponential(1.0 / request_rate_per_client))
                task = asyncio.create_task(self.send_request(session, data, queue))
                tasks.append(task)
            results = await asyncio.gather(*tasks)
        return results

    def wrapper_async_run(self, return_dict, queue, request_rate_per_client: float) -> None:
        return_dict[self.index] = asyncio.run(self.async_run(queue, request_rate_per_client))

    async def async_warm_up(self) -> None:
        tasks = []
        connector = aiohttp.TCPConnector(
            limit_per_host=None, limit=None, keepalive_timeout=60  # type: ignore
        )
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=1800), connector=connector
        ) as session:
            for data in self.dataset[:self.warmup_size]:
                task = asyncio.create_task(self.send_request(session, data, Queue()))
                tasks.append(task)
            await asyncio.gather(*tasks)

    def warm_up(self) -> None:
        asyncio.run(self.async_warm_up())
