# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

# pylint: disable=duplicate-code

"""VLLM engine client."""

import json
from typing import Any, Dict

from workload.base import RequestData

from engine_client.base import CompletionResult, EngineClient, RequestConfig


class VllmRequestConfig(RequestConfig):  # pylint: disable=too-few-public-methods
    """Request config for the Vllm Engine."""

    name: str


class VllmClient(EngineClient):
    """Client for the Vllm Engine."""

    @property
    def request_url(self) -> str:
        return f"{self.base_uri}/v1/completions"

    @property
    def health_url(self) -> str:
        return f"{self.base_uri}/v1/models"

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
            response_length = 1  # Add first token
            for line in response_text.split("\n"):  # SSE is a Line-based protocol
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    if line[len("data: ") :] == "[DONE]":
                        break
                    response_length += 1
        else:
            response_json = json.loads(response_text)
            response_length = response_json["usage"]["completion_tokens"]
        return CompletionResult(
            start_time=start,
            first_token_end_time=first_token_end_time,
            end_time=end,
            prompt_length=prompt_length,
            response_length=response_length,
        )

    def build_http_request(self, data: RequestData) -> Dict[str, Any]:
        assert isinstance(self.request_config, VllmRequestConfig)
        request_body = {
            "prompt": data.prompt_tokens if self.use_token else data.prompt,
            "max_tokens": len(data.response_tokens),
            # to ensure that the lenght of the response is always response_length
            "ignore_eos": True,
            "top_k": 1,
            "n": 1,
            "stream": self.request_config.stream,
            "model": self.request_config.name,
        }
        return request_body
