import json

from typing import Dict, Any

from engine_client.base import EngineClient, RequestData, RequestConfig, CompletionResult


class FriendliRequestConfig(RequestConfig):
    """Request config for the Friendli Engine."""

class FriendliClient(EngineClient):
    """Client for the Friendli Engine."""

    @property
    def request_url(self) -> str:
        return f"{self.base_uri}/v1/completions"

    @property
    def health_url(self) -> str:
        return f"{self.base_uri}/health"

    def get_completion_result(self, start, first_token_end_time, end, prompt_length, response_length, response_text) -> CompletionResult:
        if self.request_config.stream:
            for line in response_text.split("\n"):  # SSE is a Line-based protocol
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    data = json.loads(line[len("data:") :])
                    if data["event"] == "complete":
                        response_length =  data["usage"]["completion_tokens"]  # Add first token
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

    def build_request(self,data: RequestData) -> Dict[str, Any]:
        assert isinstance(self.request_config, FriendliRequestConfig)
        request_body: Dict[str, Any] = {
            "max_tokens": len(data.response_tokens),
            "min_tokens": len(data.response_tokens),
            "top_k": 1,
            "stream": self.request_config.stream,
            "no_persist": False,
        }
        prompt_or_tokens_key = "tokens" if self.use_token else "prompt"
        request_body[prompt_or_tokens_key] = data.prompt_tokens if self.use_token else data.prompt
        return request_body
