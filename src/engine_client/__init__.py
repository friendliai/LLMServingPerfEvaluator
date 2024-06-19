# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Initialize engine client module."""

import os
from pathlib import Path

import yaml

from engine_client.base import Engine, EngineClient, RequestConfig

from . import friendli, vllm

MODEL_REGISTRY = {
    Engine.FRIENDLI: friendli.FriendliClient,
    Engine.VLLM: vllm.VllmClient,
}

REQUEST_CONFIG_REGISTRY = {
    Engine.FRIENDLI: friendli.FriendliRequestConfig,
    Engine.VLLM: vllm.VllmRequestConfig,
}


def get_engine_client_factory(model_name: Engine) -> EngineClient:
    """Get engine client factory."""
    return MODEL_REGISTRY[model_name]


def get_request_config(model_name: Engine, request_config_path: Path) -> RequestConfig:
    """Get engine request config."""
    if os.path.exists(request_config_path):
        with open(request_config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
            try:
                return REQUEST_CONFIG_REGISTRY[model_name].parse_obj(config_dict)
            except Exception as e:
                raise ValueError(f"Error in parsing request config: {e}") from e
    return RequestConfig()
