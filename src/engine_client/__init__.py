from . import friendli
from . import vllm

import os
import yaml

from pathlib import Path
from engine_client.base import Engine, EngineClient, RequestConfig

MODEL_REGISTRY = {
    Engine.FRIENDLI : friendli.FriendliClient,
    Engine.VLLM : vllm.VllmClient,
}

REQUEST_CONFIG_REGISTRY = {
    Engine.FRIENDLI : friendli.FriendliRequestConfig,
    Engine.VLLM : vllm.VllmRequestConfig,
}

def get_engine_client_factory(model_name: Engine) -> EngineClient:
    return MODEL_REGISTRY[model_name]

def get_request_config(model_name: Engine, request_config_path: Path) -> RequestConfig:
    if os.path.exists(request_config_path):
        with open(request_config_path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
            try:
                return REQUEST_CONFIG_REGISTRY[model_name].parse_obj(config_dict)
            except Exception as e:
                raise ValueError(f"Error in parsing request config: {e}")
    return RequestConfig()
