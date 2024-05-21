from . import dummy
from . import huggingface

import yaml
from workload.base import RequestDataset

from transformers import PreTrainedTokenizer
from schemas import DummyWorkloadConfig, HfWorkloadConfig

REQUEST_DATASET_REGISTRY = {
    "dummy": dummy.DummyRequestDataset,
    "hf_cnn_dailymail": huggingface.CNNDailyMailRequestDataset,
    "hf_dolly": huggingface.DollyRequestDataset,
    "hf_alpaca": huggingface.AlpacaRequestDataset,
    "hf_sharegpt": huggingface.ShareGPTRequestDataset,
}

def get_workload(workload_config_path: str, dataset_size: int, tokenizer: PreTrainedTokenizer) -> RequestDataset:
    config_dict = {}
    with open(workload_config_path, encoding="utf-8") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
        try:
            type = config_dict["type"]
            config_dict["dataset_size"] = dataset_size
            if type == "dummy":
                workload_config = DummyWorkloadConfig(**config_dict)
            elif type.startswith("hf_"):
                workload_config = HfWorkloadConfig(**config_dict)
            else:
                raise RuntimeError(f"Invalid workload type: {type}")
        except Exception as e:
            raise RuntimeError(f"Invalid workload config: {e}")
    if workload_config.type not in REQUEST_DATASET_REGISTRY:
        raise RuntimeError(f"Workload type {workload_config.type} is not supported.")
    return REQUEST_DATASET_REGISTRY[workload_config.type](workload_config, tokenizer)
