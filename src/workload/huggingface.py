# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Huggingface dataset workload."""

import os
from abc import abstractmethod
from typing import Iterator, List, Tuple, cast

from datasets import Dataset, load_dataset
from schemas import HFDatasetConfig
from transformers import PreTrainedTokenizer

from workload.base import RequestData, RequestDataset


def safe_load_dataset(config: HFDatasetConfig) -> Dataset:
    """Load dataset."""
    data_path = config.path_or_name
    data_split = config.split

    try:
        if os.path.exists(data_path):
            dataset = load_dataset(
                config.format,
                data_files=data_path,
                split=data_split,
            )
        else:
            data_name_parts = data_path.split(":")
            if len(data_name_parts) == 1:
                dataset = load_dataset(data_path, split=data_split)
            elif len(data_name_parts) == 2:
                data_name, subset_name = data_name_parts
                dataset = load_dataset(data_name, subset_name, split=data_split)
            else:
                raise RuntimeError(
                    "Dataset name is in invalid format. "
                    "(valid format: '<dataset_name>' or '<dataset_name>:<subset_name>')"
                )
    except ValueError as err:
        raise RuntimeError(f"load_dataset failed. {str(err)}") from err

    if not isinstance(dataset, Dataset):
        raise RuntimeError("This dataset format is not supported.")

    return dataset


class HfRequestDataset(RequestDataset):
    """Huggingface request dataset."""

    @abstractmethod
    def iterate_dataset(self, dataset: Dataset) -> Iterator[Tuple[str, str]]:
        """Iterate dataset."""

    @property
    def warmup_size(self) -> int:
        """Warmup size."""
        return 1

    def process(self, tokenizer: PreTrainedTokenizer) -> List[RequestData]:
        """Process dataset to list of RequestData."""
        config = cast(HFDatasetConfig, self.workload_config.dataset_config)
        hf_dataset = safe_load_dataset(config)

        dataset: List[RequestData] = []
        for prompt, response in self.iterate_dataset(hf_dataset):
            prompt_token = tokenizer.encode(prompt)
            response_token = tokenizer.encode(response)
            seq_len = len(prompt_token) + len(response_token)
            if seq_len > config.max_length or seq_len < config.min_length:
                continue
            dataset.append(
                RequestData(
                    prompt=prompt,
                    prompt_tokens=prompt_token,
                    response=response,
                    response_tokens=response_token,
                )
            )
            if len(dataset) >= self.workload_config.dataset_size:
                break

        if len(dataset) < self.workload_config.dataset_size:
            raise RuntimeError(
                "Processed dataset is smaller than requested dataset size."
            )

        return dataset


### Custom RequestDataset for CNN_DailyMail
class CNNDailyMailRequestDataset(HfRequestDataset):
    """CNN_DailyMail request dataset."""

    def iterate_dataset(self, dataset: Dataset) -> Iterator[Tuple[str, str]]:
        """Iterate dataset."""
        for data in dataset:
            yield data["article"], data["highlights"]


### Custom RequestDataset for Dolly
class DollyRequestDataset(HfRequestDataset):
    """Dolly request dataset."""

    def iterate_dataset(self, dataset: Dataset) -> Iterator[Tuple[str, str]]:
        """Iterate dataset."""
        for data in dataset:
            yield data["instruction"], data["response"]


### Custom RequestDataset for Alpaca
class AlpacaRequestDataset(HfRequestDataset):
    """Alpaca request dataset."""

    def iterate_dataset(self, dataset: Dataset) -> Iterator[Tuple[str, str]]:
        """Iterate dataset."""
        for data in dataset:
            yield (
                data["instruction"] + " " + data["input"]
                if data["input"]
                else data["instruction"]
            ), data["output"]


### Custom RequestDataset for ShareGPT
class ShareGPTRequestDataset(HfRequestDataset):
    """ShareGPT request dataset."""

    def iterate_dataset(self, dataset: Dataset) -> Iterator[Tuple[str, str]]:
        """Iterate dataset."""
        for data in dataset:
            if len(data["conversations"]) >= 2:
                yield data["conversations"][0]["value"], data["conversations"][1][
                    "value"
                ]
