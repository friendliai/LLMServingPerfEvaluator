# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Dummy workload."""

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, cast

from enums import Distribution
from schemas import DummyDatasetConfig
from transformers import PreTrainedTokenizer

from workload.base import RequestData, RequestDataset


class DummySystemPromptSampler:
    """Dummy systemprompt sampler class."""

    def __init__(self, name: str, size: int, vocab_size: int):
        """Initialize DummySystemPromptSampler."""
        assert size > 0
        assert vocab_size > 0
        self.name = name
        self.size = size
        self.vocab_size = vocab_size
        self.system_prompt = [
            random.randint(0, self.vocab_size - 1) for _ in range(self.size)
        ]

    def get_system_prompt(self) -> List[int]:
        """Get systemprompt from DummySystemPromptSampler."""
        return self.system_prompt

    def __repr__(self):
        """Overwrite DummySystemPromptSampler repr."""
        return (
            f"DummySystemPromptSampler(name: {self.name}"
            f", size: {self.size}, vocab_size: {self.vocab_size})"
        )


class DummySystemPromptSamplerMixture:
    """Dummy systemprompt sampler mixture."""

    def __init__(
        self,
        system_prompt_samplers: List[DummySystemPromptSampler],
        input_system_prompt_distribution,  # name: weight
    ):
        """Initialize DummySystemPromptSamplerMixture."""
        self.system_prompt_samplers = []
        self.weight_list = []
        for system_prompt_sampler in system_prompt_samplers:
            if system_prompt_sampler.name in input_system_prompt_distribution.keys():
                self.system_prompt_samplers.append(system_prompt_sampler)
                self.weight_list.append(
                    input_system_prompt_distribution[system_prompt_sampler.name]
                )

    def get_system_prompt(self) -> List[int]:
        """Get systemprompt."""
        system_prompt_sampler = random.choices(
            self.system_prompt_samplers, self.weight_list
        )[0]
        return system_prompt_sampler.get_system_prompt()

    def __repr__(self):
        return (
            "DummySystemPromptSamplerMixture(system_prompt_samplers: "
            f"{self.system_prompt_samplers}, weight_list: {self.weight_list})"
        )


class DummyDatasetBaseSampler(ABC):
    """Dummy dataset base sampler."""

    def __init__(self, minimum: int, maximum: int, weight: int = 1):
        assert minimum <= maximum
        assert weight >= 1
        self.min = minimum
        self.max = maximum
        self.weight = weight

    @abstractmethod
    def get_value(self) -> int:
        """Get value interface function."""

    @abstractmethod
    def __repr__(self) -> str:
        pass


class DummyDatasetUniformSampler(DummyDatasetBaseSampler):
    """Dummy dataset uniform distribution sampler."""

    def get_value(self) -> int:
        """Get value from sampler."""
        return random.randint(self.min, self.max)

    def __repr__(self) -> str:
        return (
            f"DummyDatasetUniformSampler("
            f"min: {self.min}, max: {self.max}, weight: {self.weight})"
        )


class DummyDatasetNormalSampler(DummyDatasetBaseSampler):
    """Dummy dataset normal distribution sampler."""

    def __init__(  # pylint: disable=too-many-arguments
        self, minimum: int, maximum: int, mean: float, std: float, weight: int = 1
    ):
        """Initialize DummyDatasetNormalSampler."""
        super(DummyDatasetNormalSampler, self).__init__(minimum, maximum, weight)
        self.mean = mean
        self.std = std

    def get_value(self) -> int:
        """Get value from sampler."""
        for _ in range(1000):
            value = round(random.normalvariate(self.mean, self.std))
            if self.min <= value <= self.max:
                return value
        return random.randint(self.min, self.max)

    def __repr__(self) -> str:
        return (
            f"DummyDatasetNormalSampler"
            f"(min: {self.min}, max: {self.max}, mean: {self.mean}"
            f", std: {self.std}, weight: {self.weight})"
        )


class DummyDatasetSamplerMixture:
    """Dummy dataset sampler mixture class."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        input_samplers: List[DummyDatasetBaseSampler],
        output_samplers: List[DummyDatasetBaseSampler],
        input_system_prompt_distribution,
        system_prompt_samplers,
        weight: int = 1,
    ):
        """Initialize DummyDatasetSamplerMixture."""
        self.input_samplers = input_samplers
        self.output_samplers = output_samplers
        self.input_system_prompt_distribution = input_system_prompt_distribution
        self.weight = weight

        self.enable_system_prompt = (input_system_prompt_distribution is not None) and (
            system_prompt_samplers is not None
        )
        self.system_prompt_sampler_mixture = (
            DummySystemPromptSamplerMixture(
                system_prompt_samplers, input_system_prompt_distribution
            )
            if self.enable_system_prompt
            else None
        )

    def get_input_len(self) -> int:
        """Get input length."""
        input_sampler = random.choices(
            self.input_samplers, [s.weight for s in self.input_samplers]
        )[0]
        return input_sampler.get_value()

    def get_output_len(self) -> int:
        """Get output length."""
        output_sampler = random.choices(
            self.output_samplers, [s.weight for s in self.output_samplers]
        )[0]
        return output_sampler.get_value()

    def get_input_system_prompt_distribution(self) -> int:
        """Get input systemprompt distribution."""
        return self.input_system_prompt_distribution

    def get_system_prompt(self):
        """Get systemprompt."""
        if self.enable_system_prompt:
            return self.system_prompt_sampler_mixture.get_system_prompt()
        return None

    def __repr__(self):
        return (
            "DummyDatasetSamplerMixture("
            f"input_samplers: {self.input_samplers}"
            f", output_samplers: {self.output_samplers}"
            f", system_prompt_sampler_mixture: {self.system_prompt_sampler_mixture}"
            f", weight: {self.weight})"
        )


def get_input_output_len_with_system_prompt(
    sampler_mixture_list: List[DummyDatasetSamplerMixture],
):
    """Get input output length with system prompt."""
    sampler_mixture = random.choices(
        sampler_mixture_list, [s.weight for s in sampler_mixture_list]
    )[0]
    return (
        sampler_mixture.get_input_len(),
        sampler_mixture.get_output_len(),
        sampler_mixture.get_system_prompt(),
    )


def get_system_prompt_sampler(
    config: Dict, vocab_size: int
) -> DummySystemPromptSampler:
    """Get system prompt sampler."""
    return DummySystemPromptSampler(config["name"], config["size"], vocab_size)


def get_system_prompt_sampler_mixture(
    system_prompt_samplers: List[DummySystemPromptSampler],
    input_system_prompt_distribution: Dict[str, int],  # name: weight
) -> DummySystemPromptSamplerMixture:
    """Get system prompt sampler"""
    return DummySystemPromptSamplerMixture(
        system_prompt_samplers, input_system_prompt_distribution
    )


def get_dataset_sampler(config: Dict) -> DummyDatasetBaseSampler:
    """Get dataset sampler."""
    sampler_type = config["type"]
    weight = 1 if "weight" not in config else config["weight"]
    if sampler_type == Distribution.NORMAL:
        return DummyDatasetNormalSampler(
            config["min"],
            config["max"],
            config["mean"],
            config["std"],
            weight,
        )
    if sampler_type == Distribution.UNIFORM:
        return DummyDatasetUniformSampler(
            config["min"],
            config["max"],
            weight,
        )
    raise RuntimeError("Not supported distribution type")


def get_dataset_sampler_mixture(
    config: Dict, system_prompt_samplers: Optional[List[DummySystemPromptSampler]]
) -> DummyDatasetSamplerMixture:
    """Get dataset sampler mixture from config."""
    input_samplers = [
        get_dataset_sampler(sampler_config) for sampler_config in config["input"]
    ]
    output_samplers = [
        get_dataset_sampler(sampler_config) for sampler_config in config["output"]
    ]
    weight = 1 if "weight" not in config else config["weight"]
    input_system_prompt_distribution = (
        {
            system_prompt["name"]: system_prompt["weight"]
            for system_prompt in config["system_prompt"]
        }
        if system_prompt_samplers and "system_prompt" in config
        else None
    )
    return DummyDatasetSamplerMixture(
        input_samplers,
        output_samplers,
        input_system_prompt_distribution,
        system_prompt_samplers,
        weight,
    )


class DummyRequestDataset(RequestDataset):
    """Dummy request dataset."""

    @property
    def warmup_size(self) -> int:
        """Get warmup size."""
        return (
            len(self.system_prompt_sampler_list)
            if self.system_prompt_sampler_list
            else 0
        )

    @property
    def system_prompt_sampler_list(self) -> Optional[List[DummyDatasetSamplerMixture]]:
        """Get sampler list."""
        config = cast(DummyDatasetConfig, self.workload_config.dataset_config)
        system_prompt_sampler_list = []
        if config.system_prompt_config:
            for sp_config in config.system_prompt_config:
                system_prompt_sampler_list.append(
                    get_system_prompt_sampler(sp_config.model_dump(), config.vocab_size)
                )
            return system_prompt_sampler_list
        return None

    @property
    def dataset_sampler_list(self) -> List[DummyDatasetSamplerMixture]:
        """Get dataset sampler list."""
        config = cast(DummyDatasetConfig, self.workload_config.dataset_config)
        dataset_sampler_list = []
        for data in config.dataset:
            dataset_sampler_list.append(
                get_dataset_sampler_mixture(
                    data.model_dump(), self.system_prompt_sampler_list
                )
            )
        return dataset_sampler_list

    def process(self, tokenizer: PreTrainedTokenizer) -> List[RequestData]:
        """Process dataset to list of RequestData."""
        dataset = []
        config = cast(DummyDatasetConfig, self.workload_config.dataset_config)

        # Warm up all the system prompt
        if self.system_prompt_sampler_list:
            for prompt_sampler in self.system_prompt_sampler_list:
                input_len, output_len, _ = get_input_output_len_with_system_prompt(
                    self.dataset_sampler_list
                )
                system_prompt = prompt_sampler.get_system_prompt()
                input_tokens = system_prompt + [
                    random.randint(0, config.vocab_size - 1) for _ in range(input_len)
                ]
                output_tokens = [
                    random.randint(0, config.vocab_size - 1) for _ in range(output_len)
                ]
                input_prompt = tokenizer.decode(input_tokens, add_special_tokens=False)
                output_prompt = tokenizer.decode(
                    output_tokens, add_special_tokens=False
                )
                dataset.append(
                    RequestData(
                        input_prompt, input_tokens, output_prompt, output_tokens
                    )
                )

        for _ in range(self.workload_config.dataset_size):
            input_len, output_len, system_prompt = (
                get_input_output_len_with_system_prompt(self.dataset_sampler_list)
            )
            input_tokens = (system_prompt if system_prompt else []) + [
                random.randint(0, config.vocab_size - 1) for _ in range(input_len)
            ]

            output_tokens = [
                random.randint(0, config.vocab_size - 1) for _ in range(output_len)
            ]
            input_prompt = tokenizer.decode(input_tokens, add_special_tokens=False)
            output_prompt = tokenizer.decode(output_tokens, add_special_tokens=False)
            dataset.append(
                RequestData(input_prompt, input_tokens, output_prompt, output_tokens)
            )

        return dataset

    def __repr__(self):
        return (
            "DummyRequestDataset("
            f"sampler_list: {self.dataset_sampler_list}, "
            f"system_prompt_sampler_list: {self.system_prompt_sampler_list})"
        )
