# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

# pylint: disable=too-few-public-methods

"""Experiment Schemas."""

from typing import List, Optional, Union, Literal

from pydantic import BaseModel, RootModel, Field

from enums import Distribution, HfDatasetFormat, WorkloadDataType

#########################
# Dummy Workload Schema #
#########################


class DummySystemPromptSamplerConfig(BaseModel):
    """Dummy system prompt sampler config."""

    name: Union[str, int]
    size: int


class DummyDatasetSamplerConfig(BaseModel):
    """Dummy dataset sampler base config."""

    type: Literal[Distribution.UNIFORM, Distribution.NORMAL]
    min: int
    max: int
    weight: int = 1


class DummyDatasetNormalSamplerConfig(DummyDatasetSamplerConfig):
    """Dummy dataset sampler config."""

    type: Literal[Distribution.NORMAL]
    std: float
    mean: float


class DummyDatasetUniformSamplerConfig(DummyDatasetSamplerConfig):
    """Dummy dataset sampler config."""

    type: Literal[Distribution.UNIFORM]


class DummyDatasetSamplerConfigUnion(RootModel):
    root: Union[DummyDatasetNormalSamplerConfig,
                DummyDatasetUniformSamplerConfig] \
                    = Field(..., discriminator="type")


class DummyDatasetSystemPromptSelectorConfig(BaseModel):
    """Dummy dataset system prompt selector config."""

    name: Union[str, int]
    weight: int = 1


class DummyDatasetSamplerMixtureConfig(BaseModel):
    """Dummy dataset sampler mixture config."""

    input: List[DummyDatasetSamplerConfigUnion]
    output: List[DummyDatasetSamplerConfigUnion]
    system_prompt: Optional[List[DummyDatasetSystemPromptSelectorConfig]] = None
    weight: int = 1


class DummyDatasetConfig(BaseModel):
    """Dummy dataset config."""

    vocab_size: int
    system_prompt_config: Optional[List[DummySystemPromptSamplerConfig]] = None
    dataset: List[DummyDatasetSamplerMixtureConfig]


#############################
# Huggingface Dataset Schema #
#############################


class HFDatasetConfig(BaseModel):
    """Huggingface dataset config."""

    path_or_name: str = "cnn_dailymail:3.0.0"
    format: HfDatasetFormat = HfDatasetFormat.JSON
    split: str = "test"
    max_length: int = 2048
    min_length: int = 1


class WorkloadConfig(BaseModel):
    """Dataset config."""

    type: WorkloadDataType
    dataset_size: int


class DummyWorkloadConfig(WorkloadConfig):
    """Dummy workload config."""

    dataset_config: DummyDatasetConfig


class HfWorkloadConfig(WorkloadConfig):
    """Huggingface workload config."""

    dataset_config: HFDatasetConfig


OneOfWorkloadConfig = Union[DummyWorkloadConfig, HfWorkloadConfig]
