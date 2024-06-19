# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""Workload base."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from schemas import WorkloadConfig
from transformers import PreTrainedTokenizer


@dataclass
class RequestData:
    """Request data."""

    prompt: str
    prompt_tokens: List[int]
    response: str
    response_tokens: List[int]


class RequestDataset(ABC):
    """Request dataset base class."""

    def __init__(self, workload_config: WorkloadConfig, tokenizer: PreTrainedTokenizer):
        """Initialize RequestDataset."""
        self.workload_config = workload_config
        self.dataset = self.process(tokenizer)

    @property
    @abstractmethod
    def warmup_size(self) -> int:
        """Warmup size."""

    @abstractmethod
    def process(self, tokenizer: PreTrainedTokenizer) -> List[RequestData]:
        """Process dataset to list of RequestData."""

    def __getitem__(self, index: int) -> RequestData:
        """Get item from dataset."""

        return self.dataset[index]
