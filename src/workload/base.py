import yaml

from dataclasses import dataclass
from abc import ABC, abstractmethod
from transformers import PreTrainedTokenizer
from typing import List
from pathlib import Path

from schemas import WorkloadConfig

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
        pass

    @abstractmethod
    def process(self, tokenizer: PreTrainedTokenizer)-> List[RequestData]:
        """Process dataset to list of RequestData."""
        pass

    def __getitem__(self, index: int) -> RequestData:
        """Get item from dataset."""
        return self.dataset[index]
