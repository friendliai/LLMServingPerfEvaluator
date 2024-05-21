from enum import Enum

class WorkloadDataType(str, Enum):
    """Workload dataset type."""

    DUMMY = "dummy"
    HUGGINGFACE = "hf"

class Distribution(str, Enum):
    """Dummy dataset distribution."""

    UNIFORM = "uniform"
    NORMAL = "normal"

class HfDatasetFormat(str, Enum):
    """Supported file format for Huggingface dataset."""

    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    TXT = "txt"
