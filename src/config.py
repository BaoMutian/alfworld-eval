"""Configuration management for ALFWorld evaluation."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class LLMConfig:
    """LLM service configuration."""
    api_base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""
    model: str = "qwen/qwen3-8b"
    temperature: float = 0.3
    max_tokens: int = 1024
    timeout: int = 60
    # Qwen3 specific: enable thinking mode (for vLLM deployment)
    enable_thinking: Optional[bool] = None


@dataclass
class RetryConfig:
    """Retry configuration."""
    max_retries: int = 3
    retry_interval: float = 1.0
    max_retry_interval: float = 30.0


@dataclass
class TestConfig:
    """Test configuration."""
    num_games: Optional[int] = None
    task_types: Optional[List[int]] = None
    split: str = "valid_seen"
    seed: int = 42
    max_steps: int = 30


@dataclass
class PromptConfig:
    """Prompt configuration."""
    use_few_shot: bool = True
    history_length: int = 10


@dataclass
class RuntimeConfig:
    """Runtime configuration."""
    save_interval: int = 10
    output_dir: str = "results"
    debug: bool = False


@dataclass
class DataConfig:
    """Data configuration."""
    alfworld_data_path: str = "./alfworld_data"


@dataclass
class MaTTSConfig:
    """MaTTS (Memory-aware Test-Time Scaling) configuration."""
    enabled: bool = False
    sample_n: int = 3  # Number of parallel samples
    temperature: float = 0.7  # Higher temperature for diverse sampling
    max_tokens: int = 2048  # Max tokens for extraction response
    # Qwen3 specific: enable thinking mode for MaTTS extraction
    enable_thinking: Optional[bool] = None


@dataclass
class MemoryConfig:
    """Memory system configuration."""
    enabled: bool = False
    mode: str = "baseline"  # baseline | retrieve_only | retrieve_and_extract

    # Storage configuration
    memory_dir: str = "memory_banks"
    task_name: str = "alfworld"

    # Embedding model configuration
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_device: str = "cpu"

    # Retrieval parameters
    top_k: int = 1
    similarity_threshold: float = 0.5

    # MaTTS configuration
    matts: MaTTSConfig = field(default_factory=MaTTSConfig)

    def should_retrieve(self) -> bool:
        """Check if retrieval is enabled."""
        return self.enabled and self.mode in ("retrieve_only", "retrieve_and_extract")

    def should_extract(self) -> bool:
        """Check if extraction is enabled."""
        return self.enabled and self.mode == "retrieve_and_extract"


@dataclass
class Config:
    """Main configuration class."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    test: TestConfig = field(default_factory=TestConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "Config":
        """Create configuration from dictionary."""
        config = cls()

        if "llm" in data:
            config.llm = LLMConfig(**data["llm"])
        if "retry" in data:
            config.retry = RetryConfig(**data["retry"])
        if "test" in data:
            config.test = TestConfig(**data["test"])
        if "prompt" in data:
            config.prompt = PromptConfig(**data["prompt"])
        if "runtime" in data:
            # Filter out parallel_workers if present (deprecated)
            runtime_data = {
                k: v for k, v in data["runtime"].items() if k != "parallel_workers"}
            config.runtime = RuntimeConfig(**runtime_data)
        if "data" in data:
            config.data = DataConfig(**data["data"])
        if "memory" in data:
            memory_data = data["memory"].copy()
            # Handle nested matts config
            if "matts" in memory_data:
                memory_data["matts"] = MaTTSConfig(**memory_data["matts"])
            else:
                memory_data["matts"] = MaTTSConfig()
            config.memory = MemoryConfig(**memory_data)

        return config

    def validate(self) -> None:
        """Validate configuration."""
        # Check API key
        api_key = self.llm.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "API key not set. Set it in config or OPENAI_API_KEY environment variable."
            )
        self.llm.api_key = api_key

        # Check data path
        data_path = Path(self.data.alfworld_data_path)
        if not data_path.exists():
            raise ValueError(f"ALFWorld data path does not exist: {data_path}")

        # Check split
        valid_splits = ["valid_seen", "valid_train", "valid_unseen", "train"]
        if self.test.split not in valid_splits:
            raise ValueError(
                f"Invalid split: {self.test.split}. Must be one of {valid_splits}")

        # Check task types
        valid_task_types = [1, 2, 3, 4, 5, 6]
        if self.test.task_types is not None:
            for t in self.test.task_types:
                if t not in valid_task_types:
                    raise ValueError(
                        f"Invalid task type: {t}. Must be one of {valid_task_types}")

        # Create output directory
        Path(self.runtime.output_dir).mkdir(parents=True, exist_ok=True)

        # Validate memory configuration
        valid_memory_modes = ["baseline",
                              "retrieve_only", "retrieve_and_extract"]
        if self.memory.mode not in valid_memory_modes:
            raise ValueError(
                f"Invalid memory mode: {self.memory.mode}. "
                f"Must be one of {valid_memory_modes}"
            )

        # Create memory directory if memory is enabled
        if self.memory.enabled:
            Path(self.memory.memory_dir).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "llm": {
                "api_base_url": self.llm.api_base_url,
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
                "timeout": self.llm.timeout,
                "enable_thinking": self.llm.enable_thinking,
            },
            "retry": {
                "max_retries": self.retry.max_retries,
                "retry_interval": self.retry.retry_interval,
                "max_retry_interval": self.retry.max_retry_interval,
            },
            "test": {
                "num_games": self.test.num_games,
                "task_types": self.test.task_types,
                "split": self.test.split,
                "seed": self.test.seed,
                "max_steps": self.test.max_steps,
            },
            "prompt": {
                "use_few_shot": self.prompt.use_few_shot,
                "history_length": self.prompt.history_length,
            },
            "runtime": {
                "save_interval": self.runtime.save_interval,
                "output_dir": self.runtime.output_dir,
                "debug": self.runtime.debug,
            },
            "data": {
                "alfworld_data_path": self.data.alfworld_data_path,
            },
            "memory": {
                "enabled": self.memory.enabled,
                "mode": self.memory.mode,
                "memory_dir": self.memory.memory_dir,
                "task_name": self.memory.task_name,
                "embedding_model": self.memory.embedding_model,
                "embedding_device": self.memory.embedding_device,
                "top_k": self.memory.top_k,
                "similarity_threshold": self.memory.similarity_threshold,
                "matts": {
                    "enabled": self.memory.matts.enabled,
                    "sample_n": self.memory.matts.sample_n,
                    "temperature": self.memory.matts.temperature,
                    "max_tokens": self.memory.matts.max_tokens,
                    "enable_thinking": self.memory.matts.enable_thinking,
                },
            },
        }


def load_config(config_path: Optional[str] = None) -> Config:
    """Load and validate configuration.

    Args:
        config_path: Path to YAML config file. If None, uses default config.

    Returns:
        Validated Config object.
    """
    if config_path is None:
        # Use default config
        default_path = Path(__file__).parent.parent / "config" / "default.yaml"
        if default_path.exists():
            config = Config.from_yaml(str(default_path))
        else:
            config = Config()
    else:
        config = Config.from_yaml(config_path)

    config.validate()
    return config
