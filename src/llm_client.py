"""LLM client with retry mechanism for OpenAI-compatible APIs."""

import logging
from typing import List, Dict, Optional, Any

from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from .config import LLMConfig, RetryConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM client supporting OpenAI-compatible APIs with retry mechanism."""

    def __init__(self, llm_config: LLMConfig, retry_config: RetryConfig):
        """Initialize LLM client.

        Args:
            llm_config: LLM service configuration.
            retry_config: Retry configuration.
        """
        self.config = llm_config
        self.retry_config = retry_config

        # Use "EMPTY" as api_key for local deployment if not set
        api_key = llm_config.api_key or "EMPTY"

        self.client = OpenAI(
            api_key=api_key,
            base_url=llm_config.api_base_url,
            timeout=float(llm_config.timeout),
        )

        # Build extra_body for vLLM (Qwen3 thinking mode)
        self.extra_body: Optional[Dict[str, Any]] = None
        if llm_config.enable_thinking is not None:
            self.extra_body = {
                "chat_template_kwargs": {
                    "enable_thinking": llm_config.enable_thinking
                }
            }

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send chat completion request with retry.

        Args:
            messages: List of message dicts with 'role' and 'content'.

        Returns:
            Model response content.

        Raises:
            Exception: If all retries fail.
        """
        @retry(
            stop=stop_after_attempt(self.retry_config.max_retries),
            wait=wait_exponential(
                multiplier=self.retry_config.retry_interval,
                max=self.retry_config.max_retry_interval,
            ),
            retry=retry_if_exception_type(Exception),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        def _call():
            kwargs = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
            }
            if self.config.max_tokens and self.config.max_tokens > 0:
                kwargs["max_tokens"] = self.config.max_tokens
            if self.extra_body:
                kwargs["extra_body"] = self.extra_body

            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content

        return _call()

    def chat_simple(self, system_prompt: str, user_prompt: str) -> str:
        """Simple chat interface with system and user prompts.

        Args:
            system_prompt: System prompt content.
            user_prompt: User prompt content.

        Returns:
            Model response content.
        """
        return self.chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
