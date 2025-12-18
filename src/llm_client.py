"""LLM client with retry mechanism for OpenAI-compatible APIs."""

import logging
from typing import List, Dict, Any, Optional

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
        
        self.client = OpenAI(
            api_key=llm_config.api_key,
            base_url=llm_config.api_base_url,
            timeout=llm_config.timeout,
        )
        
        # Create retry decorator with config
        self._chat_with_retry = self._create_retry_wrapper()

    def _create_retry_wrapper(self):
        """Create a retry-wrapped chat completion function."""
        @retry(
            stop=stop_after_attempt(self.retry_config.max_retries),
            wait=wait_exponential(
                multiplier=self.retry_config.retry_interval,
                max=self.retry_config.max_retry_interval,
            ),
            retry=retry_if_exception_type((Exception,)),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        def _chat(messages: List[Dict[str, str]]) -> str:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content
        
        return _chat

    def chat(
        self,
        messages: List[Dict[str, str]],
        debug: bool = False,
    ) -> str:
        """Send chat completion request with retry.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            debug: Whether to print debug information.
            
        Returns:
            Model response content.
            
        Raises:
            Exception: If all retries fail.
        """
        if debug:
            logger.debug("=" * 60)
            logger.debug("LLM REQUEST")
            logger.debug("=" * 60)
            for msg in messages:
                logger.debug(f"[{msg['role'].upper()}]")
                logger.debug(msg['content'][:500] + "..." if len(msg['content']) > 500 else msg['content'])
            logger.debug("=" * 60)
        
        try:
            response = self._chat_with_retry(messages)
            
            if debug:
                logger.debug("LLM RESPONSE")
                logger.debug("=" * 60)
                logger.debug(response)
                logger.debug("=" * 60)
            
            return response
            
        except Exception as e:
            logger.error(f"LLM request failed after {self.retry_config.max_retries} retries: {e}")
            raise

    def chat_simple(
        self,
        system_prompt: str,
        user_prompt: str,
        debug: bool = False,
    ) -> str:
        """Simple chat interface with system and user prompts.
        
        Args:
            system_prompt: System prompt content.
            user_prompt: User prompt content.
            debug: Whether to print debug information.
            
        Returns:
            Model response content.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.chat(messages, debug=debug)

