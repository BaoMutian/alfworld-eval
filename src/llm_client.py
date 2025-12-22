"""LLM client with retry mechanism for OpenAI-compatible APIs."""

import logging
from typing import List, Dict, Optional, Callable

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

# Optional callback for logging LLM interactions
_llm_log_callback: Optional[Callable] = None


def set_llm_log_callback(callback: Optional[Callable]) -> None:
    """Set callback for logging LLM interactions.
    
    Args:
        callback: Function(context, system_prompt, user_prompt, response, **kwargs)
    """
    global _llm_log_callback
    _llm_log_callback = callback


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
        
        # Context for logging
        self._current_context: str = "LLM Call"
        self._current_step: Optional[int] = None
        self._current_game_id: Optional[str] = None

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

    def set_context(
        self,
        context: str = "LLM Call",
        step: Optional[int] = None,
        game_id: Optional[str] = None,
    ) -> None:
        """Set context for logging.
        
        Args:
            context: Context label for this call
            step: Optional step number
            game_id: Optional game ID
        """
        self._current_context = context
        self._current_step = step
        self._current_game_id = game_id

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send chat completion request with retry.

        Args:
            messages: List of message dicts with 'role' and 'content'.

        Returns:
            Model response content.

        Raises:
            Exception: If all retries fail.
        """
        try:
            response = self._chat_with_retry(messages)
            return response
        except Exception as e:
            logger.error(
                f"LLM request failed after {self.retry_config.max_retries} retries: {e}")
            raise

    def chat_simple(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[str] = None,
        step: Optional[int] = None,
        game_id: Optional[str] = None,
    ) -> str:
        """Simple chat interface with system and user prompts.

        Args:
            system_prompt: System prompt content.
            user_prompt: User prompt content.
            context: Optional context label for logging.
            step: Optional step number for logging.
            game_id: Optional game ID for logging.

        Returns:
            Model response content.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        response = self.chat(messages)
        
        # Log the interaction if callback is set
        if _llm_log_callback:
            ctx = context if context else self._current_context
            stp = step if step is not None else self._current_step
            gid = game_id if game_id else self._current_game_id
            
            _llm_log_callback(
                context=ctx,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response=response,
                step=stp,
                game_id=gid,
            )
        
        return response
