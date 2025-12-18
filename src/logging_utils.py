"""Logging utilities for ALFWorld evaluation."""

import logging
import sys
from datetime import datetime
from typing import Optional


# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    # Regular colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    
    @classmethod
    def success(cls, text: str) -> str:
        return f"{cls.BRIGHT_GREEN}{text}{cls.RESET}"
    
    @classmethod
    def error(cls, text: str) -> str:
        return f"{cls.BRIGHT_RED}{text}{cls.RESET}"
    
    @classmethod
    def warning(cls, text: str) -> str:
        return f"{cls.BRIGHT_YELLOW}{text}{cls.RESET}"
    
    @classmethod
    def info(cls, text: str) -> str:
        return f"{cls.BRIGHT_CYAN}{text}{cls.RESET}"
    
    @classmethod
    def highlight(cls, text: str) -> str:
        return f"{cls.BOLD}{cls.WHITE}{text}{cls.RESET}"
    
    @classmethod
    def dim(cls, text: str) -> str:
        return f"{cls.WHITE}{text}{cls.RESET}"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    LEVEL_COLORS = {
        logging.DEBUG: Colors.WHITE,
        logging.INFO: Colors.RESET,
        logging.WARNING: Colors.BRIGHT_YELLOW,
        logging.ERROR: Colors.BRIGHT_RED,
        logging.CRITICAL: Colors.BOLD + Colors.BRIGHT_RED,
    }
    
    def format(self, record):
        # Add color based on level
        color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        
        # Build message
        if record.levelno >= logging.ERROR:
            prefix = f"{Colors.BRIGHT_RED}[{timestamp}] ERROR:{Colors.RESET}"
        elif record.levelno >= logging.WARNING:
            prefix = f"{Colors.BRIGHT_YELLOW}[{timestamp}] WARN:{Colors.RESET}"
        else:
            prefix = f"{Colors.dim(f'[{timestamp}]')}"
        
        return f"{prefix} {color}{record.getMessage()}{Colors.RESET}"


# Debug logger for detailed prompt/response logging to file
_debug_file_logger: Optional[logging.Logger] = None


def get_debug_logger() -> Optional[logging.Logger]:
    """Get the debug file logger."""
    return _debug_file_logger


def setup_logging(debug: bool = False, log_file: Optional[str] = None) -> None:
    """Setup logging configuration.
    
    Args:
        debug: Whether to enable debug logging.
        log_file: Optional path to log file.
    """
    global _debug_file_logger
    
    level = logging.DEBUG if debug else logging.INFO
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with colors (INFO level for cleaner output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Terminal shows INFO and above
    console_handler.setFormatter(ColoredFormatter())
    root_logger.addHandler(console_handler)
    
    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # Setup debug file logger for detailed prompt/response logging
    if debug and log_file:
        _debug_file_logger = logging.getLogger("alfworld.debug")
        _debug_file_logger.setLevel(logging.DEBUG)
        _debug_file_logger.propagate = False  # Don't propagate to root logger
        
        # File handler for debug logger
        file_formatter = logging.Formatter(
            "[%(asctime)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        _debug_file_logger.addHandler(file_handler)


def log_game_start(game_id: str, goal: str) -> None:
    """Log the start of a game to debug file.
    
    Args:
        game_id: Game identifier.
        goal: Task goal description.
    """
    debug_logger = get_debug_logger()
    if debug_logger:
        debug_logger.debug("=" * 80)
        debug_logger.debug(f"GAME START: {game_id}")
        debug_logger.debug(f"GOAL: {goal}")
        debug_logger.debug("=" * 80)


def log_game_end(game_id: str, success: bool, steps: int) -> None:
    """Log the end of a game to debug file.
    
    Args:
        game_id: Game identifier.
        success: Whether the game was successful.
        steps: Number of steps taken.
    """
    debug_logger = get_debug_logger()
    if debug_logger:
        status = "SUCCESS" if success else "FAILED"
        debug_logger.debug("-" * 80)
        debug_logger.debug(f"GAME END: {game_id} | {status} | {steps} steps")
        debug_logger.debug("=" * 80)
        debug_logger.debug("")


def log_step_interaction(
    step: int,
    system_prompt: str,
    user_prompt: str,
    response: str,
    action: str,
    observation: str,
) -> None:
    """Log a complete step interaction to debug file.
    
    Args:
        step: Step number.
        system_prompt: System prompt sent to LLM.
        user_prompt: User prompt sent to LLM.
        response: LLM response.
        action: Parsed action.
        observation: Environment observation.
    """
    debug_logger = get_debug_logger()
    if debug_logger:
        debug_logger.debug("-" * 80)
        debug_logger.debug(f"STEP {step}")
        debug_logger.debug("-" * 80)
        debug_logger.debug("")
        debug_logger.debug(">>> SYSTEM PROMPT:")
        debug_logger.debug(system_prompt)
        debug_logger.debug("")
        debug_logger.debug(">>> USER PROMPT:")
        debug_logger.debug(user_prompt)
        debug_logger.debug("")
        debug_logger.debug(">>> LLM RESPONSE:")
        debug_logger.debug(response)
        debug_logger.debug("")
        debug_logger.debug(f">>> PARSED ACTION: {action}")
        debug_logger.debug(f">>> OBSERVATION: {observation}")
        debug_logger.debug("")


def format_step_info(step: int, action: str, observation: str, max_obs_len: int = 80) -> str:
    """Format step information for terminal display.
    
    Args:
        step: Step number.
        action: Action taken.
        observation: Environment observation.
        max_obs_len: Maximum observation length to display.
        
    Returns:
        Formatted step string.
    """
    obs_display = observation.replace('\n', ' ')[:max_obs_len]
    if len(observation) > max_obs_len:
        obs_display += "..."
    
    return (
        f"  {Colors.dim(f'[Step {step:2d}]')} "
        f"{Colors.info('Action:')} {action}\n"
        f"           {Colors.dim('Obs:')} {obs_display}"
    )

