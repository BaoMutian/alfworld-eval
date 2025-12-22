"""Logging and terminal output utilities."""

import logging
import sys
from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import GameResult


class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    WHITE = "\033[37m"
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

    def format(self, record):
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")

        if record.levelno >= logging.ERROR:
            prefix = f"{Colors.BRIGHT_RED}[{timestamp}] ERROR:{Colors.RESET}"
        elif record.levelno >= logging.WARNING:
            prefix = f"{Colors.BRIGHT_YELLOW}[{timestamp}] WARN:{Colors.RESET}"
        else:
            prefix = f"{Colors.dim(f'[{timestamp}]')}"

        return f"{prefix} {record.getMessage()}"


# Module-level debug logger
_debug_file_logger: Optional[logging.Logger] = None


def setup_logging(debug: bool = False, log_file: Optional[str] = None) -> None:
    """Setup logging configuration."""
    global _debug_file_logger

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    root_logger.handlers.clear()

    # Console handler - only warnings and errors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(ColoredFormatter())
    root_logger.addHandler(console_handler)

    # Suppress noisy loggers
    for name in ("httpx", "httpcore", "openai", "urllib3", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # Setup debug file logger
    if debug and log_file:
        _debug_file_logger = logging.getLogger("alfworld.debug")
        _debug_file_logger.setLevel(logging.DEBUG)
        _debug_file_logger.propagate = False
        _debug_file_logger.handlers.clear()

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        _debug_file_logger.addHandler(file_handler)


def _log_debug(message: str) -> None:
    """Log to debug file if available."""
    if _debug_file_logger:
        _debug_file_logger.debug(message)


def _get_timestamp() -> str:
    """Get formatted timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# LLM Interaction Logging
# ============================================================

def log_llm_interaction(
    context: str,
    system_prompt: str,
    user_prompt: str,
    response: str,
    step: Optional[int] = None,
    game_id: Optional[str] = None,
) -> None:
    """Log a complete LLM interaction (prompt + response).
    
    Args:
        context: Context label (e.g., "Agent Step", "Memory Extraction")
        system_prompt: System prompt content
        user_prompt: User prompt content
        response: LLM response content
        step: Optional step number for agent interactions
        game_id: Optional game ID for context
    """
    _log_debug("")
    _log_debug("╔" + "═" * 78 + "╗")
    
    # Header
    header = f"║ [{_get_timestamp()}] {context}"
    if step is not None:
        header += f" - Step {step}"
    if game_id:
        header += f" ({game_id[:40]}...)" if len(game_id) > 40 else f" ({game_id})"
    header = header.ljust(79) + "║"
    _log_debug(header)
    
    _log_debug("╠" + "═" * 78 + "╣")
    
    # System Prompt (abbreviated if too long)
    _log_debug("║ SYSTEM PROMPT:".ljust(79) + "║")
    _log_debug("╟" + "─" * 78 + "╢")
    _log_multiline(system_prompt, max_lines=20)
    
    _log_debug("╟" + "─" * 78 + "╢")
    
    # User Prompt
    _log_debug("║ USER PROMPT:".ljust(79) + "║")
    _log_debug("╟" + "─" * 78 + "╢")
    _log_multiline(user_prompt)
    
    _log_debug("╟" + "─" * 78 + "╢")
    
    # LLM Response
    _log_debug("║ LLM RESPONSE:".ljust(79) + "║")
    _log_debug("╟" + "─" * 78 + "╢")
    _log_multiline(response)
    
    _log_debug("╚" + "═" * 78 + "╝")
    _log_debug("")


def _log_multiline(text: str, max_lines: Optional[int] = None) -> None:
    """Log multiline text with proper formatting."""
    lines = text.split('\n')
    
    if max_lines and len(lines) > max_lines:
        # Show first portion and last portion
        show_first = max_lines // 2
        show_last = max_lines - show_first - 1
        
        for line in lines[:show_first]:
            _log_debug(f"║ {line[:76]}".ljust(79) + "║")
        _log_debug(f"║ ... ({len(lines) - max_lines} lines omitted) ...".ljust(79) + "║")
        for line in lines[-show_last:]:
            _log_debug(f"║ {line[:76]}".ljust(79) + "║")
    else:
        for line in lines:
            # Handle long lines
            while len(line) > 76:
                _log_debug(f"║ {line[:76]}".ljust(79) + "║")
                line = line[76:]
            _log_debug(f"║ {line}".ljust(79) + "║")


def log_system_prompt(system_prompt: str) -> None:
    """Log the system prompt once at the beginning (simplified header)."""
    _log_debug("")
    _log_debug("╔" + "═" * 78 + "╗")
    _log_debug(f"║ [{_get_timestamp()}] SYSTEM PROMPT (used for all agent interactions)".ljust(79) + "║")
    _log_debug("╠" + "═" * 78 + "╣")
    _log_multiline(system_prompt)
    _log_debug("╚" + "═" * 78 + "╝")
    _log_debug("")


def log_game_header(game_id: str, goal: str) -> None:
    """Log game header (minimal, just for context)."""
    _log_debug("")
    _log_debug("▓" * 80)
    _log_debug(f"  GAME: {game_id}")
    _log_debug(f"  GOAL: {goal}")
    _log_debug("▓" * 80)


def log_game_result(game_id: str, success: bool, steps: int) -> None:
    """Log game result summary."""
    status = "✓ SUCCESS" if success else "✗ FAILED"
    _log_debug("")
    _log_debug(f"  RESULT: {status} in {steps} steps")
    _log_debug("▓" * 80)
    _log_debug("")


# ============================================================
# Legacy functions (for backward compatibility)
# ============================================================

def log_game_start(game_id: str, goal: str) -> None:
    """Log the start of a game."""
    log_game_header(game_id, goal)


def log_game_end(game_id: str, success: bool, steps: int) -> None:
    """Log the end of a game."""
    log_game_result(game_id, success, steps)


def log_step_interaction(
    step: int,
    user_prompt: str,
    response: str,
    action: str,
    observation: str,
) -> None:
    """Legacy function - now handled via log_llm_interaction."""
    # This is now a no-op as logging is done at LLM client level
    pass


# ============================================================
# Terminal Output Formatting
# ============================================================

def format_step_info(step: int, action: str, observation: str, max_obs_len: int = 80) -> str:
    """Format step information for terminal display."""
    obs_display = observation.replace('\n', ' ')[:max_obs_len]
    if len(observation) > max_obs_len:
        obs_display += "..."

    return (
        f"  {Colors.dim(f'[Step {step:2d}]')} "
        f"{Colors.info('Action:')} {action}\n"
        f"           {Colors.dim('Obs:')} {obs_display}"
    )


def format_progress(current: int, total: int, successes: int, success_steps: int = 0) -> str:
    """Format progress string with colors."""
    success_rate = successes / current * 100 if current > 0 else 0
    avg_success_steps = success_steps / successes if successes > 0 else 0

    if success_rate >= 70:
        rate_color = Colors.BRIGHT_GREEN
    elif success_rate >= 50:
        rate_color = Colors.BRIGHT_YELLOW
    else:
        rate_color = Colors.BRIGHT_RED

    progress = f"{Colors.BRIGHT_CYAN}[{current}/{total}]{Colors.RESET}"
    rate = f"{rate_color}{success_rate:.1f}%{Colors.RESET}"

    if successes > 0:
        steps_info = f"{Colors.dim(f'avg_steps={avg_success_steps:.1f}')}"
        return f"{progress} SR: {rate} ({successes}/{current}) {steps_info}"
    return f"{progress} SR: {rate} ({successes}/{current})"


def format_game_result(result: "GameResult") -> str:
    """Format a single game result for display."""
    if result.success:
        status = Colors.success("✓ SUCCESS")
    elif result.error:
        status = Colors.error(f"✗ ERROR: {result.error[:30]}...")
    else:
        status = Colors.warning("✗ FAILED")

    game_id_short = result.game_id.split("/")[0][:40]
    return f"{status} | {Colors.dim(game_id_short)} | {result.steps} steps"
