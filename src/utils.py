"""Utility functions for ALFWorld evaluation."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from .agent import GameResult


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


def get_timestamp() -> str:
    """Get current timestamp string.
    
    Returns:
        ISO format timestamp string.
    """
    return datetime.now().isoformat(timespec="seconds")


def game_result_to_dict(result: GameResult) -> Dict[str, Any]:
    """Convert GameResult to dictionary.
    
    Args:
        result: GameResult object.
        
    Returns:
        Dictionary representation.
    """
    return {
        "game_id": result.game_id,
        "game_file": result.game_file,
        "task_type": result.task_type,
        "task_type_id": result.task_type_id,
        "goal": result.goal,  # Include goal before actions
        "success": result.success,
        "steps": result.steps,
        "actions": result.actions,
        "observations": result.observations,
        "thoughts": result.thoughts,
        "error": result.error,
    }


def compute_summary(results: List[GameResult]) -> Dict[str, Any]:
    """Compute summary statistics from results.
    
    Args:
        results: List of GameResult objects.
        
    Returns:
        Summary statistics dictionary.
    """
    if not results:
        return {
            "total_games": 0,
            "successes": 0,
            "success_rate": 0.0,
            "avg_steps": 0.0,
            "success_avg_steps": 0.0,
            "by_task_type": {},
        }
    
    total = len(results)
    successes = sum(1 for r in results if r.success)
    total_steps = sum(r.steps for r in results)
    success_steps = sum(r.steps for r in results if r.success)
    
    # Per task type statistics
    by_task_type = {}
    task_type_results = {}
    
    for r in results:
        task_type = r.task_type
        if task_type not in task_type_results:
            task_type_results[task_type] = []
        task_type_results[task_type].append(r)
    
    for task_type, type_results in task_type_results.items():
        type_total = len(type_results)
        type_successes = sum(1 for r in type_results if r.success)
        type_steps = sum(r.steps for r in type_results)
        type_success_steps = sum(r.steps for r in type_results if r.success)
        
        by_task_type[task_type] = {
            "total": type_total,
            "successes": type_successes,
            "success_rate": type_successes / type_total if type_total > 0 else 0.0,
            "avg_steps": type_steps / type_total if type_total > 0 else 0.0,
            "success_avg_steps": type_success_steps / type_successes if type_successes > 0 else 0.0,
        }
    
    return {
        "total_games": total,
        "successes": successes,
        "success_rate": successes / total if total > 0 else 0.0,
        "avg_steps": total_steps / total if total > 0 else 0.0,
        "success_avg_steps": success_steps / successes if successes > 0 else 0.0,
        "by_task_type": by_task_type,
    }


def save_results(
    results: List[GameResult],
    config_dict: Dict[str, Any],
    output_path: str,
    model_name: str,
) -> None:
    """Save evaluation results to JSON file.
    
    Args:
        results: List of GameResult objects.
        config_dict: Configuration dictionary.
        output_path: Path to output file.
        model_name: Model name for the results.
    """
    summary = compute_summary(results)
    
    output = {
        "model": model_name,
        "timestamp": get_timestamp(),
        "config": config_dict,
        "summary": summary,
        "results": [game_result_to_dict(r) for r in results],
    }
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load checkpoint from file.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        
    Returns:
        Checkpoint data dictionary.
    """
    if not Path(checkpoint_path).exists():
        return {"completed_game_ids": set(), "results": []}
    
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Convert game IDs list to set for faster lookup
    data["completed_game_ids"] = set(data.get("completed_game_ids", []))
    
    return data


def save_checkpoint(
    checkpoint_path: str,
    completed_game_ids: set,
    results: List[Dict[str, Any]],
) -> None:
    """Save checkpoint to file.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        completed_game_ids: Set of completed game IDs.
        results: List of result dictionaries.
    """
    data = {
        "completed_game_ids": list(completed_game_ids),
        "results": results,
        "timestamp": get_timestamp(),
    }
    
    # Ensure directory exists
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def format_progress(current: int, total: int, successes: int, success_steps: int = 0) -> str:
    """Format progress string with colors.
    
    Args:
        current: Current task number.
        total: Total number of tasks.
        successes: Number of successes so far.
        success_steps: Total steps for successful games.
        
    Returns:
        Formatted progress string with colors.
    """
    success_rate = successes / current * 100 if current > 0 else 0
    avg_success_steps = success_steps / successes if successes > 0 else 0
    
    # Color code the success rate
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
    else:
        return f"{progress} SR: {rate} ({successes}/{current})"


def format_game_result(result: GameResult, game_num: int, total: int) -> str:
    """Format a single game result for display.
    
    Args:
        result: Game result.
        game_num: Current game number.
        total: Total games.
        
    Returns:
        Formatted result string.
    """
    if result.success:
        status = Colors.success("✓ SUCCESS")
    elif result.error:
        status = Colors.error(f"✗ ERROR: {result.error[:30]}...")
    else:
        status = Colors.warning("✗ FAILED")
    
    game_id_short = result.game_id.split("/")[0][:40]
    
    return f"{status} | {Colors.dim(game_id_short)} | {result.steps} steps"


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
