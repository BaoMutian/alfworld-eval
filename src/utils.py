"""Utility functions for ALFWorld evaluation."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, TYPE_CHECKING

# Re-export logging utilities
from .logging_utils import (
    Colors,
    ColoredFormatter,
    setup_logging,
    get_debug_logger,
    log_game_start,
    log_game_end,
    log_step_interaction,
    format_step_info,
)

if TYPE_CHECKING:
    from .agent import GameResult

# Export all logging utilities
__all__ = [
    "Colors",
    "ColoredFormatter",
    "setup_logging",
    "get_debug_logger",
    "log_game_start",
    "log_game_end",
    "log_step_interaction",
    "format_step_info",
    "get_timestamp",
    "game_result_to_dict",
    "compute_summary",
    "save_results",
    "load_checkpoint",
    "save_checkpoint",
    "format_progress",
    "format_game_result",
]


def get_timestamp() -> str:
    """Get current timestamp string.
    
    Returns:
        ISO format timestamp string.
    """
    return datetime.now().isoformat(timespec="seconds")


def game_result_to_dict(result: "GameResult") -> Dict[str, Any]:
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


def compute_summary(results: List[Any]) -> Dict[str, Any]:
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
    results: List[Any],
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


def format_game_result(result: "GameResult", game_num: int, total: int) -> str:
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
