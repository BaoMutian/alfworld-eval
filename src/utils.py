"""Utility functions for ALFWorld evaluation."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from .agent import GameResult


def setup_logging(debug: bool = False, log_file: Optional[str] = None) -> None:
    """Setup logging configuration.
    
    Args:
        debug: Whether to enable debug logging.
        log_file: Optional path to log file.
    """
    level = logging.DEBUG if debug else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


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


def format_progress(current: int, total: int, successes: int) -> str:
    """Format progress string.
    
    Args:
        current: Current task number.
        total: Total number of tasks.
        successes: Number of successes so far.
        
    Returns:
        Formatted progress string.
    """
    success_rate = successes / current * 100 if current > 0 else 0
    return f"[{current}/{total}] Success: {successes}/{current} ({success_rate:.1f}%)"

