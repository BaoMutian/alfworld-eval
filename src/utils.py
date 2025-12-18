"""Utility functions for data processing and checkpointing."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import GameResult


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().isoformat(timespec="seconds")


def game_result_to_dict(result: "GameResult") -> Dict[str, Any]:
    """Convert GameResult to dictionary."""
    data = {
        "game_id": result.game_id,
        "game_file": result.game_file,
        "task_type": result.task_type,
        "task_type_id": result.task_type_id,
        "goal": result.goal,
        "success": result.success,
        "steps": result.steps,
        "actions": result.actions,
        "observations": result.observations,
        "thoughts": result.thoughts,
        "error": result.error,
    }
    
    # Add memory-related fields if present
    if hasattr(result, 'used_memories') and result.used_memories:
        data["used_memories"] = result.used_memories
    
    return data


def compute_summary(results: List[Any]) -> Dict[str, Any]:
    """Compute summary statistics from results."""
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
    task_type_results: Dict[str, List[Any]] = {}

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
    """Save evaluation results to JSON file."""
    summary = compute_summary(results)

    output = {
        "model": model_name,
        "timestamp": get_timestamp(),
        "config": config_dict,
        "summary": summary,
        "results": [game_result_to_dict(r) for r in results],
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load checkpoint from file."""
    if not Path(checkpoint_path).exists():
        return {"completed_game_ids": set(), "results": []}

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["completed_game_ids"] = set(data.get("completed_game_ids", []))
    return data


def save_checkpoint(
    checkpoint_path: str,
    completed_game_ids: set,
    results: List[Dict[str, Any]],
) -> None:
    """Save checkpoint to file."""
    data = {
        "completed_game_ids": list(completed_game_ids),
        "results": results,
        "timestamp": get_timestamp(),
    }

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
