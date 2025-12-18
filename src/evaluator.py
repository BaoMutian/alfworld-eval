"""Parallel evaluator for ALFWorld with checkpoint support."""

import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Set
from threading import Lock

from termcolor import colored
from tqdm import tqdm

from .config import Config
from .llm_client import LLMClient
from .environment import AlfWorldEnv, get_game_id_from_path
from .agent import GameResult, run_single_game
from .utils import (
    game_result_to_dict,
    compute_summary,
    save_results,
    load_checkpoint,
    save_checkpoint,
    format_progress,
    get_timestamp,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """Parallel evaluator for ALFWorld tasks."""

    def __init__(self, config: Config):
        """Initialize evaluator."""
        self.config = config
        self.llm_client = LLMClient(config.llm, config.retry)
        
        # Thread-safe state
        self._lock = Lock()
        self._completed_game_ids: Set[str] = set()
        self._results: List[GameResult] = []
        self._success_count = 0
        self._total_success_steps = 0
        
        # Setup paths
        self.output_dir = Path(config.runtime.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate run ID
        timestamp = get_timestamp().replace(":", "-")
        model_short = config.llm.model.split("/")[-1]
        self.run_id = f"{model_short}_{config.test.split}_{timestamp}"
        
        self.checkpoint_path = self.output_dir / f"{self.run_id}_checkpoint.json"
        self.results_path = self.output_dir / f"{self.run_id}_results.json"

    def _load_checkpoint(self) -> None:
        """Load checkpoint if exists."""
        checkpoint = load_checkpoint(str(self.checkpoint_path))
        self._completed_game_ids = checkpoint["completed_game_ids"]
        
        for r in checkpoint.get("results", []):
            result = GameResult(
                game_id=r["game_id"],
                game_file=r["game_file"],
                task_type=r["task_type"],
                task_type_id=r["task_type_id"],
                success=r["success"],
                steps=r["steps"],
                goal=r.get("goal", ""),
                actions=r.get("actions", []),
                observations=r.get("observations", []),
                thoughts=r.get("thoughts", []),
                error=r.get("error"),
            )
            self._results.append(result)
            if result.success:
                self._success_count += 1
                self._total_success_steps += result.steps
        
        if self._completed_game_ids:
            print(colored(f"Loaded checkpoint: {len(self._completed_game_ids)} games completed", "cyan"))

    def _save_checkpoint(self) -> None:
        """Save current checkpoint."""
        with self._lock:
            save_checkpoint(
                str(self.checkpoint_path),
                self._completed_game_ids,
                [game_result_to_dict(r) for r in self._results],
            )

    def _add_result(self, result: GameResult) -> None:
        """Add a result thread-safely."""
        with self._lock:
            self._results.append(result)
            self._completed_game_ids.add(result.game_id)
            if result.success:
                self._success_count += 1
                self._total_success_steps += result.steps

    def _get_stats(self) -> tuple:
        """Get current statistics thread-safely."""
        with self._lock:
            completed = len(self._results)
            successes = self._success_count
            success_avg_steps = self._total_success_steps / successes if successes > 0 else 0.0
            return completed, successes, success_avg_steps

    def get_game_files(self) -> List[str]:
        """Get list of game files based on configuration."""
        env = AlfWorldEnv(self.config.data.alfworld_data_path)
        game_files = env.get_game_files(
            split=self.config.test.split,
            task_types=self.config.test.task_types,
        )
        
        random.seed(self.config.test.seed)
        random.shuffle(game_files)
        
        if self.config.test.num_games is not None:
            game_files = game_files[:self.config.test.num_games]
        
        return game_files

    def _run_game_wrapper(self, game_file: str) -> Optional[GameResult]:
        """Wrapper for running a single game."""
        game_id = get_game_id_from_path(game_file)
        
        with self._lock:
            if game_id in self._completed_game_ids:
                return None
        
        result = run_single_game(
            alfworld_data_path=self.config.data.alfworld_data_path,
            game_file=game_file,
            llm_client=self.llm_client,
            use_few_shot=self.config.prompt.use_few_shot,
            history_length=self.config.prompt.history_length,
            max_steps=self.config.test.max_steps,
            debug=self.config.runtime.debug,
        )
        
        return result

    def run(self) -> None:
        """Run the evaluation."""
        # Print header
        print(colored("=" * 60, "cyan"))
        print(colored("ALFWorld Evaluation", "cyan", attrs=["bold"]))
        print(colored("=" * 60, "cyan"))
        print(f"Model: {colored(self.config.llm.model, 'white', attrs=['bold'])}")
        print(f"Split: {self.config.test.split}")
        print(f"Task types: {self.config.test.task_types or 'all'}")
        print(f"Workers: {self.config.runtime.parallel_workers}")
        print(colored("=" * 60, "cyan"))
        
        # Load checkpoint
        self._load_checkpoint()
        
        # Get game files
        game_files = self.get_game_files()
        total_games = len(game_files)
        
        # Filter completed games
        remaining_files = [
            f for f in game_files
            if get_game_id_from_path(f) not in self._completed_game_ids
        ]
        
        if not remaining_files:
            print(colored("All games already completed!", "green"))
        else:
            print(f"Total: {total_games} | Completed: {len(self._completed_game_ids)} | Remaining: {len(remaining_files)}")
            print(colored("-" * 60, "cyan"))
            
            completed_since_save = 0
            
            with ThreadPoolExecutor(max_workers=self.config.runtime.parallel_workers) as executor:
                futures = {
                    executor.submit(self._run_game_wrapper, f): f
                    for f in remaining_files
                }
                
                with tqdm(total=len(remaining_files), desc="Evaluating", ncols=80) as pbar:
                    for future in as_completed(futures):
                        game_file = futures[future]
                        
                        try:
                            result = future.result()
                            
                            if result is not None:
                                self._add_result(result)
                                completed_since_save += 1
                                
                                # Get stats
                                completed, successes, success_avg_steps = self._get_stats()
                                progress = format_progress(completed, total_games, successes, success_avg_steps)
                                
                                # Format result line with color
                                if result.success:
                                    status = colored("✓", "green")
                                    step_info = colored(f"{result.steps} steps", "green")
                                elif result.error:
                                    status = colored("✗", "red")
                                    step_info = colored("error", "red")
                                else:
                                    status = colored("✗", "yellow")
                                    step_info = f"{result.steps} steps"
                                
                                # Short game ID
                                short_id = result.game_id.split("/")[0][:30]
                                
                                tqdm.write(f"{progress} | {status} {short_id} | {step_info}")
                                
                                # Save checkpoint periodically
                                if completed_since_save >= self.config.runtime.save_interval:
                                    self._save_checkpoint()
                                    completed_since_save = 0
                        
                        except Exception as e:
                            tqdm.write(colored(f"Error: {game_file}: {e}", "red"))
                        
                        pbar.update(1)
        
        # Final save
        self._save_checkpoint()
        
        save_results(
            results=self._results,
            config_dict=self.config.to_dict(),
            output_path=str(self.results_path),
            model_name=self.config.llm.model,
        )
        
        # Print summary
        summary = compute_summary(self._results)
        
        print(colored("=" * 60, "cyan"))
        print(colored("EVALUATION COMPLETE", "green", attrs=["bold"]))
        print(colored("=" * 60, "cyan"))
        print(f"Total games: {summary['total_games']}")
        print(f"Successes: {colored(str(summary['successes']), 'green')}")
        print(f"Success rate: {colored(f\"{summary['success_rate']:.2%}\", 'green', attrs=['bold'])}")
        print(f"Average steps: {summary['avg_steps']:.1f}")
        print(f"Success avg steps: {colored(f\"{summary['success_avg_steps']:.1f}\", 'cyan')}")
        
        if summary["by_task_type"]:
            print(colored("-" * 40, "cyan"))
            print("By Task Type:")
            for task_type, stats in sorted(summary["by_task_type"].items()):
                rate_color = "green" if stats['success_rate'] >= 0.5 else "yellow"
                print(
                    f"  {task_type}: {stats['successes']}/{stats['total']} "
                    f"({colored(f\"{stats['success_rate']:.0%}\", rate_color)})"
                )
        
        print(colored("=" * 60, "cyan"))
        print(f"Results saved to: {colored(str(self.results_path), 'white', attrs=['bold'])}")


def run_evaluation(config: Config) -> None:
    """Run evaluation with the given configuration."""
    evaluator = Evaluator(config)
    evaluator.run()
