"""Parallel evaluator for ALFWorld with checkpoint support."""

import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Set
from threading import Lock

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
        """Initialize evaluator.
        
        Args:
            config: Evaluation configuration.
        """
        self.config = config
        self.llm_client = LLMClient(config.llm, config.retry)
        
        # Thread-safe state
        self._lock = Lock()
        self._completed_game_ids: Set[str] = set()
        self._results: List[GameResult] = []
        self._success_count = 0
        
        # Setup paths
        self.output_dir = Path(config.runtime.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate run ID for this evaluation
        timestamp = get_timestamp().replace(":", "-")
        model_short = config.llm.model.split("/")[-1]
        self.run_id = f"{model_short}_{config.test.split}_{timestamp}"
        
        self.checkpoint_path = self.output_dir / f"{self.run_id}_checkpoint.json"
        self.results_path = self.output_dir / f"{self.run_id}_results.json"

    def _load_checkpoint(self) -> None:
        """Load checkpoint if exists."""
        checkpoint = load_checkpoint(str(self.checkpoint_path))
        self._completed_game_ids = checkpoint["completed_game_ids"]
        
        # Reconstruct GameResult objects from checkpoint
        for r in checkpoint.get("results", []):
            result = GameResult(
                game_id=r["game_id"],
                game_file=r["game_file"],
                task_type=r["task_type"],
                task_type_id=r["task_type_id"],
                success=r["success"],
                steps=r["steps"],
                actions=r.get("actions", []),
                observations=r.get("observations", []),
                thoughts=r.get("thoughts", []),
                error=r.get("error"),
            )
            self._results.append(result)
            if result.success:
                self._success_count += 1
        
        if self._completed_game_ids:
            logger.info(f"Loaded checkpoint with {len(self._completed_game_ids)} completed games")

    def _save_checkpoint(self) -> None:
        """Save current checkpoint."""
        with self._lock:
            save_checkpoint(
                str(self.checkpoint_path),
                self._completed_game_ids,
                [game_result_to_dict(r) for r in self._results],
            )

    def _add_result(self, result: GameResult) -> None:
        """Add a result thread-safely.
        
        Args:
            result: Game result to add.
        """
        with self._lock:
            self._results.append(result)
            self._completed_game_ids.add(result.game_id)
            if result.success:
                self._success_count += 1

    def _get_progress(self) -> tuple:
        """Get current progress thread-safely.
        
        Returns:
            Tuple of (completed_count, success_count).
        """
        with self._lock:
            return len(self._results), self._success_count

    def get_game_files(self) -> List[str]:
        """Get list of game files based on configuration.
        
        Returns:
            List of game file paths.
        """
        env = AlfWorldEnv(self.config.data.alfworld_data_path)
        game_files = env.get_game_files(
            split=self.config.test.split,
            task_types=self.config.test.task_types,
        )
        
        # Shuffle with seed
        random.seed(self.config.test.seed)
        random.shuffle(game_files)
        
        # Limit number of games if specified
        if self.config.test.num_games is not None:
            game_files = game_files[:self.config.test.num_games]
        
        return game_files

    def _run_game_wrapper(self, game_file: str) -> Optional[GameResult]:
        """Wrapper for running a single game.
        
        Args:
            game_file: Path to game file.
            
        Returns:
            GameResult or None if already completed.
        """
        game_id = get_game_id_from_path(game_file)
        
        # Skip if already completed (checkpoint resume)
        with self._lock:
            if game_id in self._completed_game_ids:
                return None
        
        # Run the game
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
        logger.info("=" * 60)
        logger.info("ALFWorld Evaluation")
        logger.info("=" * 60)
        logger.info(f"Model: {self.config.llm.model}")
        logger.info(f"Split: {self.config.test.split}")
        logger.info(f"Task types: {self.config.test.task_types or 'all'}")
        logger.info(f"Parallel workers: {self.config.runtime.parallel_workers}")
        logger.info(f"Results path: {self.results_path}")
        logger.info("=" * 60)
        
        # Load checkpoint
        self._load_checkpoint()
        
        # Get game files
        game_files = self.get_game_files()
        total_games = len(game_files)
        
        # Filter out already completed games
        remaining_files = [
            f for f in game_files
            if get_game_id_from_path(f) not in self._completed_game_ids
        ]
        
        if not remaining_files:
            logger.info("All games already completed!")
        else:
            logger.info(f"Total games: {total_games}")
            logger.info(f"Already completed: {len(self._completed_game_ids)}")
            logger.info(f"Remaining: {len(remaining_files)}")
            logger.info("=" * 60)
            
            # Run evaluation with parallel workers
            completed_since_save = 0
            
            with ThreadPoolExecutor(max_workers=self.config.runtime.parallel_workers) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(self._run_game_wrapper, f): f
                    for f in remaining_files
                }
                
                # Process results as they complete
                with tqdm(total=len(remaining_files), desc="Evaluating") as pbar:
                    for future in as_completed(futures):
                        game_file = futures[future]
                        
                        try:
                            result = future.result()
                            
                            if result is not None:
                                self._add_result(result)
                                completed_since_save += 1
                                
                                # Log progress
                                completed, successes = self._get_progress()
                                status = "✓" if result.success else "✗"
                                logger.info(
                                    f"{format_progress(completed, total_games, successes)} "
                                    f"| {result.game_id} | {status} | {result.steps} steps"
                                )
                                
                                # Save checkpoint periodically
                                if completed_since_save >= self.config.runtime.save_interval:
                                    self._save_checkpoint()
                                    completed_since_save = 0
                                    logger.debug("Checkpoint saved")
                        
                        except Exception as e:
                            logger.error(f"Error processing {game_file}: {e}")
                        
                        pbar.update(1)
        
        # Final save
        self._save_checkpoint()
        
        # Save final results
        save_results(
            results=self._results,
            config_dict=self.config.to_dict(),
            output_path=str(self.results_path),
            model_name=self.config.llm.model,
        )
        
        # Print summary
        summary = compute_summary(self._results)
        
        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total games: {summary['total_games']}")
        logger.info(f"Successes: {summary['successes']}")
        logger.info(f"Success rate: {summary['success_rate']:.2%}")
        logger.info(f"Average steps: {summary['avg_steps']:.1f}")
        logger.info(f"Success avg steps: {summary['success_avg_steps']:.1f}")
        
        if summary["by_task_type"]:
            logger.info("-" * 40)
            logger.info("By Task Type:")
            for task_type, stats in sorted(summary["by_task_type"].items()):
                logger.info(
                    f"  {task_type}: {stats['successes']}/{stats['total']} "
                    f"({stats['success_rate']:.2%})"
                )
        
        logger.info("=" * 60)
        logger.info(f"Results saved to: {self.results_path}")


def run_evaluation(config: Config) -> None:
    """Run evaluation with the given configuration.
    
    Args:
        config: Evaluation configuration.
    """
    evaluator = Evaluator(config)
    evaluator.run()

