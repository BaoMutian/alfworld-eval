"""Sequential evaluator for ALFWorld with checkpoint support and memory integration."""

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import List, Set, Optional

from tqdm import tqdm

from .config import Config
from .llm_client import LLMClient, set_llm_log_callback
from .environment import AlfWorldEnv, get_game_id_from_path, get_task_type_from_path
from .agent import GameResult, ReActAgent
from .prompts import extract_task_description
from .utils import (
    game_result_to_dict,
    compute_summary,
    save_results,
    load_checkpoint,
    save_checkpoint,
    get_timestamp,
)
from .logging_utils import (
    Colors,
    setup_logging,
    log_llm_call,
    format_progress,
    format_game_result,
)

logger = logging.getLogger(__name__)


def generate_run_id(config: Config) -> str:
    """Generate a stable run ID based on configuration."""
    key_params = {
        "model": config.llm.model,
        "split": config.test.split,
        "task_types": sorted(config.test.task_types) if config.test.task_types else None,
        "num_games": config.test.num_games,
        "seed": config.test.seed,
        "max_steps": config.test.max_steps,
        "use_few_shot": config.prompt.use_few_shot,
        "history_length": config.prompt.history_length,
        # Include memory config in hash
        "memory_enabled": config.memory.enabled,
        "memory_mode": config.memory.mode,
    }

    params_hash = hashlib.md5(
        json.dumps(key_params, sort_keys=True).encode()
    ).hexdigest()[:8]

    model_short = config.llm.model.split("/")[-1]
    task_str = "all" if not config.test.task_types else f"t{''.join(map(str, config.test.task_types))}"
    num_str = "full" if config.test.num_games is None else f"n{config.test.num_games}"

    # Add memory mode suffix if enabled
    memory_suffix = ""
    if config.memory.enabled:
        mode_short = {"baseline": "base", "retrieve_only": "ret",
                      "retrieve_and_extract": "retex"}
        memory_suffix = f"_mem{mode_short.get(config.memory.mode, config.memory.mode[:3])}"

    return f"{model_short}_{config.test.split}_{task_str}_{num_str}{memory_suffix}_{params_hash}"


class Evaluator:
    """Sequential evaluator for ALFWorld tasks with optional memory support."""

    def __init__(self, config: Config):
        """Initialize evaluator."""
        self.config = config
        self.llm_client = LLMClient(config.llm, config.retry)

        # State
        self._completed_game_ids: Set[str] = set()
        self._results: List[GameResult] = []
        self._success_count = 0
        self._success_steps = 0

        # Setup paths
        self.output_dir = Path(config.runtime.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = generate_run_id(config)
        self.checkpoint_path = self.output_dir / \
            f"{self.run_id}_checkpoint.json"
        self.results_path = self.output_dir / f"{self.run_id}_results.json"
        self.debug_log_path = self.output_dir / f"{self.run_id}_debug.log"

        if config.runtime.debug:
            setup_logging(debug=True, log_file=str(self.debug_log_path))
            # Set LLM logging callback
            set_llm_log_callback(log_llm_call)

        # Initialize memory components if enabled
        self.memory_store = None
        self.memory_retriever = None
        self.memory_extractor = None
        self._init_memory()

    def _init_memory(self) -> None:
        """Initialize memory components if enabled."""
        if not self.config.memory.enabled:
            return

        try:
            from .memory import (
                EmbeddingModel,
                MemoryStore,
                MemoryRetriever,
                MemoryExtractor,
            )

            # Initialize embedding model
            embedding_model = EmbeddingModel(
                model_name=self.config.memory.embedding_model,
                device=self.config.memory.embedding_device,
            )

            # Initialize memory store
            self.memory_store = MemoryStore(
                memory_dir=self.config.memory.memory_dir,
                task_name=self.config.memory.task_name,
                embedding_model=embedding_model,
            )

            # Initialize retriever if needed
            if self.config.memory.should_retrieve():
                self.memory_retriever = MemoryRetriever(
                    store=self.memory_store,
                    embedding_model=embedding_model,
                    top_k=self.config.memory.top_k,
                    similarity_threshold=self.config.memory.similarity_threshold,
                )

            # Initialize extractor if needed
            if self.config.memory.should_extract():
                self.memory_extractor = MemoryExtractor(
                    llm_client=self.llm_client,
                    temperature=self.config.llm.temperature,
                    max_tokens=self.config.llm.max_tokens,
                )

            logger.info(
                f"Memory system initialized: mode={self.config.memory.mode}, "
                f"store_size={self.memory_store.size()}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize memory system: {e}")
            logger.warning("Falling back to baseline mode (no memory)")
            self.memory_store = None
            self.memory_retriever = None
            self.memory_extractor = None

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
                used_memories=r.get("used_memories", []),
            )
            self._results.append(result)
            if result.success:
                self._success_count += 1
                self._success_steps += result.steps

        if self._completed_game_ids:
            print(
                f"{Colors.info('Checkpoint found:')} {len(self._completed_game_ids)} games completed")

    def _save_checkpoint(self) -> None:
        """Save current checkpoint."""
        save_checkpoint(
            str(self.checkpoint_path),
            self._completed_game_ids,
            [game_result_to_dict(r) for r in self._results],
        )

    def _add_result(self, result: GameResult) -> None:
        """Add a result."""
        self._results.append(result)
        self._completed_game_ids.add(result.game_id)
        if result.success:
            self._success_count += 1
            self._success_steps += result.steps

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

    def _retrieve_memories(self, goal: str) -> list:
        """Retrieve relevant memories for a task goal.

        Args:
            goal: Task goal description.

        Returns:
            List of RetrievedMemory objects.
        """
        if not self.memory_retriever:
            return []

        try:
            # Retrieve memories
            retrieved = self.memory_retriever.retrieve(goal)

            # Display retrieval info
            if retrieved:
                for rm in retrieved:
                    result_tag = Colors.success(
                        "✓") if rm.is_success else Colors.warning("✗")
                    tqdm.write(
                        f"  {Colors.info('📚 Memory:')} {result_tag} "
                        f"sim={rm.similarity:.2f} | {rm.memory_items[0].title if rm.memory_items else 'No title'}"
                    )

            if self.config.runtime.debug and retrieved:
                logger.debug(
                    f"Retrieved {len(retrieved)} memories for goal: {goal[:50]}..."
                )

            return retrieved

        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            return []

    def _extract_and_store_memory(self, result: GameResult) -> None:
        """Extract memory from game result and store it.

        Args:
            result: Game result to extract memory from.
        """
        if not self.memory_extractor or not self.memory_store:
            return

        try:
            # Build trajectory from result
            trajectory = []
            for i, action in enumerate(result.actions):
                obs = result.observations[i + 1] if i + \
                    1 < len(result.observations) else ""
                trajectory.append({
                    "action": action,
                    "observation": obs,
                })

            # Extract memory
            memory = self.memory_extractor.extract(
                task_id=result.game_id,
                task_type=result.task_type,
                goal=result.goal,
                trajectory=trajectory,
                is_success=result.success,
            )

            if memory:
                self.memory_store.add(memory)
                # Display extraction info
                result_tag = Colors.success(
                    "✓") if memory.is_success else Colors.warning("✗")
                item_titles = [item.title for item in memory.memory_items[:2]]
                titles_str = ", ".join(item_titles)
                if len(memory.memory_items) > 2:
                    titles_str += f" +{len(memory.memory_items) - 2}"
                tqdm.write(
                    f"  {Colors.info('💡 Extracted:')} {result_tag} "
                    f"{len(memory.memory_items)} items | {titles_str}"
                )

                if self.config.runtime.debug:
                    logger.debug(
                        f"Extracted and stored memory {memory.memory_id} "
                        f"({len(memory.memory_items)} items)"
                    )

        except Exception as e:
            tqdm.write(f"  {Colors.error('⚠ Extract failed:')} {str(e)[:50]}")
            logger.error(f"Memory extraction failed for {result.game_id}: {e}")
            # Don't propagate - extraction failure shouldn't stop evaluation

    def _run_game(self, game_file: str) -> GameResult:
        """Run a single game with optional memory support."""
        env = None
        try:
            # Create environment and get initial state
            env = AlfWorldEnv(self.config.data.alfworld_data_path)
            obs, info = env.reset(game_file)
            goal = extract_task_description(obs)

            # Retrieve relevant memories using the goal
            retrieved_memories = self._retrieve_memories(goal)

            # Create agent
            agent = ReActAgent(
                llm_client=self.llm_client,
                use_few_shot=self.config.prompt.use_few_shot,
                history_length=self.config.prompt.history_length,
                debug=self.config.runtime.debug,
                retrieved_memories=retrieved_memories,
            )

            # Run game with existing environment
            result = agent.run_game(
                env, obs, info, max_steps=self.config.test.max_steps)

            # Extract and store memory if enabled
            if self.config.memory.should_extract():
                self._extract_and_store_memory(result)

            return result

        except Exception as e:
            logger.error(f"Error running game {game_file}: {e}")
            task_type_id, task_type = get_task_type_from_path(game_file)
            return GameResult(
                game_id=get_game_id_from_path(game_file),
                game_file=game_file,
                task_type=task_type,
                task_type_id=task_type_id,
                success=False,
                steps=0,
                goal="",
                error=str(e),
            )
        finally:
            if env:
                env.close()

    def run(self) -> None:
        """Run the evaluation."""
        # Print header
        print()
        print(Colors.highlight("=" * 60))
        print(Colors.highlight("  ALFWorld Evaluation"))
        print(Colors.highlight("=" * 60))
        print(f"  Model:    {Colors.info(self.config.llm.model)}")
        print(f"  Split:    {Colors.info(self.config.test.split)}")
        print(
            f"  Tasks:    {Colors.info(str(self.config.test.task_types or 'all'))}")
        print(f"  Run ID:   {Colors.dim(self.run_id)}")

        # Print memory info
        if self.config.memory.enabled:
            print(Colors.dim("-" * 40))
            print(f"  Memory:   {Colors.info(self.config.memory.mode)}")
            if self.memory_store:
                stats = self.memory_store.get_stats()
                print(
                    f"  Bank:     {Colors.info(str(stats['total_memories']))} memories")
            else:
                print(f"  Bank:     {Colors.warning('Not initialized')}")

        print(Colors.highlight("=" * 60))
        print()


        self._load_checkpoint()

        game_files = self.get_game_files()
        total_games = len(game_files)

        # Filter out already completed games
        remaining_files = [
            f for f in game_files
            if get_game_id_from_path(f) not in self._completed_game_ids
        ]

        if not remaining_files:
            print(Colors.success("All games already completed!"))
        else:
            print(f"Total games: {Colors.info(str(total_games))}")
            if self._completed_game_ids:
                print(
                    f"{Colors.success('Resuming:')} {len(self._completed_game_ids)} done, "
                    f"{Colors.warning(str(len(remaining_files)))} remaining"
                )
            else:
                print(
                    f"Remaining: {Colors.warning(str(len(remaining_files)))}")
            print()

            completed_since_save = 0

            # Sequential evaluation with progress bar
            with tqdm(
                remaining_files,
                desc="Evaluating",
                unit="game",
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            ) as pbar:
                for game_file in pbar:
                    game_id = get_game_id_from_path(game_file)

                    # Skip if already completed
                    if game_id in self._completed_game_ids:
                        continue

                    try:
                        result = self._run_game(game_file)
                        self._add_result(result)
                        completed_since_save += 1

                        # Log progress
                        completed = len(self._results)
                        progress_str = format_progress(
                            completed, total_games, self._success_count, self._success_steps
                        )
                        result_str = format_game_result(result)
                        tqdm.write(f"{progress_str} | {result_str}")

                        # Save checkpoint periodically
                        if completed_since_save >= self.config.runtime.save_interval:
                            self._save_checkpoint()
                            completed_since_save = 0

                    except Exception as e:
                        logger.error(f"Error processing {game_file}: {e}")

        # Final save
        self._save_checkpoint()

        timestamp = get_timestamp().replace(":", "-")
        final_results_path = self.output_dir / \
            f"{self.run_id}_{timestamp}_results.json"

        save_results(
            results=self._results,
            config_dict=self.config.to_dict(),
            output_path=str(final_results_path),
            model_name=self.config.llm.model,
        )

        save_results(
            results=self._results,
            config_dict=self.config.to_dict(),
            output_path=str(self.results_path),
            model_name=self.config.llm.model,
        )

        # Print summary
        self._print_summary(final_results_path)

    def _print_summary(self, final_results_path: Path) -> None:
        """Print evaluation summary."""
        summary = compute_summary(self._results)

        print()
        print(Colors.highlight("=" * 60))
        print(Colors.highlight("  EVALUATION COMPLETE"))
        print(Colors.highlight("=" * 60))
        print()

        rate_color = (
            Colors.BRIGHT_GREEN if summary["success_rate"] >= 0.7
            else Colors.BRIGHT_YELLOW if summary["success_rate"] >= 0.5
            else Colors.BRIGHT_RED
        )
        print(f"  Total games:     {summary['total_games']}")
        print(
            f"  Successes:       {Colors.success(str(summary['successes']))}")
        print(
            f"  Success rate:    {rate_color}{summary['success_rate']:.2%}{Colors.RESET}")
        print(f"  Avg steps:       {summary['avg_steps']:.1f}")
        print(f"  Success avg:     {summary['success_avg_steps']:.1f}")

        if summary["by_task_type"]:
            print()
            print(Colors.dim("-" * 40))
            print("  By Task Type:")
            for task_type, stats in sorted(summary["by_task_type"].items()):
                type_rate_color = (
                    Colors.BRIGHT_GREEN if stats["success_rate"] >= 0.7
                    else Colors.BRIGHT_YELLOW if stats["success_rate"] >= 0.5
                    else Colors.BRIGHT_RED
                )
                print(
                    f"    {task_type[:30]:30s} "
                    f"{type_rate_color}{stats['successes']:2d}/{stats['total']:2d} "
                    f"({stats['success_rate']:.0%}){Colors.RESET}"
                )

        # Print memory statistics if enabled
        if self.config.memory.enabled and self.memory_store:
            print()
            print(Colors.dim("-" * 40))
            print("  Memory Statistics:")
            stats = self.memory_store.get_stats()
            print(f"    Total memories:   {stats['total_memories']}")
            print(f"    Success memories: {stats['success_memories']}")
            print(f"    Failure memories: {stats['failure_memories']}")

        print()
        print(Colors.highlight("=" * 60))
        print(f"  Results: {Colors.info(str(final_results_path))}")
        print(f"  Checkpoint: {Colors.dim(str(self.checkpoint_path))}")
        if self.memory_store:
            print(
                f"  Memory bank: {Colors.dim(str(self.memory_store.memories_path))}")
        print(Colors.highlight("=" * 60))
        print()


def run_evaluation(config: Config) -> None:
    """Run evaluation with the given configuration."""
    Evaluator(config).run()
