"""Sequential evaluator for ALFWorld with checkpoint support and memory integration."""

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import List, Set, Optional

from tqdm import tqdm

from .config import Config, LLMConfig, RetryConfig
from .llm_client import LLMClient
from .environment import AlfWorldEnv, get_game_id_from_path, get_task_type_from_path
from .agent import GameResult, ReActAgent
from .prompts import get_system_prompt, extract_task_description
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
    log_system_prompt,
    format_progress,
    format_game_result,
)

# Type for trajectory data used in MaTTS
# Contains: trajectory, is_success, total_steps, initial_observation
TrajectoryData = dict

logger = logging.getLogger(__name__)


def generate_run_id(config: Config) -> str:
    """Generate a stable run ID based on configuration.

    All parameters that affect evaluation results should be included in the hash
    to ensure different configurations produce different run IDs.
    """
    key_params = {
        # LLM config
        "model": config.llm.model,
        "temperature": config.llm.temperature,
        # Test config
        "split": config.test.split,
        "task_types": sorted(config.test.task_types) if config.test.task_types else None,
        "num_games": config.test.num_games,
        "seed": config.test.seed,
        "max_steps": config.test.max_steps,
        # Prompt config
        "use_few_shot": config.prompt.use_few_shot,
        "history_length": config.prompt.history_length,
        # Memory config
        "memory_enabled": config.memory.enabled,
        "memory_mode": config.memory.mode,
        # Retrieval parameters
        "memory_top_k": config.memory.top_k,
        "memory_similarity_threshold": config.memory.similarity_threshold,
        # MaTTS config
        "matts_enabled": config.memory.matts.enabled,
        "matts_sample_n": config.memory.matts.sample_n if config.memory.matts.enabled else None,
        "matts_temperature": config.memory.matts.temperature if config.memory.matts.enabled else None,
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
        # Add MaTTS suffix if enabled
        if config.memory.matts.enabled:
            memory_suffix += "_matts"

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

        # Initialize memory components if enabled
        self.memory_store = None
        self.memory_retriever = None
        self.memory_extractor = None
        self.matts_llm_client = None  # Separate LLM client for MaTTS
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

            # Initialize MaTTS-specific LLM client if MaTTS is enabled
            if self.config.memory.matts.enabled:
                self._init_matts_client()

            logger.info(
                f"Memory system initialized: mode={self.config.memory.mode}, "
                f"store_size={self.memory_store.size()}, "
                f"matts={self.config.memory.matts.enabled}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize memory system: {e}")
            logger.warning("Falling back to baseline mode (no memory)")
            self.memory_store = None
            self.memory_retriever = None
            self.memory_extractor = None
            self.matts_llm_client = None

    def _init_matts_client(self) -> None:
        """Initialize separate LLM client for MaTTS with different parameters."""
        matts_config = self.config.memory.matts

        # Create LLM config for MaTTS with higher temperature
        matts_llm_config = LLMConfig(
            api_base_url=self.config.llm.api_base_url,
            api_key=self.config.llm.api_key,
            model=self.config.llm.model,
            temperature=matts_config.temperature,  # Higher temperature for diversity
            max_tokens=matts_config.max_tokens,
            timeout=self.config.llm.timeout,
            enable_thinking=matts_config.enable_thinking,  # MaTTS-specific thinking mode
        )

        self.matts_llm_client = LLMClient(matts_llm_config, self.config.retry)

        logger.info(
            f"MaTTS LLM client initialized: "
            f"sample_n={matts_config.sample_n}, "
            f"temperature={matts_config.temperature}, "
            f"enable_thinking={matts_config.enable_thinking}"
        )

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
                    # Show reference success rate if memory has been referenced before
                    ref_info = ""
                    if rm.reference_count > 0:
                        ref_rate = rm.reference_success_rate
                        rate_color = (
                            Colors.BRIGHT_GREEN if ref_rate >= 0.7
                            else Colors.BRIGHT_YELLOW if ref_rate >= 0.4
                            else Colors.BRIGHT_RED
                        )
                        ref_info = f" | {rate_color}refs:{rm.reference_count} sr:{ref_rate:.0%}{Colors.RESET}"
                    tqdm.write(
                        f"  {Colors.info('📚 Memory:')} {result_tag} "
                        f"sim={rm.similarity:.2f}{ref_info} | "
                        f"{rm.memory_items[0].title if rm.memory_items else 'No title'}"
                    )

            if self.config.runtime.debug and retrieved:
                logger.debug(
                    f"Retrieved {len(retrieved)} memories for goal: {goal[:50]}..."
                )

            return retrieved

        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            return []

    def _build_trajectory_data(self, result: GameResult) -> TrajectoryData:
        """Build trajectory data dict from game result.

        Args:
            result: Game result to convert.

        Returns:
            Trajectory data dict with trajectory, is_success, total_steps, initial_observation.
        """
        trajectory = []
        for i, action in enumerate(result.actions):
            obs = result.observations[i + 1] if i + \
                1 < len(result.observations) else ""
            trajectory.append({
                "action": action,
                "observation": obs,
            })

        return {
            "trajectory": trajectory,
            "is_success": result.success,
            "total_steps": result.steps,
            "initial_observation": result.observations[0] if result.observations else "",
        }

    def _extract_and_store_memory(self, result: GameResult) -> None:
        """Extract memory from game result and store it.

        Args:
            result: Game result to extract memory from.
        """
        if not self.memory_extractor or not self.memory_store:
            return

        try:
            traj_data = self._build_trajectory_data(result)

            # Extract memory
            memory = self.memory_extractor.extract(
                task_id=result.game_id,
                task_type=result.task_type,
                goal=result.goal,
                trajectory=traj_data["trajectory"],
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

    def _run_matts_sampling(
        self,
        game_file: str,
        goal: str,
        retrieved_memories: list,
    ) -> List[TrajectoryData]:
        """Run multiple sampling attempts for MaTTS.

        Args:
            game_file: Path to game file.
            goal: Task goal description.
            retrieved_memories: Retrieved memories to use for all samples.

        Returns:
            List of trajectory data dicts from all samples.
        """
        sample_n = self.config.memory.matts.sample_n
        trajectories: List[TrajectoryData] = []

        tqdm.write(
            f"  {Colors.info('🎲 MaTTS:')} Sampling {sample_n} trajectories...")

        for i in range(sample_n):
            env = None
            try:
                # Create fresh environment for each sample
                env = AlfWorldEnv(self.config.data.alfworld_data_path)
                obs, info = env.reset(game_file)

                # Create agent with MaTTS LLM client (higher temperature)
                agent = ReActAgent(
                    llm_client=self.matts_llm_client or self.llm_client,
                    use_few_shot=self.config.prompt.use_few_shot,
                    history_length=self.config.prompt.history_length,
                    debug=False,  # Don't flood debug log with MaTTS samples
                    retrieved_memories=retrieved_memories,
                )

                # Run game
                result = agent.run_game(
                    env, obs, info, max_steps=self.config.test.max_steps
                )

                # Build trajectory data
                traj_data = self._build_trajectory_data(result)
                trajectories.append(traj_data)

                # Display sampling progress
                result_tag = Colors.success(
                    "✓") if result.success else Colors.warning("✗")
                tqdm.write(
                    f"    Sample {i+1}/{sample_n}: {result_tag} "
                    f"steps={result.steps}"
                )

            except Exception as e:
                logger.error(f"MaTTS sample {i+1} failed: {e}")
                tqdm.write(
                    f"    Sample {i+1}/{sample_n}: {Colors.error('ERROR')} {str(e)[:30]}")

            finally:
                if env:
                    env.close()

        return trajectories

    def _run_matts_extraction(
        self,
        result: GameResult,
        all_trajectories: List[TrajectoryData],
    ) -> None:
        """Run MaTTS contrastive extraction from multiple trajectories.

        Args:
            result: Primary game result (for metadata).
            all_trajectories: List of all trajectory data including the main one.
        """
        if not self.memory_extractor or not self.memory_store:
            return

        if len(all_trajectories) < 2:
            tqdm.write(
                f"  {Colors.warning('⚠ MaTTS:')} Not enough trajectories for contrastive extraction")
            # Fall back to single trajectory extraction
            self._extract_and_store_memory(result)
            return

        try:
            # Display MaTTS extraction info
            num_success = sum(
                1 for t in all_trajectories if t.get("is_success", False))
            num_failed = len(all_trajectories) - num_success
            tqdm.write(
                f"  {Colors.info('🔍 MaTTS Extraction:')} "
                f"{len(all_trajectories)} trajectories "
                f"({Colors.success(str(num_success) + '✓')} {Colors.warning(str(num_failed) + '✗')})"
            )

            # Run contrastive extraction with MaTTS LLM client
            memory = self.memory_extractor.extract_contrastive(
                task_id=result.game_id,
                task_type=result.task_type,
                goal=result.goal,
                trajectories=all_trajectories,
                llm_client=self.matts_llm_client,  # Use MaTTS client for extraction
            )

            if memory:
                self.memory_store.add(memory)
                # Display extraction result
                item_titles = [item.title for item in memory.memory_items[:2]]
                titles_str = ", ".join(item_titles)
                if len(memory.memory_items) > 2:
                    titles_str += f" +{len(memory.memory_items) - 2}"
                tqdm.write(
                    f"  {Colors.success('✨ MaTTS Result:')} "
                    f"{len(memory.memory_items)} items | {titles_str}"
                )

                if self.config.runtime.debug:
                    logger.debug(
                        f"MaTTS extracted memory {memory.memory_id} "
                        f"({len(memory.memory_items)} items from "
                        f"{len(all_trajectories)} trajectories)"
                    )
            else:
                tqdm.write(
                    f"  {Colors.warning('⚠ MaTTS:')} No valid items extracted")

        except Exception as e:
            tqdm.write(f"  {Colors.error('⚠ MaTTS failed:')} {str(e)[:50]}")
            logger.error(f"MaTTS extraction failed for {result.game_id}: {e}")
            # Fall back to single trajectory extraction
            self._extract_and_store_memory(result)

    def _run_game(self, game_file: str) -> GameResult:
        """Run a single game with optional memory and MaTTS support."""
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

            # Close the main environment before MaTTS sampling
            env.close()
            env = None

            # Update reference statistics for used memories
            if retrieved_memories and self.memory_store:
                memory_ids = [rm.memory_id for rm in retrieved_memories]
                self.memory_store.update_reference_stats(
                    memory_ids, result.success)

            # Handle memory extraction
            if self.config.memory.should_extract():
                if self.config.memory.matts.enabled:
                    # MaTTS mode: sample multiple trajectories
                    main_traj_data = self._build_trajectory_data(result)

                    # Run additional samples
                    matts_trajectories = self._run_matts_sampling(
                        game_file, goal, retrieved_memories
                    )

                    # Combine main trajectory with MaTTS samples
                    all_trajectories = [main_traj_data] + matts_trajectories

                    # Run contrastive extraction
                    self._run_matts_extraction(result, all_trajectories)
                else:
                    # Normal single-trajectory extraction
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

            # Print MaTTS info
            if self.config.memory.matts.enabled:
                matts = self.config.memory.matts
                print(Colors.dim("-" * 40))
                print(f"  {Colors.highlight('MaTTS:')}")
                print(f"    Samples:     {Colors.info(str(matts.sample_n))}")
                print(
                    f"    Temperature: {Colors.info(str(matts.temperature))}")
                if matts.enable_thinking is not None:
                    thinking_str = "enabled" if matts.enable_thinking else "disabled"
                    print(f"    Thinking:    {Colors.info(thinking_str)}")

        print(Colors.highlight("=" * 60))
        print()

        if self.config.runtime.debug:
            log_system_prompt(get_system_prompt(
                self.config.prompt.use_few_shot))

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

            # Reference statistics
            if stats['total_references'] > 0:
                ref_rate = stats['overall_reference_success_rate']
                ref_rate_color = (
                    Colors.BRIGHT_GREEN if ref_rate >= 0.7
                    else Colors.BRIGHT_YELLOW if ref_rate >= 0.4
                    else Colors.BRIGHT_RED
                )
                print(
                    f"    Referenced:       {stats['referenced_memories']} memories")
                print(f"    Total refs:       {stats['total_references']}")
                print(
                    f"    Ref success rate: {ref_rate_color}{ref_rate:.1%}{Colors.RESET}"
                )

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
