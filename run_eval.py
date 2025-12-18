#!/usr/bin/env python3
"""Main entry point for ALFWorld evaluation."""

from src.utils import setup_logging
from src.evaluator import run_evaluation
from src.config import load_config, Config
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ALFWorld LLM Agent Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python run_eval.py
  
  # Run with custom config
  python run_eval.py --config config/my_config.yaml
  
  # Override specific settings
  python run_eval.py --model gpt-4 --split valid_unseen --workers 8
  
  # Debug mode
  python run_eval.py --debug --num-games 5
""",
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML config file (default: config/default.yaml)",
    )

    # LLM settings
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model name to use (overrides config)",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="API base URL (overrides config)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (overrides config and env var)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (overrides config)",
    )

    # Test settings
    parser.add_argument(
        "--num-games", "-n",
        type=int,
        default=None,
        help="Number of games to evaluate (overrides config)",
    )
    parser.add_argument(
        "--split", "-s",
        type=str,
        choices=["valid_seen", "valid_train", "valid_unseen", "train"],
        default=None,
        help="Dataset split to use (overrides config)",
    )
    parser.add_argument(
        "--task-types", "-t",
        type=int,
        nargs="+",
        default=None,
        help="Task types to evaluate (1-6, overrides config)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum steps per game (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )

    # Prompt settings
    parser.add_argument(
        "--no-few-shot",
        action="store_true",
        help="Disable few-shot examples",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=None,
        help="Number of history entries to include (overrides config)",
    )

    # Runtime settings
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Number of parallel workers (overrides config)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for results (overrides config)",
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug mode",
    )

    # Data settings
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to ALFWorld data directory (overrides config)",
    )

    return parser.parse_args()


def apply_overrides(config: Config, args) -> Config:
    """Apply command line overrides to config.

    Args:
        config: Base configuration.
        args: Parsed command line arguments.

    Returns:
        Updated configuration.
    """
    # LLM overrides
    if args.model is not None:
        config.llm.model = args.model
    if args.api_base is not None:
        config.llm.api_base_url = args.api_base
    if args.api_key is not None:
        config.llm.api_key = args.api_key
    if args.temperature is not None:
        config.llm.temperature = args.temperature

    # Test overrides
    if args.num_games is not None:
        config.test.num_games = args.num_games
    if args.split is not None:
        config.test.split = args.split
    if args.task_types is not None:
        config.test.task_types = args.task_types
    if args.max_steps is not None:
        config.test.max_steps = args.max_steps
    if args.seed is not None:
        config.test.seed = args.seed

    # Prompt overrides
    if args.no_few_shot:
        config.prompt.use_few_shot = False
    if args.history_length is not None:
        config.prompt.history_length = args.history_length

    # Runtime overrides
    if args.workers is not None:
        config.runtime.parallel_workers = args.workers
    if args.output_dir is not None:
        config.runtime.output_dir = args.output_dir
    if args.debug:
        config.runtime.debug = True

    # Data overrides
    if args.data_path is not None:
        config.data.alfworld_data_path = args.data_path

    return config


def main():
    """Main entry point."""
    args = parse_args()

    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        print("Using default configuration...")
        config = Config()

    # Apply command line overrides
    config = apply_overrides(config, args)

    # Validate config (this will also set API key from env if not provided)
    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

    # Setup logging (basic setup, debug log will be set by evaluator with run_id)
    setup_logging(debug=config.runtime.debug, log_file=None)

    # Run evaluation
    try:
        run_evaluation(config)
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        print("Progress has been saved to checkpoint.")
        sys.exit(130)
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        if config.runtime.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
