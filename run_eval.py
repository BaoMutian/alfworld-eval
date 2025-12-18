#!/usr/bin/env python3
"""Main entry point for ALFWorld evaluation."""

import argparse
import sys

from src.config import load_config, Config
from src.evaluator import run_evaluation
from src.logging_utils import setup_logging


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
    parser.add_argument("--model", "-m", type=str, help="Model name")
    parser.add_argument("--api-base", type=str, help="API base URL")
    parser.add_argument("--api-key", type=str, help="API key")
    parser.add_argument("--temperature", type=float,
                        help="Sampling temperature")

    # Test settings
    parser.add_argument("--num-games", "-n", type=int, help="Number of games")
    parser.add_argument(
        "--split", "-s",
        type=str,
        choices=["valid_seen", "valid_train", "valid_unseen", "train"],
        help="Dataset split",
    )
    parser.add_argument("--task-types", "-t", type=int,
                        nargs="+", help="Task types (1-6)")
    parser.add_argument("--max-steps", type=int, help="Max steps per game")
    parser.add_argument("--seed", type=int, help="Random seed")

    # Prompt settings
    parser.add_argument("--no-few-shot", action="store_true",
                        help="Disable few-shot examples")
    parser.add_argument("--history-length", type=int,
                        help="History entries to include")

    # Runtime settings
    parser.add_argument("--workers", "-w", type=int, help="Parallel workers")
    parser.add_argument("--output-dir", "-o", type=str,
                        help="Output directory")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Enable debug mode")

    # Data settings
    parser.add_argument("--data-path", type=str,
                        help="ALFWorld data directory")

    return parser.parse_args()


def apply_overrides(config: Config, args) -> Config:
    """Apply command line overrides to config."""
    # LLM overrides
    if args.model:
        config.llm.model = args.model
    if args.api_base:
        config.llm.api_base_url = args.api_base
    if args.api_key:
        config.llm.api_key = args.api_key
    if args.temperature is not None:
        config.llm.temperature = args.temperature

    # Test overrides
    if args.num_games is not None:
        config.test.num_games = args.num_games
    if args.split:
        config.test.split = args.split
    if args.task_types:
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
    if args.output_dir:
        config.runtime.output_dir = args.output_dir
    if args.debug:
        config.runtime.debug = True

    # Data overrides
    if args.data_path:
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

    config = apply_overrides(config, args)

    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

    # Setup basic logging (debug log file set by evaluator)
    setup_logging(debug=config.runtime.debug)

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
