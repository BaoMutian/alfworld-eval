"""ReAct Agent for ALFWorld evaluation."""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

from .llm_client import LLMClient
from .environment import AlfWorldEnv
from .prompts import get_system_prompt, build_user_prompt
from .prompts.system import extract_task_description
from .logging_utils import (
    Colors,
    log_game_start,
    log_game_end,
    log_step_interaction,
    format_step_info,
)

logger = logging.getLogger(__name__)


@dataclass
class GameResult:
    """Result of a single game run."""
    game_id: str
    game_file: str
    task_type: str
    task_type_id: int
    success: bool
    steps: int
    goal: str = ""  # Task goal description
    actions: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    thoughts: List[str] = field(default_factory=list)
    error: Optional[str] = None


class ReActAgent:
    """ReAct-style agent for ALFWorld tasks."""

    def __init__(
        self,
        llm_client: LLMClient,
        use_few_shot: bool = True,
        history_length: int = 10,
        debug: bool = False,
    ):
        """Initialize ReAct agent.

        Args:
            llm_client: LLM client for generating responses.
            use_few_shot: Whether to include few-shot examples in system prompt.
            history_length: Number of recent history entries to include.
            debug: Whether to enable debug mode.
        """
        self.llm_client = llm_client
        self.use_few_shot = use_few_shot
        self.history_length = history_length
        self.debug = debug
        # Get system prompt (with or without few-shot examples)
        self.system_prompt = get_system_prompt(use_few_shot)

    def parse_response(self, response: str) -> Tuple[str, str]:
        """Parse LLM response to extract thought and action.

        Args:
            response: Raw LLM response.

        Returns:
            Tuple of (thought, action).
        """
        thought = ""
        action = ""

        # Try to extract Think section
        think_patterns = [
            r"Think:\s*(.+?)(?=Action:|$)",
            r"THINK:\s*(.+?)(?=ACTION:|$)",
            r"Thought:\s*(.+?)(?=Action:|$)",
        ]

        for pattern in think_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                thought = match.group(1).strip()
                break

        # Try to extract Action section
        action_patterns = [
            r"Action:\s*(.+?)(?=Think:|Thought:|$)",
            r"ACTION:\s*(.+?)(?=THINK:|THOUGHT:|$)",
        ]

        for pattern in action_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                action = match.group(1).strip()
                # Take only the first line of action (in case there's more text)
                action = action.split("\n")[0].strip()
                break

        # If no structured format found, try to find any action-like text
        if not action:
            # Look for common action patterns
            action_keywords = [
                "go to", "take", "move", "open", "close", "use",
                "heat", "cool", "clean", "look", "examine", "inventory",
                "check valid actions"
            ]
            lines = response.split("\n")
            for line in lines:
                line_lower = line.lower().strip()
                for keyword in action_keywords:
                    if line_lower.startswith(keyword):
                        action = line.strip()
                        break
                if action:
                    break

        # If still no action, use the last line as a fallback
        if not action:
            lines = [l.strip() for l in response.split("\n") if l.strip()]
            if lines:
                action = lines[-1]

        return thought, action

    def run_game(
        self,
        env: AlfWorldEnv,
        initial_obs: str,
        info: Dict[str, Any],
        max_steps: int = 30,
    ) -> GameResult:
        """Run a single game with the agent.

        Args:
            env: Initialized ALFWorld environment.
            initial_obs: Initial observation from env.reset().
            info: Info dict from env.reset().
            max_steps: Maximum steps allowed.

        Returns:
            GameResult with trajectory and outcome.
        """
        # Extract task description
        task_description = extract_task_description(initial_obs)

        # Initialize result
        result = GameResult(
            game_id=info["game_id"],
            game_file=env.current_game_path,
            task_type=info["task_type"],
            task_type_id=info["task_type_id"],
            success=False,
            steps=0,
            goal=task_description,  # Save the goal
        )

        # Initialize history and current observation
        history: List[Tuple[str, str]] = []
        current_obs = initial_obs
        result.observations.append(current_obs)

        # Log game start to debug file
        if self.debug:
            log_game_start(info["game_id"], task_description)
            # Terminal: show game start
            print(f"\n{Colors.info('Game:')} {info['game_id']}")
            print(f"{Colors.dim('Goal:')} {task_description}")

        try:
            for step in range(max_steps):
                # Build user prompt (few-shot is now in system prompt)
                user_prompt = build_user_prompt(
                    task_description=task_description,
                    history=history,
                    current_observation=current_obs,
                    history_length=self.history_length,
                )

                # Get LLM response
                response = self.llm_client.chat_simple(
                    system_prompt=self.system_prompt,
                    user_prompt=user_prompt,
                )

                # Parse response
                thought, action = self.parse_response(response)

                result.thoughts.append(thought)
                result.actions.append(action)

                # Execute action
                obs, reward, done, step_info = env.step(action)
                result.observations.append(obs)

                # Log to debug file (user prompt and response only)
                if self.debug:
                    log_step_interaction(
                        step=step + 1,
                        user_prompt=user_prompt,
                        response=response,
                        action=action,
                        observation=obs,
                    )
                    # Terminal: show concise step info
                    print(format_step_info(step + 1, action, obs))

                # Update history
                history.append((action, obs))
                current_obs = obs
                result.steps = step + 1

                # Check if won
                if step_info.get("won", False):
                    result.success = True
                    if self.debug:
                        print(f"  {Colors.success('>>> Task completed!')}")
                    break

                # Check if done (but not won)
                if done and not step_info.get("won", False):
                    if self.debug:
                        print(f"  {Colors.warning('>>> Game ended (not won)')}")
                    break

        except Exception as e:
            result.error = str(e)
            logger.error(f"Error during game {info['game_id']}: {e}")

        # Log game end to debug file
        if self.debug:
            log_game_end(info["game_id"], result.success, result.steps)

        return result


def run_single_game(
    alfworld_data_path: str,
    game_file: str,
    llm_client: LLMClient,
    use_few_shot: bool = True,
    history_length: int = 10,
    max_steps: int = 30,
    debug: bool = False,
) -> GameResult:
    """Run a single game from scratch.

    This function creates its own environment and agent, making it suitable
    for parallel execution where each worker needs independent instances.

    Args:
        alfworld_data_path: Path to ALFWorld data.
        game_file: Path to game.tw-pddl file.
        llm_client: LLM client (can be shared across workers).
        use_few_shot: Whether to use few-shot examples in system prompt.
        history_length: History length for prompts.
        max_steps: Maximum steps per game.
        debug: Debug mode.

    Returns:
        GameResult with trajectory and outcome.
    """
    env = None
    try:
        # Create environment
        env = AlfWorldEnv(alfworld_data_path)
        obs, info = env.reset(game_file)

        # Create agent
        agent = ReActAgent(
            llm_client=llm_client,
            use_few_shot=use_few_shot,
            history_length=history_length,
            debug=debug,
        )

        # Run game
        result = agent.run_game(env, obs, info, max_steps=max_steps)
        return result

    except Exception as e:
        logger.error(f"Error running game {game_file}: {e}")
        # Return error result
        from .environment import get_game_id_from_path, get_task_type_from_path
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
        if env is not None:
            env.close()
