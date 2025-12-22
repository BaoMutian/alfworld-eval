"""ReAct Agent for ALFWorld evaluation."""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING

from .llm_client import LLMClient
from .environment import AlfWorldEnv
from .prompts import (
    get_system_prompt_with_memory,
    build_user_prompt,
    build_memory_section,
    extract_task_description,
)
from .logging_utils import (
    Colors,
    log_game_start,
    log_game_end,
    log_step_interaction,
    format_step_info,
)

if TYPE_CHECKING:
    from .memory import RetrievedMemory

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
    goal: str = ""
    actions: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    thoughts: List[str] = field(default_factory=list)
    error: Optional[str] = None
    # Memory-related fields
    used_memories: List[Dict[str, Any]] = field(default_factory=list)


class ReActAgent:
    """ReAct-style agent for ALFWorld tasks."""

    # Action keywords for fallback parsing
    ACTION_KEYWORDS = (
        "go to", "take", "move", "open", "close", "use",
        "heat", "cool", "clean", "look", "examine", "inventory",
        "check valid actions"
    )

    def __init__(
        self,
        llm_client: LLMClient,
        use_few_shot: bool = True,
        history_length: int = 10,
        debug: bool = False,
        retrieved_memories: Optional[List["RetrievedMemory"]] = None,
    ):
        """Initialize ReAct agent.
        
        Args:
            llm_client: LLM client for generating responses.
            use_few_shot: Whether to include few-shot examples.
            history_length: Number of history entries to include.
            debug: Whether to enable debug logging.
            retrieved_memories: Optional list of retrieved memories to use.
        """
        self.llm_client = llm_client
        self.history_length = history_length
        self.debug = debug
        self.retrieved_memories = retrieved_memories or []
        
        # Build system prompt with optional memories
        self.system_prompt = get_system_prompt_with_memory(
            use_few_shot=use_few_shot,
            retrieved_memories=self.retrieved_memories,
        )

    def parse_response(self, response: str) -> Tuple[str, str]:
        """Parse LLM response to extract thought and action."""
        thought = ""
        action = ""

        # Extract Think section
        think_match = re.search(
            r"Think(?:ing)?:\s*(.+?)(?=Action:|$)",
            response, re.DOTALL | re.IGNORECASE
        )
        if think_match:
            thought = think_match.group(1).strip()

        # Extract Action section
        action_match = re.search(
            r"Action:\s*(.+?)(?=Think|Thought|$)",
            response, re.DOTALL | re.IGNORECASE
        )
        if action_match:
            action = action_match.group(1).strip().split("\n")[0].strip()

        # Fallback: look for action-like lines
        if not action:
            for line in response.split("\n"):
                line_lower = line.lower().strip()
                if any(line_lower.startswith(kw) for kw in self.ACTION_KEYWORDS):
                    action = line.strip()
                    break

        # Last resort: use last non-empty line
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
        """Run a single game with the agent."""
        task_description = extract_task_description(initial_obs)

        # Record used memories
        used_memories = [rm.get_summary() for rm in self.retrieved_memories]

        result = GameResult(
            game_id=info["game_id"],
            game_file=env.current_game_path,
            task_type=info["task_type"],
            task_type_id=info["task_type_id"],
            success=False,
            steps=0,
            goal=task_description,
            used_memories=used_memories,
        )

        history: List[Tuple[str, str]] = []
        current_obs = initial_obs
        result.observations.append(current_obs)

        if self.debug:
            memory_section = build_memory_section(self.retrieved_memories) if self.retrieved_memories else ""
            log_game_start(info["game_id"], task_description, memory_section)
            print(f"\n{Colors.info('Game:')} {info['game_id']}")
            print(f"{Colors.dim('Goal:')} {task_description}")
            if self.retrieved_memories:
                print(f"{Colors.dim('Using')} {Colors.info(str(len(self.retrieved_memories)))} {Colors.dim('retrieved memories')}")

        try:
            for step in range(max_steps):
                user_prompt = build_user_prompt(
                    task_description=task_description,
                    history=history,
                    current_observation=current_obs,
                    history_length=self.history_length,
                )

                response = self.llm_client.chat_simple(
                    system_prompt=self.system_prompt,
                    user_prompt=user_prompt,
                )

                thought, action = self.parse_response(response)
                result.thoughts.append(thought)
                result.actions.append(action)

                obs, reward, done, step_info = env.step(action)
                result.observations.append(obs)

                if self.debug:
                    log_step_interaction(
                        step + 1, user_prompt, response, action, obs)
                    print(format_step_info(step + 1, action, obs))

                history.append((action, obs))
                current_obs = obs
                result.steps = step + 1

                if step_info.get("won", False):
                    result.success = True
                    if self.debug:
                        print(f"  {Colors.success('>>> Task completed!')}")
                    break

                if done:
                    if self.debug:
                        print(f"  {Colors.warning('>>> Game ended (not won)')}")
                    break

        except Exception as e:
            result.error = str(e)
            logger.error(f"Error during game {info['game_id']}: {e}")

        if self.debug:
            log_game_end(info["game_id"], result.success, result.steps)

        return result
