"""ALFWorld environment wrapper using textworld.gym interface."""

import os
import sys
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import textworld
import textworld.gym

logger = logging.getLogger(__name__)


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout (for PDDL planner output)."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# Task type mapping
TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}

TASK_TYPE_TO_ID = {v: k for k, v in TASK_TYPES.items()}


def get_task_type_from_path(game_path: str) -> Tuple[int, str]:
    """Extract task type from game path.

    Args:
        game_path: Path to game file.

    Returns:
        Tuple of (task_type_id, task_type_name).
    """
    path_str = str(game_path)
    for task_id, task_name in TASK_TYPES.items():
        if task_name in path_str:
            return task_id, task_name

    # Handle additional task type (pick_and_place_with_movable_recep)
    if "pick_and_place_with_movable_recep" in path_str:
        return 1, "pick_and_place_with_movable_recep"

    return 0, "unknown"


def get_game_id_from_path(game_path: str) -> str:
    """Extract game ID from game path.

    Args:
        game_path: Path to game file.

    Returns:
        Game ID string.
    """
    # Extract the task folder name and trial folder
    # e.g., pick_and_place_simple-Book-None-SideTable-329/trial_T20190908_050633_745514
    parts = Path(game_path).parts
    for i, part in enumerate(parts):
        if any(task in part for task in TASK_TYPES.values()) or "pick_and_place_with_movable_recep" in part:
            if i + 1 < len(parts) and parts[i + 1].startswith("trial_"):
                return f"{part}/{parts[i + 1]}"
            return part
    return Path(game_path).stem


class AlfredDemangler(textworld.core.Wrapper):
    """Wrapper to demangle Alfred object names.
    
    This is a fixed version of the original ALFWorld demangler that handles
    duplicate object IDs correctly.
    """

    def load(self, *args, **kwargs):
        super().load(*args, **kwargs)
        self._demangle_names()

    def _demangle_names(self):
        """Demangle Alfred object names to human-readable format."""
        # Collect all unique IDs first
        ids = sorted(set(info.id for info in self._entity_infos.values()))
        
        # Count object types (fixed: count unique IDs only)
        obj_count = {}
        for obj_id in ids:
            splits = obj_id.split("_bar_", 1)
            if len(splits) > 1:
                name = splits[0]
                if "basin" in obj_id:
                    name += "basin"
                obj_count[name] = obj_count.get(name, 0) + 1

        # Create ID assignment (1-indexed)
        obj_num_ids = {name: list(range(1, count + 1)) for name, count in obj_count.items()}
        
        # Build name mapping
        obj_names = {}
        for obj_id in ids:
            text = obj_id.replace("_bar_", "|").replace("_minus_", "-")
            text = text.replace("_plus_", "+").replace("_dot_", ".").replace("_comma_", ",")
            
            splits = text.split("|", 1)
            if len(splits) == 1:
                obj_names[obj_id] = f"{text}"
            else:
                name = splits[0]
                if "basin" in obj_id:
                    name += "basin"
                num_id = obj_num_ids[name].pop(0)  # pop from front for consistent ordering
                obj_names[obj_id] = f"{name} {num_id}"

        # Apply demangled names
        for info in self._entity_infos.values():
            if info.id in obj_names:
                info.name = obj_names[info.id]


class AlfWorldEnv:
    """Wrapper for ALFWorld TextWorld environment."""

    # Special action for getting valid actions
    CHECK_VALID_ACTIONS = "check valid actions"

    def __init__(self, alfworld_data_path: str):
        """Initialize ALFWorld environment.

        Args:
            alfworld_data_path: Path to ALFWorld data directory.
        """
        self.alfworld_data_path = Path(alfworld_data_path).absolute()
        self.env = None
        self.env_id = None
        self.current_game_path = None
        self.current_game_id = None
        self.current_task_type = None
        self.current_task_type_id = None
        self.admissible_commands = []

        # Load game logic files
        self.domain_path = self.alfworld_data_path / "logic" / "alfred.pddl"
        self.grammar_path = self.alfworld_data_path / "logic" / "alfred.twl2"

        if not self.domain_path.exists():
            raise FileNotFoundError(
                f"Domain file not found: {self.domain_path}")
        if not self.grammar_path.exists():
            raise FileNotFoundError(
                f"Grammar file not found: {self.grammar_path}")

        # Set environment variable
        os.environ["ALFWORLD_DATA"] = str(self.alfworld_data_path)

    def get_game_files(self, split: str, task_types: Optional[List[int]] = None) -> List[str]:
        """Get list of game files for a split.

        Args:
            split: Dataset split (valid_seen, valid_train, valid_unseen, train).
            task_types: Optional list of task type IDs to filter.

        Returns:
            List of game file paths.
        """
        split_path = self.alfworld_data_path / "json_2.1.1" / split
        if not split_path.exists():
            raise ValueError(f"Split path does not exist: {split_path}")

        game_files = []
        for task_dir in split_path.iterdir():
            if not task_dir.is_dir():
                continue

            # Skip movable receptacle tasks (not supported)
            if "movable_recep" in task_dir.name or "Sliced" in task_dir.name:
                continue

            # Check task type filter
            if task_types is not None:
                task_type_id, _ = get_task_type_from_path(str(task_dir))
                if task_type_id not in task_types:
                    continue

            # Find game files in trial directories
            for trial_dir in task_dir.iterdir():
                if trial_dir.is_dir() and trial_dir.name.startswith("trial_"):
                    game_file = trial_dir / "game.tw-pddl"
                    if game_file.exists():
                        # Check if game is solvable
                        try:
                            with open(game_file, 'r') as f:
                                gamedata = json.load(f)
                            if gamedata.get('solvable', True):
                                game_files.append(str(game_file))
                        except (json.JSONDecodeError, KeyError):
                            # If we can't verify, include it anyway
                            game_files.append(str(game_file))

        return sorted(game_files)

    def reset(self, game_path: str) -> Tuple[str, Dict[str, Any]]:
        """Reset environment with a specific game.

        Args:
            game_path: Path to game.tw-pddl file.

        Returns:
            Tuple of (initial_observation, info_dict).
        """
        # Close previous environment if exists
        self.close()

        self.current_game_path = game_path
        self.current_game_id = get_game_id_from_path(game_path)
        self.current_task_type_id, self.current_task_type = get_task_type_from_path(
            game_path)

        # Register the game with textworld
        request_infos = textworld.EnvInfos(
            won=True,
            admissible_commands=True,
            score=True,
            max_score=True,
        )

        # Suppress PDDL planner output during game registration
        with suppress_stdout():
            self.env_id = textworld.gym.register_game(
                game_path,
                request_infos,
                max_episode_steps=1000,
                wrappers=[AlfredDemangler()],
            )

            # Create environment
            self.env = textworld.gym.make(self.env_id)

            # Reset and get initial observation
            obs, infos = self.env.reset()

        # Store admissible commands
        self.admissible_commands = infos.get("admissible_commands", [])

        return obs, {
            "admissible_commands": self.admissible_commands,
            "won": False,
            "game_id": self.current_game_id,
            "task_type": self.current_task_type,
            "task_type_id": self.current_task_type_id,
        }

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute action in environment.

        Args:
            action: Action string to execute.

        Returns:
            Tuple of (observation, reward, done, info).
        """
        # Handle special "check valid actions" command
        if action.lower().strip() == self.CHECK_VALID_ACTIONS:
            valid_actions_str = "Valid actions:\n" + \
                "\n".join(f"  - {cmd}" for cmd in self.admissible_commands)
            return valid_actions_str, 0, False, {
                "admissible_commands": self.admissible_commands,
                "won": False,
            }

        # Execute action in environment
        obs, score, done, infos = self.env.step(action)

        # Update admissible commands
        self.admissible_commands = infos.get("admissible_commands", [])

        # Check if won
        won = infos.get("won", False)

        return obs, score, done, {
            "admissible_commands": self.admissible_commands,
            "won": won,
        }

    def close(self):
        """Close the environment."""
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None
            self.env_id = None


def create_env_for_game(alfworld_data_path: str, game_path: str) -> Tuple[AlfWorldEnv, str, Dict]:
    """Create and reset environment for a specific game.

    This is a helper function for parallel execution where each worker
    needs its own environment instance.

    Args:
        alfworld_data_path: Path to ALFWorld data.
        game_path: Path to game file.

    Returns:
        Tuple of (env, initial_observation, info).
    """
    env = AlfWorldEnv(alfworld_data_path)
    obs, info = env.reset(game_path)
    return env, obs, info
