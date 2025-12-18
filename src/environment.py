"""ALFWorld environment wrapper."""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

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


class AlfWorldEnv:
    """Wrapper for ALFWorld TextWorld environment."""

    # Special action for getting valid actions
    CHECK_VALID_ACTIONS = "check valid actions"

    def __init__(self, alfworld_data_path: str):
        """Initialize ALFWorld environment.
        
        Args:
            alfworld_data_path: Path to ALFWorld data directory.
        """
        self.alfworld_data_path = Path(alfworld_data_path)
        self.env = None
        self.current_game_path = None
        self.current_game_id = None
        self.current_task_type = None
        self.current_task_type_id = None
        self.admissible_commands = []
        self._setup_alfworld_env()

    def _setup_alfworld_env(self):
        """Setup ALFWorld environment variable."""
        os.environ["ALFWORLD_DATA"] = str(self.alfworld_data_path.absolute())

    def _load_environment(self, game_path: str):
        """Load a specific game environment.
        
        Args:
            game_path: Path to game.tw-pddl file.
        """
        import alfworld.agents.environment as environment
        
        # Create environment config
        config = {
            "env": {
                "type": "AlfredTWEnv",
                "regen_game_files": False,
                "domain_randomization": False,
                "task_types": [1, 2, 3, 4, 5, 6],
                "expert_timeout_steps": 150,
                "expert_type": "handcoded",
            },
            "logic": {
                "domain": str(self.alfworld_data_path / "logic" / "alfred.pddl"),
                "grammar": str(self.alfworld_data_path / "logic" / "alfred.twl2"),
            },
            "general": {
                "random_seed": 42,
                "training": {
                    "batch_size": 1,
                    "max_episode": 50000,
                },
            },
        }
        
        # Load single game
        self.env = environment.AlfredTWEnv(config, train_eval="eval")
        self.env.game_files = [game_path]
        self.env.num_games = 1

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
                        game_files.append(str(game_file))
        
        return sorted(game_files)

    def reset(self, game_path: str) -> Tuple[str, Dict[str, Any]]:
        """Reset environment with a specific game.
        
        Args:
            game_path: Path to game.tw-pddl file.
            
        Returns:
            Tuple of (initial_observation, info_dict).
        """
        self._load_environment(game_path)
        self.current_game_path = game_path
        self.current_game_id = get_game_id_from_path(game_path)
        self.current_task_type_id, self.current_task_type = get_task_type_from_path(game_path)
        
        obs, info = self.env.reset()
        
        # Extract text observation
        if isinstance(obs, list):
            obs = obs[0]
        
        # Store admissible commands
        self.admissible_commands = info.get("admissible_commands", [[]])[0]
        
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
            valid_actions_str = "Valid actions:\n" + "\n".join(f"  - {cmd}" for cmd in self.admissible_commands)
            return valid_actions_str, 0, False, {
                "admissible_commands": self.admissible_commands,
                "won": False,
            }
        
        # Execute action in environment
        obs, scores, dones, infos = self.env.step([action])
        
        # Extract values from batch
        obs = obs[0] if isinstance(obs, list) else obs
        done = dones[0] if isinstance(dones, list) else dones
        score = scores[0] if isinstance(scores, list) else scores
        
        # Update admissible commands
        self.admissible_commands = infos.get("admissible_commands", [[]])[0]
        
        # Check if won
        won = infos.get("won", [False])[0]
        
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

