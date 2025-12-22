"""Prompt templates for memory extraction."""

from typing import List, Dict

# Environment context for memory extraction
ENVIRONMENT_CONTEXT = """## Environment Background
ALFWorld is a text-based household environment where an agent must complete tasks by navigating rooms and interacting with objects.

**Key Rules:**
- Agent can only carry ONE object at a time
- Objects have numbered names (e.g., "apple 1", "fridge 1")
- Some receptacles (fridge, cabinet) need to be opened before accessing contents

**Available Commands:**
- Navigation: look, go to [receptacle]
- Object: take [object] from [receptacle], move [object] to [receptacle]
- Container: open [receptacle], close [receptacle]
- Processing: heat/cool/clean [object] with [receptacle], use [object]
- Utility: inventory, examine [object]

**Task Types:**
1. pick_and_place_simple: Find an object and place it somewhere
2. look_at_obj_in_light: Examine an object under a light source
3. pick_clean_then_place: Clean an object and place it
4. pick_heat_then_place: Heat an object and place it
5. pick_cool_then_place: Cool an object and place it
6. pick_two_obj_and_place: Find two objects and place them
"""

# Prompt for extracting strategies from successful trajectories
EXTRACTION_PROMPT_SUCCESS = """You are an expert at analyzing task execution trajectories and extracting reusable reasoning strategies.

{environment_context}

## Task Context
- Task Type: {task_type}
- Task Goal: {goal}
- Result: SUCCESS

## Trajectory
{trajectory}

## Instructions
Analyze this SUCCESSFUL trajectory and extract 1-3 reusable strategies that contributed to success.
For each strategy, provide:
1. **title**: A short, descriptive name (e.g., "Systematic Object Search", "Check Container Before Taking")
2. **description**: A one-sentence summary of when this strategy applies
3. **content**: Detailed actionable insight on the technique or logic

Focus on:
- Key decision points that led to success
- Efficient patterns or shortcuts discovered
- General principles that could apply to similar tasks
- How environment rules were leveraged effectively

## Output Format
Return a JSON array of strategy objects:
```json
[
  {{
    "title": "Strategy Name",
    "description": "When to use this strategy",
    "content": "Detailed explanation of the strategy and how to apply it"
  }}
]
```

Output ONLY the JSON array, no additional text."""

# Prompt for extracting lessons from failed trajectories
EXTRACTION_PROMPT_FAILURE = """You are an expert at analyzing task execution trajectories and extracting lessons from failures.

{environment_context}

## Task Context
- Task Type: {task_type}
- Task Goal: {goal}
- Result: FAILED

## Trajectory
{trajectory}

## Instructions
Analyze this FAILED trajectory and extract 1-3 preventive lessons that could help avoid similar failures.
For each lesson, provide:
1. **title**: A short, descriptive name (e.g., "Avoid Blind Searching", "Check Container Status First")
2. **description**: A one-sentence summary of the pitfall to avoid
3. **content**: Detailed explanation of what went wrong and how to prevent it

Focus on:
- Critical mistakes or wrong assumptions
- Inefficient patterns that wasted steps
- Missing checks or validations that should have been done
- How environment rules were violated or misunderstood

## Output Format
Return a JSON array of lesson objects:
```json
[
  {{
    "title": "Lesson Name",
    "description": "Pitfall to avoid",
    "content": "What went wrong and how to prevent it"
  }}
]
```

Output ONLY the JSON array, no additional text."""

# Prompt for contrastive extraction (MaTTS)
EXTRACTION_PROMPT_CONTRASTIVE = """You are an expert at analyzing multiple task execution trajectories and extracting consistent patterns.

{environment_context}
## Task Context
- Task Type: {task_type}
- Task Goal: {goal}
- Number of Trajectories: {num_trajectories}

## Trajectories
{trajectories}

## Instructions
Compare these {num_trajectories} trajectories for the SAME task and extract insights:
1. If some succeeded and some failed, identify what distinguishes successful from failed attempts
2. If all succeeded, identify consistent winning strategies
3. If all failed, identify common pitfalls to avoid

Extract 1-3 high-quality strategies/lessons that:
- Are consistent across multiple attempts (not coincidental)
- Represent general patterns, not task-specific details
- Could help with similar future tasks
- Reflect understanding of environment rules and constraints

## Output Format
Return a JSON array:
```json
[
  {{
    "title": "Strategy/Lesson Name",
    "description": "One-sentence summary",
    "content": "Detailed explanation derived from comparing multiple trajectories"
  }}
]
```

Output ONLY the JSON array, no additional text."""


def format_trajectory(trajectory: List[Dict[str, str]]) -> str:
    """Format trajectory for prompt.

    Args:
        trajectory: List of action-observation pairs.

    Returns:
        Formatted trajectory string.
    """
    lines = []
    for i, step in enumerate(trajectory, 1):
        action = step.get("action", "")
        observation = step.get("observation", "")
        lines.append(f"Step {i}:")
        lines.append(f"  Action: {action}")
        lines.append(f"  Observation: {observation}")
        lines.append("")
    return "\n".join(lines)


def format_multiple_trajectories(
    trajectories: List[Dict],
) -> str:
    """Format multiple trajectories for contrastive extraction.

    Args:
        trajectories: List of trajectory dicts with 'trajectory' and 'is_success' keys.

    Returns:
        Formatted string with all trajectories.
    """
    lines = []
    for i, traj_data in enumerate(trajectories, 1):
        result = "SUCCESS" if traj_data.get("is_success", False) else "FAILED"
        lines.append(f"=== Trajectory {i} ({result}) ===")
        lines.append(format_trajectory(traj_data.get("trajectory", [])))
        lines.append("")
    return "\n".join(lines)


def build_extraction_prompt(
    task_type: str,
    goal: str,
    trajectory: List[Dict[str, str]],
    is_success: bool,
) -> str:
    """Build extraction prompt for a single trajectory.

    Args:
        task_type: Type of the task.
        goal: Task goal description.
        trajectory: List of action-observation pairs.
        is_success: Whether the task was successful.

    Returns:
        Formatted prompt string.
    """
    template = EXTRACTION_PROMPT_SUCCESS if is_success else EXTRACTION_PROMPT_FAILURE
    formatted_trajectory = format_trajectory(trajectory)

    return template.format(
        environment_context=ENVIRONMENT_CONTEXT,
        task_type=task_type,
        goal=goal,
        trajectory=formatted_trajectory,
    )


def build_contrastive_extraction_prompt(
    task_type: str,
    goal: str,
    trajectories: List[Dict],
) -> str:
    """Build extraction prompt for multiple trajectories (MaTTS).

    Args:
        task_type: Type of the task.
        goal: Task goal description.
        trajectories: List of trajectory dicts.

    Returns:
        Formatted prompt string.
    """
    formatted_trajectories = format_multiple_trajectories(trajectories)

    return EXTRACTION_PROMPT_CONTRASTIVE.format(
        environment_context=ENVIRONMENT_CONTEXT,
        task_type=task_type,
        goal=goal,
        num_trajectories=len(trajectories),
        trajectories=formatted_trajectories,
    )
