"""System prompt and user prompt builder for ALFWorld evaluation."""

from typing import List, Tuple, Optional, TYPE_CHECKING

from .few_shot import FEW_SHOT_EXAMPLES

if TYPE_CHECKING:
    from ..memory import RetrievedMemory

# Base system prompt without few-shot examples
_SYSTEM_PROMPT_BASE = """You are an intelligent agent operating in a household environment. Your goal is to complete tasks by interacting with objects and navigating through rooms.

==================================================
ENVIRONMENT RULES
==================================================
1. You can only carry ONE object at a time
2. Use exact object names with numbers (e.g., "apple 1", "fridge 1")

==================================================
AVAILABLE COMMANDS
==================================================
Navigation:
  - look                          : View current surroundings
  - go to [receptacle]            : Move to a location (e.g., "go to fridge 1")

Object Manipulation:
  - take [object] from [receptacle] : Pick up an object (e.g., "take apple 1 from fridge 1")
  - move [object] to [receptacle]   : Place object (e.g., "move apple 1 to fridge 1")
  - open [receptacle]               : Open a container (e.g., "open fridge 1")
  - close [receptacle]              : Close a container

Object Processing:
  - heat [object] with [receptacle] : Heat with microwave (e.g., "heat egg 1 with microwave 1")
  - cool [object] with [receptacle] : Cool with fridge (e.g., "cool apple 1 with fridge 1")
  - clean [object] with [receptacle]: Clean with sink (e.g., "clean mug 1 with sinkbasin 1")
  - use [object]                    : Use/toggle object (e.g., "use desklamp 1")

Utility:
  - inventory                      : Check what you're carrying
  - examine [object]               : Look at object details
  - check valid actions            : List all currently valid actions

==================================================
OUTPUT FORMAT (REQUIRED)
==================================================
You MUST respond in EXACTLY this format:

Think: <your reasoning about the current situation and what to do next>

Action: <exact command from the list above>

IMPORTANT:
- Always include both "Think:" and "Action:" sections
- The action must be a valid command with exact object/receptacle names
- If stuck, use "check valid actions" to see available options"""

# System prompt with few-shot examples
SYSTEM_PROMPT_WITH_EXAMPLES = _SYSTEM_PROMPT_BASE + """

==================================================
EXAMPLE DEMONSTRATIONS
==================================================
The following examples show how to complete various tasks:

""" + FEW_SHOT_EXAMPLES

# System prompt without few-shot examples (for backward compatibility)
SYSTEM_PROMPT = _SYSTEM_PROMPT_BASE


def get_system_prompt(use_few_shot: bool = True) -> str:
    """Get system prompt with or without few-shot examples.

    Args:
        use_few_shot: Whether to include few-shot examples.

    Returns:
        System prompt string.
    """
    if use_few_shot:
        return SYSTEM_PROMPT_WITH_EXAMPLES
    return SYSTEM_PROMPT


def _format_trajectory_for_memory(trajectory: List[dict]) -> str:
    """Format trajectory for memory display.
    
    Args:
        trajectory: List of action-observation pairs.
        
    Returns:
        Formatted trajectory string (abbreviated).
    """
    if not trajectory:
        return "(empty)"
    
    # Show abbreviated trajectory (first few and last few steps)
    max_show = 6
    lines = []
    
    if len(trajectory) <= max_show:
        for step in trajectory:
            action = step.get("action", "")
            lines.append(f"  > {action}")
    else:
        # Show first 3 and last 3
        for step in trajectory[:3]:
            action = step.get("action", "")
            lines.append(f"  > {action}")
        lines.append(f"  ... ({len(trajectory) - 6} more steps) ...")
        for step in trajectory[-3:]:
            action = step.get("action", "")
            lines.append(f"  > {action}")
    
    return "\n".join(lines)


def _format_memory_items(memory_items: List) -> str:
    """Format memory items for display.
    
    Args:
        memory_items: List of MemoryEntry objects.
        
    Returns:
        Formatted memory items string.
    """
    if not memory_items:
        return ""
    
    lines = ["  Key Insights:"]
    for item in memory_items:
        lines.append(f"    - {item.title}: {item.description}")
        if item.content:
            # Truncate long content
            content = item.content[:200] + "..." if len(item.content) > 200 else item.content
            lines.append(f"      {content}")
    
    return "\n".join(lines)


def build_memory_section(retrieved_memories: List["RetrievedMemory"]) -> str:
    """Build the memory section for system prompt.
    
    Args:
        retrieved_memories: List of RetrievedMemory objects.
        
    Returns:
        Formatted memory section string.
    """
    if not retrieved_memories:
        return ""
    
    parts = [
        "",
        "==================================================",
        "RELEVANT EXPERIENCE FROM SIMILAR TASKS",
        "==================================================",
        "Below are experiences from past interactions that may help with your current task.",
        "Use them as reference when relevant, but adapt to the specific situation.",
        "",
    ]
    
    for i, rm in enumerate(retrieved_memories, 1):
        result_str = "SUCCESS" if rm.is_success else "FAILED"
        parts.append(f"[Experience #{i}] (Similarity: {rm.similarity:.2f}, Result: {result_str})")
        parts.append(f"  Goal: {rm.query}")
        
        # Add trajectory summary
        parts.append(f"  Actions taken:")
        parts.append(_format_trajectory_for_memory(rm.trajectory))
        
        # Add memory items (extracted insights)
        if rm.memory_items:
            parts.append(_format_memory_items(rm.memory_items))
        
        parts.append("")
    
    return "\n".join(parts)


def get_system_prompt_with_memory(
    use_few_shot: bool = True,
    retrieved_memories: Optional[List["RetrievedMemory"]] = None,
) -> str:
    """Get system prompt with optional few-shot examples and retrieved memories.

    Args:
        use_few_shot: Whether to include few-shot examples.
        retrieved_memories: Optional list of retrieved memories to include.

    Returns:
        System prompt string.
    """
    base_prompt = get_system_prompt(use_few_shot)
    
    if not retrieved_memories:
        return base_prompt
    
    memory_section = build_memory_section(retrieved_memories)
    
    # Insert memory section before OUTPUT FORMAT section
    # Find the position to insert
    output_format_marker = "==================================================\nOUTPUT FORMAT"
    
    if output_format_marker in base_prompt:
        idx = base_prompt.find(output_format_marker)
        return base_prompt[:idx] + memory_section + "\n" + base_prompt[idx:]
    else:
        # Fallback: append at the end
        return base_prompt + memory_section


def build_user_prompt(
    task_description: str,
    history: List[Tuple[str, str]],
    current_observation: str,
    history_length: int = 10,
) -> str:
    """Build user prompt with task, history, and current observation.

    Args:
        task_description: The task goal description.
        history: List of (action, observation) tuples.
        current_observation: The most recent observation.
        history_length: Number of recent history entries to include.

    Returns:
        Formatted user prompt string.
    """
    parts = []

    # Add current task
    parts.append("==================================================")
    parts.append("YOUR CURRENT TASK")
    parts.append("==================================================")
    parts.append(f"Goal: {task_description}")
    parts.append("")
    parts.append("Hints:")
    parts.append("  - Type 'check valid actions' if you're unsure what to do")
    parts.append("  - Type 'inventory' to check what you're carrying")
    parts.append("  - Type 'look' to observe your surroundings")
    parts.append("")

    # Add recent history
    parts.append("==================================================")
    parts.append("RECENT HISTORY")
    parts.append("==================================================")

    # Limit history length
    recent_history = history[-history_length:] if len(
        history) > history_length else history

    if recent_history:
        for action, observation in recent_history:
            parts.append(f"Action: {action}")
            parts.append(f"Observation: {observation}")
            parts.append("")

    # Add current observation
    parts.append("Current Observation:")
    parts.append(current_observation)
    parts.append("")

    # Reminder
    parts.append("==================================================")
    parts.append("YOUR TURN")
    parts.append("==================================================")
    parts.append(
        "Based on the task goal and current observation, decide your next action.")
    parts.append("Remember to use the exact format: Think: ... Action: ...")

    return "\n".join(parts)


def extract_task_description(initial_observation: str) -> str:
    """Extract task description from initial observation.

    Args:
        initial_observation: The initial environment observation.

    Returns:
        Task description string.
    """
    # Look for "Your task is to:" pattern
    lines = initial_observation.split("\n")
    for line in lines:
        if "your task is to" in line.lower():
            return line.strip()

    # Return full observation if no specific task found
    return initial_observation.strip()
