"""Memory extractor for extracting strategies from trajectories using LLM."""

import json
import logging
from typing import List, Dict, Any, Optional

from .schemas import Memory, MemoryEntry
from .prompts import (
    build_extraction_prompt,
    build_contrastive_extraction_prompt,
)

logger = logging.getLogger(__name__)


def _try_parse_json(text: str) -> Optional[List[Dict]]:
    """Try to parse JSON from text, with multiple strategies.
    
    Args:
        text: Text that may contain JSON.
        
    Returns:
        Parsed JSON list or None if parsing fails.
    """
    # Clean the text
    text = text.strip()
    
    # Try direct parsing first
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        return None
    except json.JSONDecodeError:
        pass

    # Try to extract JSON array from text
    # Look for [ ... ] pattern
    start_idx = text.find('[')
    end_idx = text.rfind(']')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = text[start_idx:end_idx + 1]
        try:
            result = json.loads(json_str)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Try using json_repair library if available
    try:
        from json_repair import repair_json
        repaired = repair_json(text, return_objects=True)
        if isinstance(repaired, list):
            return repaired
        elif isinstance(repaired, str):
            # repair_json might return string, try parsing again
            try:
                result = json.loads(repaired)
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass
    except ImportError:
        logger.debug("json_repair not available, skipping repair attempt")
    except Exception as e:
        logger.debug(f"json_repair failed: {e}")

    return None


def _validate_memory_items(items: List[Dict]) -> List[MemoryEntry]:
    """Validate and convert parsed items to MemoryEntry objects.
    
    Args:
        items: List of parsed dictionaries.
        
    Returns:
        List of valid MemoryEntry objects.
    """
    valid_entries = []
    
    for item in items:
        if not isinstance(item, dict):
            continue
            
        title = str(item.get("title", "")).strip()
        description = str(item.get("description", "")).strip()
        content = str(item.get("content", "")).strip()
        
        # Skip entries with missing essential fields
        if not title or not content:
            logger.debug(f"Skipping invalid entry: missing title or content")
            continue
            
        valid_entries.append(MemoryEntry(
            title=title,
            description=description,
            content=content,
        ))
    
    return valid_entries


class MemoryExtractor:
    """Extractor for extracting memory items from task trajectories.
    
    Uses LLM to analyze trajectories and extract reusable strategies.
    """

    def __init__(
        self,
        llm_client: Any,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ):
        """Initialize memory extractor.
        
        Args:
            llm_client: LLM client for generating extractions.
            temperature: Sampling temperature for extraction.
            max_tokens: Maximum tokens for extraction response.
        """
        self.llm_client = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens

    def extract(
        self,
        task_id: str,
        task_type: str,
        goal: str,
        trajectory: List[Dict[str, str]],
        is_success: bool,
    ) -> Optional[Memory]:
        """Extract memory from a single trajectory.
        
        Args:
            task_id: Unique task identifier.
            task_type: Type of the task.
            goal: Task goal description.
            trajectory: List of action-observation pairs.
            is_success: Whether the task was successful.
            
        Returns:
            Memory object if extraction succeeds, None otherwise.
        """
        if not trajectory:
            logger.warning(f"Empty trajectory for task {task_id}, skipping extraction")
            return None

        try:
            # Build extraction prompt
            prompt = build_extraction_prompt(
                task_type=task_type,
                goal=goal,
                trajectory=trajectory,
                is_success=is_success,
            )

            # Call LLM
            response = self.llm_client.chat_simple(
                system_prompt="You are an expert at analyzing task execution and extracting reusable strategies.",
                user_prompt=prompt,
                context="Memory Extraction",
                game_id=task_id,
            )

            # Parse response
            items = _try_parse_json(response)
            
            if items is None:
                logger.warning(f"Failed to parse extraction response for task {task_id}")
                logger.debug(f"Raw response: {response[:500]}...")
                return None

            # Validate and convert to MemoryEntry
            memory_items = _validate_memory_items(items)
            
            if not memory_items:
                logger.warning(f"No valid memory items extracted for task {task_id}")
                return None

            # Create Memory object
            memory = Memory(
                memory_id=Memory.generate_id(),
                task_id=task_id,
                task_type=task_type,
                query=goal,
                trajectory=trajectory,
                is_success=is_success,
                memory_items=memory_items,
            )

            logger.debug(
                f"Extracted {len(memory_items)} memory items for task {task_id} "
                f"({'success' if is_success else 'failure'})"
            )
            
            return memory

        except Exception as e:
            logger.error(f"Memory extraction failed for task {task_id}: {e}")
            return None

    def extract_contrastive(
        self,
        task_id: str,
        task_type: str,
        goal: str,
        trajectories: List[Dict],
    ) -> Optional[Memory]:
        """Extract memory from multiple trajectories using contrastive analysis.
        
        Used for MaTTS (Memory-aware Test-Time Scaling) where multiple
        attempts are compared to extract higher-quality insights.
        
        Args:
            task_id: Unique task identifier.
            task_type: Type of the task.
            goal: Task goal description.
            trajectories: List of trajectory dicts with 'trajectory' and 'is_success'.
            
        Returns:
            Memory object if extraction succeeds, None otherwise.
        """
        if not trajectories:
            logger.warning(f"No trajectories for contrastive extraction, task {task_id}")
            return None

        try:
            # Build contrastive extraction prompt
            prompt = build_contrastive_extraction_prompt(
                task_type=task_type,
                goal=goal,
                trajectories=trajectories,
            )

            # Call LLM
            response = self.llm_client.chat_simple(
                system_prompt="You are an expert at analyzing task execution and extracting patterns from multiple attempts.",
                user_prompt=prompt,
                context="Contrastive Memory Extraction",
                game_id=task_id,
            )

            # Parse response
            items = _try_parse_json(response)
            
            if items is None:
                logger.warning(f"Failed to parse contrastive extraction response for task {task_id}")
                return None

            memory_items = _validate_memory_items(items)
            
            if not memory_items:
                logger.warning(f"No valid memory items from contrastive extraction for task {task_id}")
                return None

            # Determine overall success (any success counts)
            any_success = any(t.get("is_success", False) for t in trajectories)
            
            # Use the first trajectory as representative
            # (or could combine them)
            representative_trajectory = trajectories[0].get("trajectory", [])

            memory = Memory(
                memory_id=Memory.generate_id(),
                task_id=task_id,
                task_type=task_type,
                query=goal,
                trajectory=representative_trajectory,
                is_success=any_success,
                memory_items=memory_items,
            )

            logger.debug(
                f"Contrastive extraction: {len(memory_items)} items from "
                f"{len(trajectories)} trajectories for task {task_id}"
            )
            
            return memory

        except Exception as e:
            logger.error(f"Contrastive extraction failed for task {task_id}: {e}")
            return None

