"""Few-shot examples for each task type in ALFWorld."""

from typing import Dict, Optional

# Task type ID to name mapping
TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}

# Few-shot examples for each task type
TASK_TYPE_EXAMPLES: Dict[int, str] = {
    # Task Type 1: Pick and Place Simple
    1: """Example: Put a book on the sidetable

Observation: You are in the middle of a room. Looking around you, you see a bed 1, a desk 1, a drawer 1, a sidetable 1, and a sidetable 2.
Your task is to: put some book on sidetable.

Think: I need to find a book first. Let me look around to see where books might be. The bed is a common place for books.

Action: go to bed 1

Observation: On the bed 1, you see a book 1, a pillow 1, and a cellphone 1.

Think: I found a book on the bed. Let me pick it up.

Action: take book 1 from bed 1

Observation: You pick up the book 1 from the bed 1.

Think: Now I have the book. I need to put it on a sidetable. Let me go to sidetable 1.

Action: go to sidetable 1

Observation: On the sidetable 1, you see a lamp 1.

Think: I'm at the sidetable. Now I'll put the book here.

Action: put book 1 in/on sidetable 1

Observation: You put the book 1 in/on the sidetable 1.
Task completed!""",

    # Task Type 2: Look at Object in Light
    2: """Example: Examine an alarm clock with a desk lamp

Observation: You are in the middle of a room. Looking around you, you see a bed 1, a desk 1, a desklamp 1, and a drawer 1.
Your task is to: examine the alarmclock with the desklamp.

Think: I need to find an alarm clock and examine it under a desk lamp. First, let me find where the alarm clock is.

Action: go to desk 1

Observation: On the desk 1, you see an alarmclock 1, a pen 1, and a pencil 1.

Think: I found the alarm clock on the desk. Let me pick it up.

Action: take alarmclock 1 from desk 1

Observation: You pick up the alarmclock 1 from the desk 1.

Think: Now I have the alarm clock. I need to use the desk lamp to examine it. The desklamp should be nearby.

Action: use desklamp 1

Observation: You turn on the desklamp 1. The alarmclock 1 is illuminated.
Task completed!""",

    # Task Type 3: Pick, Clean, then Place
    3: """Example: Clean a mug and put it in the coffee machine

Observation: You are in the middle of a room. Looking around you, you see a cabinet 1, a coffeemachine 1, a countertop 1, a fridge 1, and a sinkbasin 1.
Your task is to: clean some mug and put it in coffeemachine.

Think: I need to find a mug, clean it in the sink, and then put it in the coffee machine. Let me first look for a mug.

Action: go to countertop 1

Observation: On the countertop 1, you see a mug 1, a knife 1, and a apple 1.

Think: I found a mug on the countertop. Let me pick it up.

Action: take mug 1 from countertop 1

Observation: You pick up the mug 1 from the countertop 1.

Think: Now I have the mug. I need to clean it in the sink basin.

Action: go to sinkbasin 1

Observation: On the sinkbasin 1, you see nothing.

Think: I'm at the sink. Let me clean the mug.

Action: clean mug 1 with sinkbasin 1

Observation: You clean the mug 1 using the sinkbasin 1.

Think: The mug is now clean. I need to put it in the coffee machine.

Action: go to coffeemachine 1

Observation: On the coffeemachine 1, you see nothing.

Think: I'm at the coffee machine. Let me put the clean mug here.

Action: put mug 1 in/on coffeemachine 1

Observation: You put the mug 1 in/on the coffeemachine 1.
Task completed!""",

    # Task Type 4: Pick, Heat, then Place
    4: """Example: Heat an egg and put it in the fridge

Observation: You are in the middle of a room. Looking around you, you see a cabinet 1, a countertop 1, a fridge 1, and a microwave 1.
Your task is to: heat some egg and put it in fridge.

Think: I need to find an egg, heat it in the microwave, and then put it in the fridge. Let me first look for an egg.

Action: go to fridge 1

Observation: The fridge 1 is closed.

Think: The fridge is closed. Let me open it to see if there's an egg inside.

Action: open fridge 1

Observation: You open the fridge 1. The fridge 1 is open. In it, you see an egg 1, a tomato 1, and a lettuce 1.

Think: I found an egg in the fridge. Let me take it.

Action: take egg 1 from fridge 1

Observation: You pick up the egg 1 from the fridge 1.

Think: Now I have the egg. I need to heat it in the microwave.

Action: go to microwave 1

Observation: The microwave 1 is closed.

Think: I'm at the microwave. Let me heat the egg.

Action: heat egg 1 with microwave 1

Observation: You heat the egg 1 using the microwave 1.

Think: The egg is now heated. I need to put it back in the fridge.

Action: go to fridge 1

Observation: The fridge 1 is open. In it, you see a tomato 1, and a lettuce 1.

Think: I'm back at the fridge. Let me put the heated egg inside.

Action: put egg 1 in/on fridge 1

Observation: You put the egg 1 in/on the fridge 1.
Task completed!""",

    # Task Type 5: Pick, Cool, then Place
    5: """Example: Cool an apple and put it on the countertop

Observation: You are in the middle of a room. Looking around you, you see a cabinet 1, a countertop 1, a fridge 1, and a microwave 1.
Your task is to: cool some apple and put it in countertop.

Think: I need to find an apple, cool it in the fridge, and then put it on the countertop. Let me look for an apple.

Action: go to countertop 1

Observation: On the countertop 1, you see an apple 1, a knife 1, and a bread 1.

Think: I found an apple on the countertop. Let me pick it up.

Action: take apple 1 from countertop 1

Observation: You pick up the apple 1 from the countertop 1.

Think: Now I have the apple. I need to cool it in the fridge.

Action: go to fridge 1

Observation: The fridge 1 is closed.

Think: I'm at the fridge. Let me cool the apple.

Action: cool apple 1 with fridge 1

Observation: You cool the apple 1 using the fridge 1.

Think: The apple is now cooled. I need to put it on the countertop.

Action: go to countertop 1

Observation: On the countertop 1, you see a knife 1, and a bread 1.

Think: I'm at the countertop. Let me put the cooled apple here.

Action: put apple 1 in/on countertop 1

Observation: You put the apple 1 in/on the countertop 1.
Task completed!""",

    # Task Type 6: Pick Two Objects and Place
    6: """Example: Put two cellphones in a drawer

Observation: You are in the middle of a room. Looking around you, you see a bed 1, a desk 1, a drawer 1, a sidetable 1, and a sidetable 2.
Your task is to: put two cellphone in drawer.

Think: I need to find two cellphones and put them in a drawer. Let me start by looking around for cellphones.

Action: go to desk 1

Observation: On the desk 1, you see a cellphone 1, a pen 1, and a book 1.

Think: I found the first cellphone on the desk. Let me pick it up.

Action: take cellphone 1 from desk 1

Observation: You pick up the cellphone 1 from the desk 1.

Think: I have the first cellphone. Let me put it in the drawer first.

Action: go to drawer 1

Observation: The drawer 1 is closed.

Think: The drawer is closed. Let me open it.

Action: open drawer 1

Observation: You open the drawer 1. The drawer 1 is open. In it, you see nothing.

Think: The drawer is now open. Let me put the first cellphone inside.

Action: put cellphone 1 in/on drawer 1

Observation: You put the cellphone 1 in/on the drawer 1.

Think: First cellphone is in the drawer. Now I need to find the second cellphone.

Action: go to sidetable 1

Observation: On the sidetable 1, you see a cellphone 2, and a lamp 1.

Think: I found the second cellphone. Let me pick it up.

Action: take cellphone 2 from sidetable 1

Observation: You pick up the cellphone 2 from the sidetable 1.

Think: I have the second cellphone. Let me go back to the drawer to put it in.

Action: go to drawer 1

Observation: The drawer 1 is open. In it, you see a cellphone 1.

Think: I'm at the drawer with the second cellphone. Let me put it inside.

Action: put cellphone 2 in/on drawer 1

Observation: You put the cellphone 2 in/on the drawer 1.
Task completed!""",
}


def get_few_shot_examples(task_type_id: int) -> Optional[str]:
    """Get few-shot examples for a specific task type.
    
    Args:
        task_type_id: Task type ID (1-6).
        
    Returns:
        Few-shot examples string, or None if not available.
    """
    return TASK_TYPE_EXAMPLES.get(task_type_id)


def get_few_shot_examples_by_name(task_type_name: str) -> Optional[str]:
    """Get few-shot examples by task type name.
    
    Args:
        task_type_name: Task type name (e.g., "pick_and_place_simple").
        
    Returns:
        Few-shot examples string, or None if not available.
    """
    # Find task type ID from name
    for task_id, name in TASK_TYPES.items():
        if name == task_type_name:
            return TASK_TYPE_EXAMPLES.get(task_id)
    
    # Handle pick_and_place_with_movable_recep as type 1
    if task_type_name == "pick_and_place_with_movable_recep":
        return TASK_TYPE_EXAMPLES.get(1)
    
    return None

