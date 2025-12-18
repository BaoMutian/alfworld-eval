"""Static few-shot examples for ALFWorld evaluation.

Based on official ALFWorld command format:
- Navigation: look, go to [receptacle]
- Object: take [object] from [receptacle], move [object] to [receptacle]
- Container: open [receptacle], close [receptacle]
- Processing: heat/cool/clean [object] with [receptacle], use [object]
- Utility: inventory, examine [object]
"""

# Static few-shot examples covering different task types
FEW_SHOT_EXAMPLES = """Example 1: Examine an alarm clock under a desk lamp

> look
You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, a desklamp 1, a drawer 1.
Your task is to: examine the alarmclock with the desklamp.

Think: I need to find an alarm clock and examine it under the desk lamp. Let me check the desk first.
Action: go to desk 1

> go to desk 1
On the desk 1, you see an alarmclock 1, a pen 1.

Think: Found the alarm clock. Let me pick it up.
Action: take alarmclock 1 from desk 1

> take alarmclock 1 from desk 1
You pick up the alarmclock 1 from the desk 1.

Think: Now I have the alarm clock. I need to use the desk lamp to illuminate and examine it.
Action: use desklamp 1

> use desklamp 1
You turn on the desklamp 1.
Task completed!

---

Example 2: Clean a mug and put it in the coffee machine

> look
You are in the middle of a room. Looking quickly around you, you see a cabinet 1, a coffeemachine 1, a countertop 1, a fridge 1, a sinkbasin 1.
Your task is to: clean some mug and put it in coffeemachine.

Think: I need to find a mug, clean it in the sink, then put it in the coffee machine. Let me look for a mug.
Action: go to countertop 1

> go to countertop 1
On the countertop 1, you see a mug 1, a knife 1.

Think: I found a mug. Let me pick it up.
Action: take mug 1 from countertop 1

> take mug 1 from countertop 1
You pick up the mug 1 from the countertop 1.

Think: Now I need to clean the mug in the sink.
Action: go to sinkbasin 1

> go to sinkbasin 1
On the sinkbasin 1, you see nothing.

Think: I'm at the sink. Let me clean the mug.
Action: clean mug 1 with sinkbasin 1

> clean mug 1 with sinkbasin 1
You clean the mug 1 using the sinkbasin 1.

Think: The mug is clean. Now I need to put it in the coffee machine.
Action: go to coffeemachine 1

> go to coffeemachine 1
On the coffeemachine 1, you see nothing.

Think: I'm at the coffee machine. Let me put the clean mug here.
Action: move mug 1 to coffeemachine 1

> move mug 1 to coffeemachine 1
You put the mug 1 in/on the coffeemachine 1.
Task completed!

---

Example 3: Heat an egg and put it in the fridge

> look
You are in the middle of a room. Looking quickly around you, you see a countertop 1, a fridge 1, a microwave 1.
Your task is to: heat some egg and put it in fridge.

Think: I need to find an egg, heat it with the microwave, then put it in the fridge. Let me check the fridge first.
Action: go to fridge 1

> go to fridge 1
The fridge 1 is closed.

Think: The fridge is closed. Let me open it.
Action: open fridge 1

> open fridge 1
You open the fridge 1. The fridge 1 is open. In it, you see an egg 1, a tomato 1.

Think: I found an egg. Let me take it.
Action: take egg 1 from fridge 1

> take egg 1 from fridge 1
You pick up the egg 1 from the fridge 1.

Think: Now I need to heat the egg with the microwave.
Action: go to microwave 1

> go to microwave 1
The microwave 1 is closed.

Think: I'm at the microwave. Let me heat the egg.
Action: heat egg 1 with microwave 1

> heat egg 1 with microwave 1
You heat the egg 1 using the microwave 1.

Think: The egg is heated. Now I need to put it back in the fridge.
Action: go to fridge 1

> go to fridge 1
The fridge 1 is open. In it, you see a tomato 1.

Think: I'm at the fridge. Let me put the heated egg inside.
Action: move egg 1 to fridge 1

> move egg 1 to fridge 1
You put the egg 1 in/on the fridge 1.
Task completed!"""
