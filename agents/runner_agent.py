import random

class RunnerAgent:
    def act(self, frames, journal, scratchpad):
        actions = []
        # Very naive implementation: Pick a random direction and walk
        directions = ['up', 'down', 'left', 'right']
        direction = random.choice(directions)
        
        actions.append(('hold', direction, 0.5))
        new_scratchpad = f"Moved {direction} to explore."
        
        return actions, new_scratchpad
