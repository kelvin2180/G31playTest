class BattleAgent:
    def act(self, frames, journal, scratchpad):
        # Naive battle: just mash A (z) to attack
        actions = [('tap', 'z'), ('tap', 'z')]
        new_scratchpad = "Mashing Z to fight in battle."
        return actions, new_scratchpad
