class MenuAgent:
    def act(self, frames, journal, scratchpad):
        # Naive menu: Press B (x) to exit menu
        actions = [('tap', 'x')]
        new_scratchpad = "In menu, pressing B to exit."
        return actions, new_scratchpad
