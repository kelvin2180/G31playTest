from .runner_agent import RunnerAgent
from .battle_agent import BattleAgent
from .menu_agent import MenuAgent

class Orchestrator:
    def __init__(self):
        self.runner = RunnerAgent()
        self.battle = BattleAgent()
        self.menu = MenuAgent()

    def get_worker(self, state_name):
        if state_name == "BATTLE":
            return self.battle
        elif state_name == "MENU":
            return self.menu
        else: # OVERWORLD or unknown
            return self.runner
