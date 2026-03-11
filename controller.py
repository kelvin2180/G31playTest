import time
from pynput.keyboard import Controller, Key

keyboard = Controller()

# Map string inputs to pynput Keys
KEY_MAP = {
    'z': 'z', # A button
    'x': 'x', # B button
    'enter': Key.enter, # Start
    'backspace': Key.backspace, # Select
    'up': Key.up,
    'down': Key.down,
    'left': Key.left,
    'right': Key.right
}

def tap(key_str, delay=0.1):
    key = KEY_MAP.get(key_str.lower(), key_str)
    keyboard.press(key)
    time.sleep(delay)
    keyboard.release(key)
    time.sleep(0.05)

def hold(key_str, duration=1.0):
    key = KEY_MAP.get(key_str.lower(), key_str)
    keyboard.press(key)
    time.sleep(duration)
    keyboard.release(key)
    time.sleep(0.05)
    
def execute_actions(actions):
    """actions is a list of tuples like [('tap', 'z'), ('hold', 'up', 1.0)]"""
    for action in actions:
        cmd = action[0]
        if cmd == 'tap':
            tap(action[1])
        elif cmd == 'hold':
            hold(action[1], action[2])
