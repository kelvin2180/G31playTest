import mgba.core
import mgba.image
import mgba.log
from mgba.core import lib
import numpy as np
import cv2
import collections
import os

# Silence emulator warnings and debug logs
mgba.log.silence()

class EmulatorController:
    def __init__(self, rom_path):
        self.rom_path = rom_path
        self.core = mgba.core.load_path(rom_path)
        if not self.core:
            raise ValueError(f"Could not load ROM: {rom_path}")
            
        self.width, self.height = self.core.desired_video_dimensions()
        self.image = mgba.image.Image(self.width, self.height)
        self.core.set_video_buffer(self.image)
        self.core.reset()
        
        # Agent Action Mapping
        self.key_map = {
            'z': lib.GBA_KEY_A, 'x': lib.GBA_KEY_B,
            'enter': lib.GBA_KEY_START, 'backspace': lib.GBA_KEY_SELECT,
            'up': lib.GBA_KEY_UP, 'down': lib.GBA_KEY_DOWN,
            'left': lib.GBA_KEY_LEFT, 'right': lib.GBA_KEY_RIGHT,
            'l': lib.GBA_KEY_L, 'r': lib.GBA_KEY_R
        }
        
        # Human Keyboard Mapping (Tkinter keysyms)
        self.tk_key_map = {
            'z': lib.GBA_KEY_A, 'Z': lib.GBA_KEY_A,
            'x': lib.GBA_KEY_B, 'X': lib.GBA_KEY_B,
            'Return': lib.GBA_KEY_START, 'BackSpace': lib.GBA_KEY_SELECT,
            'Up': lib.GBA_KEY_UP, 'Down': lib.GBA_KEY_DOWN,
            'Left': lib.GBA_KEY_LEFT, 'Right': lib.GBA_KEY_RIGHT,
        }
        
        # State Management
        self.human_keys_state = 0
        self.agent_keys_state = 0
        self.action_queue = []
        self.action_delay = 0
        self.post_action_delay = 0 
        self.rolling_frames = collections.deque(maxlen=4)

    def human_press(self, keysym):
        val = self.tk_key_map.get(keysym)
        if val is not None:
            self.human_keys_state |= (1 << val)

    def human_release(self, keysym):
        val = self.tk_key_map.get(keysym)
        if val is not None:
            self.human_keys_state &= ~(1 << val)

    def queue_agent_actions(self, actions):
        """Receives a list of tuples: [('tap', 'z'), ('hold', 'up', 1.0)]"""
        self.action_queue = actions
        self.action_delay = 0
        self.post_action_delay = 0
        self.agent_keys_state = 0

    def process_agent_queue(self):
        """Pops and executes commands non-blockingly"""
        if self.post_action_delay > 0:
            self.post_action_delay -= 1
            return 
            
        if self.action_delay > 0:
            self.action_delay -= 1
            if self.action_delay == 0:
                self.agent_keys_state = 0 # Release key
                self.post_action_delay = 5 # Buffer 5 frames before next button press
        elif self.action_queue:
            action = self.action_queue.pop(0)
            cmd = action[0]
            key_val = self.key_map.get(action[1].lower())
            
            if key_val is not None:
                self.agent_keys_state = (1 << key_val)
                if cmd == 'tap':
                    self.action_delay = 5 # hold down for 5 frames
                elif cmd == 'hold':
                    # HOLD is temporarily disabled, converting to tap
                    self.action_delay = 5

    def run_frame(self, use_human_input=False):
        """Steps the emulator forward by exactly 1/60th of a second"""
        if use_human_input:
            self.core.set_keys(raw=self.human_keys_state)
        else:
            self.process_agent_queue()
            self.core.set_keys(raw=self.agent_keys_state)
            
        self.core.run_frame()

    def get_frame(self):
        pil_img = self.image.to_pil()
        img_np = np.array(pil_img)
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
    def add_to_buffer(self):
        self.rolling_frames.append(self.get_frame())
        
    def get_recent_frames(self):
        return list(self.rolling_frames)
        
    def save_state(self, filepath):
        """Saves an emulator state snapshot."""
        try:
            state_data = self.core.save_raw_state()
            if state_data:
                with open(filepath, 'wb') as f:
                    f.write(state_data)
        except Exception as e:
            print(f"Failed to save state: {e}")
                
    def load_state(self, filepath):
        """Loads an emulator state snapshot."""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    state_data = f.read()
                self.core.load_raw_state(state_data)
            except Exception as e:
                print(f"Failed to load state: {e}")
