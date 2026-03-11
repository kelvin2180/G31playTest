import time
import json
import cv2
import numpy as np
from google import genai
from google.genai import types
from emulator import EmulatorController

class VisionController:
    def __init__(self, emulator: EmulatorController):
        self.emulator = emulator
        # Initialize Gemini client - assumes GEMINI_API_KEY is in environment
        self.client = genai.Client()
        self.model_name = 'gemini-3.1-flash-lite-preview'
        
        self.state_schema = {
            "type": "OBJECT",
            "properties": {
                "state": {
                    "type": "STRING",
                    "enum": ["OVERWORLD", "BATTLE", "MENU", "DIALOGUE"],
                    "description": "The current game state. CRITICAL: Only set to 'DIALOGUE' if the VERY LAST image in the sequence (the most recent one) shows an active dialogue box. If dialogue ended during the sequence, use the state shown in the final frame (e.g. OVERWORLD)."
                },
                "reasoning": {
                    "type": "STRING",
                    "description": "Brief reason for this classification and planned actions."
                },
                "actions": {
                    "type": "ARRAY",
                    "description": "List of button taps to execute in sequence. DO NOT spam or mash 'A' multiple times in a row. Use a single 'A' or 'B' to advance text or interact.",
                    "items": {
                        "type": "STRING",
                        "enum": ["↑", "↓", "←", "→", "A", "B", "START", "SELECT", "NONE"]
                    }
                },
                "scratchpad_update": {
                    "type": "STRING",
                    "description": "Short note for your future self about what you just did or discovered."
                },
                "journal_update": {
                    "type": "STRING",
                    "description": "Optional. Write a new overarching goal here ONLY if you completed your previous long-term goal. Otherwise, leave blank or echo the current goal."
                }
            },
            "required": ["state", "reasoning", "actions", "scratchpad_update"]
        }

    def capture_frame(self):
        """Captures frame directly from the python mgba emulator instance."""
        return self.emulator.get_frame()

    def analyze_frames(self, frames, history_frames=None, prompt_text="Analyze these sequential game frames. What is the current game state?"):
        """Sends frames to Gemini and returns parsed JSON state + metrics."""
        if not frames:
            return {"state": "OVERWORLD", "reasoning": "No frames"}, 0, 0, 0, "{}"

        contents = []

        # Inject the historical frames if they exist
        if history_frames and len(history_frames) > 0:
            contents.append(f"IMAGES 1-{len(history_frames)}: Historical snapshots taken at the END of your last {len(history_frames)} turns:")
            for prev_frame in history_frames:
                rgb_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
                _, buffer_prev = cv2.imencode('.jpg', rgb_prev)
                contents.append(types.Part.from_bytes(data=buffer_prev.tobytes(), mime_type='image/jpeg'))
            contents.append(f"IMAGES {len(history_frames)+1}-{len(history_frames)+len(frames)}: Current sequential frames AFTER your recent actions:")
        else:
            contents.append(f"IMAGES 1-{len(frames)}: Current sequential frames:")

        # Encode current frames as JPEG to send to Gemini
        for frame in frames:
            # Convert BGR to RGB for Gemini/PIL
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode('.jpg', rgb_frame)
            contents.append(types.Part.from_bytes(data=buffer.tobytes(), mime_type='image/jpeg'))

        start_time = time.time()
        try:
            contents.append(prompt_text)
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=self.state_schema,
                    temperature=0.0
                )
            )
            
            latency = time.time() - start_time
            in_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
            out_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
            raw_json = response.text
            state_data = json.loads(raw_json)
            
            return state_data, latency, in_tokens, out_tokens, raw_json

        except Exception as e:
            print(f"Vision error: {e}")
            return {"state": "OVERWORLD", "reasoning": f"Error: {str(e)}"}, time.time() - start_time, 0, 0, "{}"
