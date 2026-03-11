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
        self.model_name = 'gemini-2.0-flash-lite-preview-02-05'
        
        self.state_schema = {
            "type": "OBJECT",
            "properties": {
                "state": {
                    "type": "STRING",
                    "enum": ["OVERWORLD", "BATTLE", "MENU", "DIALOGUE"]
                },
                "reasoning": {
                    "type": "STRING",
                    "description": "Brief reason for this classification."
                }
            },
            "required": ["state", "reasoning"]
        }

    def capture_frame(self):
        """Captures frame directly from the python mgba emulator instance."""
        return self.emulator.get_frame()

    def analyze_frames(self, frames, prompt_text="Analyze these sequential game frames. What is the current game state?"):
        """Sends frames to Gemini and returns parsed JSON state + metrics."""
        if not frames:
            return {"state": "OVERWORLD", "reasoning": "No frames"}, 0, 0, "{}"

        # Encode frames as JPEG to send to Gemini
        pil_images = []
        for frame in frames:
            # Convert BGR to RGB for Gemini/PIL
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode('.jpg', rgb_frame)
            
            part = types.Part.from_bytes(
                data=buffer.tobytes(),
                mime_type='image/jpeg'
            )
            pil_images.append(part)

        start_time = time.time()
        try:
            contents = pil_images + [prompt_text]
            
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
            tokens = response.usage_metadata.total_token_count if response.usage_metadata else 0
            raw_json = response.text
            state_data = json.loads(raw_json)
            
            return state_data, latency, tokens, raw_json

        except Exception as e:
            print(f"Vision error: {e}")
            return {"state": "OVERWORLD", "reasoning": f"Error: {str(e)}"}, time.time() - start_time, 0, "{}"
