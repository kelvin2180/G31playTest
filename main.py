import threading
import queue
import time
import os
import cv2
import customtkinter as ctk
from PIL import Image
import tkinter.filedialog as filedialog

from vision import VisionController
from memory import Memory
from agents.orchestrator import Orchestrator
from emulator import EmulatorController

ui_queue = queue.Queue()
global_emulator = None
agent_paused = True # Start paused so human can play

def emulator_thread():
    """Real-time 60FPS Game Loop"""
    global global_emulator, agent_paused
    frame_count = 0
    
    while True:
        if global_emulator is None:
            time.sleep(0.1)
            continue
            
        start_time = time.time()
        
        # Step emulator forward 1 frame
        global_emulator.run_frame(use_human_input=agent_paused)
        frame_count += 1
        
        # Save a frame to the AI buffer every ~0.5 seconds (30 frames)
        if frame_count % 30 == 0:
            global_emulator.add_to_buffer()
            
        # Push frame to UI (~30 FPS to save CPU)
        if frame_count % 2 == 0:
            img = global_emulator.get_frame()
            rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ui_queue.put({"image": Image.fromarray(rgb_frame)})
            
        # Sleep to maintain 60FPS speed
        elapsed = time.time() - start_time
        sleep_time = (1.0 / 60.0) - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def agent_thread():
    """15-Second VLM Brain Loop"""
    global global_emulator, agent_paused
    vision = None
    memory = Memory()
    orchestrator = Orchestrator()
    
    while True:
        # Rate limit: 15 seconds. (We do it in 1s chunks so we don't freeze on shutdown)
        for _ in range(15):
            time.sleep(1)
            # If we unpause mid-sleep, we keep waiting. 
        
        if agent_paused or global_emulator is None:
            continue
            
        if vision is None:
            vision = VisionController(global_emulator)
            
        ui_queue.put({"action_status": "Thinking..."})
        
        # 1. Grab buffered frames
        frames = global_emulator.get_recent_frames()
        if not frames:
            continue
            
        # 2. VLM Perception
        journal_text = memory.get_journal()
        scratchpad_text = memory.get_scratchpad()
        vlm_prompt = (
            "Analyze these 4 sequential game frames (taken over the last 2 seconds). "
            "What is the current game state?\n"
            "What actions should we take next? Output a sequence of buttons to press.\n\n"
            f"Context - Master Journal: {journal_text}\n"
            f"Context - Scratchpad: {scratchpad_text}"
        )
        state_data, latency, tokens, raw_json = vision.analyze_frames(frames, prompt_text=vlm_prompt)
        state = state_data.get("state", "OVERWORLD")
        reasoning = state_data.get("reasoning", "")
        llm_actions = state_data.get("actions", [])
        scratchpad = state_data.get("scratchpad_update", "")
        
        ui_queue.put({
            "state": state,
            "json": raw_json,
            "prompt": vlm_prompt,
            "metrics": f"Latency: {latency:.2f}s | Tokens: {tokens}"
        })
        
        # Convert LLM action strings to emulator commands
        key_mapping = {
            "A": "z", "B": "x", "START": "enter", "SELECT": "backspace",
            "UP": "up", "DOWN": "down", "LEFT": "left", "RIGHT": "right"
        }
        
        actions = []
        if state == "DIALOGUE":
            # Fast-path for dialogue to avoid slow reading
            actions = [('tap', 'x'), ('tap', 'z'), ('tap', 'x'), ('tap', 'z')]
            scratchpad = "Mashing through dialogue."
        else:
            for a in llm_actions:
                if a != "NONE":
                    actions.append(('tap', key_mapping.get(a, a.lower())))
            if not actions:
                actions = [('tap', 'x')] # safe fallback
        
        memory.update_scratchpad(scratchpad)
        
        ui_queue.put({
            "action_status": f"Agent Decided: {llm_actions}\nReasoning: {reasoning}",
            "scratchpad": scratchpad,
            "journal": memory.get_journal()
        })
        
        # 4. Queue up actions asynchronously for the emulator loop
        global_emulator.queue_agent_actions(actions)


class AgentApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Pokemon AI Agent Monitor")
        self.geometry("1100x700")
        
        ctk.set_appearance_mode("Dark")
        
        # Layout Config
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # UI Elements - Top Left (Vision)
        self.vision_frame = ctk.CTkFrame(self)
        self.vision_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.load_rom_btn = ctk.CTkButton(self.vision_frame, text="Load GBA ROM", command=self.load_rom)
        self.load_rom_btn.pack(pady=5)
        
        # Pause/Resume Button
        self.pause_btn = ctk.CTkButton(self.vision_frame, text="Agent Paused (Manual Control)", fg_color="red", hover_color="darkred", command=self.toggle_pause)
        self.pause_btn.pack(pady=5)
        
        # Save/Load State Buttons
        self.save_load_frame = ctk.CTkFrame(self.vision_frame, fg_color="transparent")
        self.save_load_frame.pack(pady=5)
        self.save_btn = ctk.CTkButton(self.save_load_frame, text="Save State", command=self.save_state_ui, width=120)
        self.save_btn.pack(side="left", padx=5)
        self.load_btn = ctk.CTkButton(self.save_load_frame, text="Load State", command=self.load_state_ui, width=120)
        self.load_btn.pack(side="left", padx=5)
        
        self.vision_label = ctk.CTkLabel(self.vision_frame, text="Select a ROM to start...")
        self.vision_label.pack(expand=True, fill="both")
        
        # UI Elements - Bottom Left (Agent State)
        self.state_frame = ctk.CTkFrame(self)
        self.state_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.state_label = ctk.CTkLabel(self.state_frame, text="State: UNKNOWN", font=("Arial", 24, "bold"))
        self.state_label.pack(pady=10)
        self.action_label = ctk.CTkLabel(self.state_frame, text="Action: None", font=("Arial", 14))
        self.action_label.pack(pady=10)
        
        # UI Elements - Top Right (Memory)
        self.mem_frame = ctk.CTkFrame(self)
        self.mem_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(self.mem_frame, text="Master Journal", font=("Arial", 14, "bold")).pack()
        self.journal_text = ctk.CTkTextbox(self.mem_frame, height=80)
        self.journal_text.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(self.mem_frame, text="Scratchpad", font=("Arial", 14, "bold")).pack()
        self.scratchpad_text = ctk.CTkTextbox(self.mem_frame, height=80)
        self.scratchpad_text.pack(fill="x", padx=5, pady=5)
        
        # UI Elements - Bottom Right (LLM & Metrics)
        self.llm_frame = ctk.CTkFrame(self)
        self.llm_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        self.metrics_label = ctk.CTkLabel(self.llm_frame, text="Latency: 0s | Tokens: 0", font=("Arial", 12, "bold"))
        self.metrics_label.pack(pady=5)
        ctk.CTkLabel(self.llm_frame, text="LLM Trace (Prompt & Response)").pack()
        self.trace_text = ctk.CTkTextbox(self.llm_frame, wrap="word")
        self.trace_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Bind keyboard events for manual control
        self.bind("<KeyPress>", self.on_key_press)
        self.bind("<KeyRelease>", self.on_key_release)
        
        # Start Threads
        threading.Thread(target=emulator_thread, daemon=True).start()
        threading.Thread(target=agent_thread, daemon=True).start()
        
        self.update_ui()

    def toggle_pause(self):
        global agent_paused
        agent_paused = not agent_paused
        if agent_paused:
            self.pause_btn.configure(text="Agent Paused (Manual Control)", fg_color="red", hover_color="darkred")
            ui_queue.put({"action_status": "Paused. Play manually with Arrows, Z, X, Enter."})
        else:
            self.pause_btn.configure(text="Agent Active (Autonomous)", fg_color="green", hover_color="darkgreen")
            ui_queue.put({"action_status": "Agent active. Waiting for next 15s cycle..."})

    def on_key_press(self, event):
        if global_emulator and agent_paused:
            global_emulator.human_press(event.keysym)
            
    def on_key_release(self, event):
        if global_emulator and agent_paused:
            global_emulator.human_release(event.keysym)

    def load_rom(self):
        global global_emulator
        rom_path = filedialog.askopenfilename(filetypes=[("GBA ROMs", "*.gba"), ("All Files", "*.*")])
        if rom_path:
            self.load_rom_btn.pack_forget() # Hide button
            global_emulator = EmulatorController(rom_path)
            # Note: We don't wake up the agent thread here. 
            # It wakes up automatically, and you control it via the Pause button.

    def save_state_ui(self):
        global global_emulator
        if global_emulator:
            save_path = filedialog.asksaveasfilename(defaultextension=".ss", filetypes=[("Save States", "*.ss")])
            if save_path:
                global_emulator.save_state(save_path)
                ui_queue.put({"action_status": f"State saved to {os.path.basename(save_path)}"})

    def load_state_ui(self):
        global global_emulator
        if global_emulator:
            load_path = filedialog.askopenfilename(filetypes=[("Save States", "*.ss"), ("All Files", "*.*")])
            if load_path:
                global_emulator.load_state(load_path)
                ui_queue.put({"action_status": f"State loaded from {os.path.basename(load_path)}"})

    def update_ui(self):
        try:
            while not ui_queue.empty():
                data = ui_queue.get_nowait()
                
                if "image" in data:
                    img = data["image"]
                    img = img.resize((480, 320), Image.Resampling.NEAREST)
                    ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(480, 320))
                    self.vision_label.configure(image=ctk_img, text="")
                
                if "state" in data:
                    self.state_label.configure(text=f"State: {data['state']}")
                if "action_status" in data:
                    self.action_label.configure(text=f"{data['action_status']}")
                    
                if "journal" in data:
                    self.journal_text.delete("0.0", "end")
                    self.journal_text.insert("0.0", data["journal"])
                if "scratchpad" in data:
                    self.scratchpad_text.delete("0.0", "end")
                    self.scratchpad_text.insert("0.0", data["scratchpad"])
                    
                if "json" in data:
                    self.trace_text.delete("0.0", "end")
                    prompt_str = data.get("prompt", "No prompt provided.")
                    trace_content = f"=== SENT PROMPT ===\n{prompt_str}\n\n=== RECEIVED JSON ===\n{data['json']}"
                    self.trace_text.insert("0.0", trace_content)
                if "metrics" in data:
                    self.metrics_label.configure(text=data["metrics"])
                    
        except queue.Empty:
            pass
        
        self.after(20, self.update_ui) # Refresh at ~50Hz

if __name__ == "__main__":
    app = AgentApp()
    app.mainloop()
