import threading
import queue
import time
import os
import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image
import tkinter.filedialog as filedialog

from vision import VisionController
from memory import Memory
from agents.orchestrator import Orchestrator
from emulator import EmulatorController

ui_queue = queue.Queue()
emulator_action_queue = queue.Queue() # Thread-safe queue for emulator actions (save/load)
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
            
        try:
            while not emulator_action_queue.empty():
                action, path = emulator_action_queue.get_nowait()
                if action == "save":
                    global_emulator.save_state(path)
                    ui_queue.put({"action_status": f"State saved to {os.path.basename(path)}"})
                elif action == "load":
                    global_emulator.load_state(path)
                    ui_queue.put({"action_status": f"State loaded from {os.path.basename(path)}"})
        except queue.Empty:
            pass
            
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


# Global variable for tick frequency (in seconds)
agent_tick_frequency = 15

# Global Metrics
total_model_calls = 0
total_in_tokens = 0
total_out_tokens = 0

def agent_thread():
    """Dynamic VLM Brain Loop"""
    global global_emulator, agent_paused, agent_tick_frequency
    global total_model_calls, total_in_tokens, total_out_tokens
    
    vision = None
    memory = Memory()
    orchestrator = Orchestrator()
    
    last_tick_time = 0
    
    # Historical memory buffers (last 10 turns)
    import collections
    history_frames = collections.deque(maxlen=10)   # Stores the final frame of each of the last 10 turns
    history_actions = collections.deque(maxlen=10)  # Stores the actions taken in each of the last 10 turns
    history_notes = collections.deque(maxlen=10)    # Stores the scratchpad notes from the last 10 turns
    
    last_direction_faced = "UNKNOWN"
    
    while True:
        # Dynamic rate limiting: wait until enough time has passed since last tick
        current_time = time.time()
        time_since_last = current_time - last_tick_time
        
        if time_since_last < agent_tick_frequency:
            time.sleep(0.5) # Sleep briefly and check again
            continue
            
        if agent_paused or global_emulator is None:
            time.sleep(0.5)
            continue
            
        last_tick_time = time.time()
        
        if vision is None:
            vision = VisionController(global_emulator)
            
        ui_queue.put({"action_status": "Thinking..."})
        
        # 1. Grab buffered frames
        frames = global_emulator.get_recent_frames()
        if not frames:
            continue
            
        # 2. VLM Perception
        journal_text = memory.get_journal()
        
        # Build history text
        history_text = ""
        if len(history_actions) == 0:
            history_text = "No history yet. This is your first turn."
        else:
            for i in range(len(history_actions)):
                history_text += f"[Turn -{len(history_actions) - i}] Actions: {history_actions[i]} | Note: {history_notes[i]}\n"
        
        vlm_prompt = (
            "You are an AI agent playing Pokémon.\n\n"
            "=== GAME MANUAL & RULES ===\n"
            "1. OVERWORLD: The player character is ALWAYS perfectly centered on the screen. The world moves around you when you walk.\n"
            "2. TURNING: If you press a directional button that is DIFFERENT from the direction you are currently facing, your character will ONLY turn in place. It will NOT move forward. You must press the button again to actually step forward.\n"
            "3. COLLISIONS: If you press a directional button but the background does not change in the subsequent frames, you bumped into an obstacle (wall, tree, NPC, ledge). Note that black straight line borders on the floor or walls are usually physical barriers/walls.\n"
            "4. CONTROLS:\n"
            "   - ↑, ↓, ←, →: Move your character or navigate menus.\n"
            "   - A: Interact with the object/NPC you are facing, confirm menu options, use attacks in battle. DO NOT mash or spam 'A' repeatedly.\n"
            "   - B: Cancel menus, back out, fast-forward dialogue, or run (if running shoes are unlocked).\n"
            "   - START: Open the main pause menu (Pokedex, Pokemon, Bag, Save).\n"
            "   - SELECT: Use registered items (like Bicycle or Rod).\n"
            "===========================\n\n"
        )
        
        num_history = len(history_frames)
        if num_history > 0:
            vlm_prompt += f"I am providing you with a filmstrip of images.\n"
            vlm_prompt += f"The FIRST {num_history} images are snapshots taken at the end of your LAST {num_history} turns (in chronological order).\n"
            vlm_prompt += f"The REMAINING {len(frames)} images show the CURRENT state resulting from your most recent actions (captured every 0.5 seconds, spanning 4 seconds total).\n"
        else:
            vlm_prompt += f"These {len(frames)} images show your CURRENT state (captured every 0.5 seconds, spanning 4 seconds total).\n"
            
        vlm_prompt += (
            f"\nBased on your previous actions, you should currently be facing: {last_direction_faced}.\n\n"
            "What is the current game state? If you took actions previously, did they work or are you blocked?\n"
            "What actions should you take next? Output a sequence of buttons to press.\n\n"
            "=== YOUR MEMORY FROM PAST TURNS ===\n"
            f"Master Journal (Long-term Goal): {journal_text}\n\n"
            f"Recent Turn History:\n{history_text}\n"
            "=======================================\n\n"
            "Based on the new frames and your history:\n"
            "1. Write a short Scratchpad note about what just happened.\n"
            "2. Decide your next actions.\n"
            "3. If you have fundamentally achieved the current Master Journal goal, OR if you just spoke to an NPC/read a sign that gave you a new objective (e.g. 'Deliver this parcel to Oak'), output a new 'journal_update' to set your next long-term objective. Otherwise, you can omit the journal update."
        )
        
        state_data, latency, in_tokens, out_tokens, raw_json = vision.analyze_frames(
            frames, 
            history_frames=list(history_frames), 
            prompt_text=vlm_prompt
        )
        
        total_model_calls += 1
        total_in_tokens += in_tokens
        total_out_tokens += out_tokens
        
        state = state_data.get("state", "OVERWORLD")
        reasoning = state_data.get("reasoning", "")
        llm_actions = state_data.get("actions", [])
        scratchpad = state_data.get("scratchpad_update", "")
        new_journal = state_data.get("journal_update", "")
        
        # Add to history buffers
        history_frames.append(frames[-1].copy())
        history_actions.append(llm_actions)
        history_notes.append(scratchpad)
        
        if new_journal and new_journal.strip():
            memory.update_journal(new_journal)
        
        # Update last direction faced based on the LLM's chosen actions
        directions = ["↑", "↓", "←", "→"]
        for action in reversed(llm_actions):
            if action in directions:
                last_direction_faced = action
                break
        
        # Calculate cost based on gemini-3.1-flash-lite-preview pricing: 
        # $0.25 per 1M input tokens, $1.50 per 1M output tokens
        cost_in = (total_in_tokens / 1_000_000) * 0.25
        cost_out = (total_out_tokens / 1_000_000) * 1.50
        total_cost = cost_in + cost_out
        
        metrics_str = f"Calls: {total_model_calls} | In: {total_in_tokens} | Out: {total_out_tokens} | Cost: ${total_cost:.5f}"
        
        # Create a single "filmstrip" image showing what the model saw to pass to UI
        filmstrip_pil = None
        try:
            filmstrip_images = []
            filmstrip_images.extend(list(history_frames))
            filmstrip_images.extend(frames)
            
            # Resize them to a uniform height to stack them horizontally cleanly
            h, w = filmstrip_images[0].shape[:2]
            scale = 80 / h  # Resize height to 80px
            resized_images = [cv2.resize(img, (int(w * scale), 80)) for img in filmstrip_images]
            
            # Combine horizontally
            filmstrip = np.hstack(resized_images)
            filmstrip_rgb = cv2.cvtColor(filmstrip, cv2.COLOR_BGR2RGB)
            filmstrip_pil = Image.fromarray(filmstrip_rgb)
        except Exception as e:
            print(f"Failed to create vision filmstrip: {e}")

        ui_queue.put({
            "state": state,
            "json": raw_json,
            "prompt": vlm_prompt,
            "metrics": metrics_str,
            "filmstrip": filmstrip_pil
        })
        
        # Log to terminal and filesystem for debugging
        log_dir = os.path.expanduser("~/fire/logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Save the prompt
        with open(os.path.join(log_dir, f"{timestamp}_prompt.txt"), "w") as f:
            f.write(vlm_prompt)
            f.write("\n\n=== JSON RESPONSE ===\n")
            f.write(raw_json)

        print("\n" + "="*50)
        print(f"[{time.strftime('%H:%M:%S')}] AGENT BRAIN TICK ({agent_tick_frequency}s)")
        print(f"Saved Prompt to ~/fire/logs/{timestamp}_prompt.txt")
        
        # Convert LLM action strings to emulator commands
        key_mapping = {
            "A": "z", "B": "x", "START": "enter", "SELECT": "backspace",
            "↑": "up", "↓": "down", "←": "left", "→": "right"
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
        self.geometry("1100x820")
        
        ctk.set_appearance_mode("Dark")
        
        # Layout Config
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0) # Bottom row for filmstrip
        
        # UI Elements - Top Left (Vision)
        self.vision_frame = ctk.CTkFrame(self)
        self.vision_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.load_rom_btn = ctk.CTkButton(self.vision_frame, text="Load GBA ROM", command=self.load_rom)
        self.load_rom_btn.pack(pady=5)
        
        # Pause/Resume Button
        self.pause_btn = ctk.CTkButton(self.vision_frame, text="Agent Paused (Manual Control)", fg_color="red", hover_color="darkred", command=self.toggle_pause)
        self.pause_btn.pack(pady=5)
        
        # Agent Tick Frequency Control
        self.freq_frame = ctk.CTkFrame(self.vision_frame, fg_color="transparent")
        self.freq_frame.pack(pady=5)
        
        ctk.CTkLabel(self.freq_frame, text="Agent Tick Rate:").pack(side="left", padx=(0, 5))
        self.freq_var = ctk.StringVar(value="15s (Safe 24/7)")
        self.freq_menu = ctk.CTkOptionMenu(
            self.freq_frame, 
            values=["5s (Fast Play)", "10s (Moderate)", "15s (Safe 24/7)", "30s (Slow)"],
            variable=self.freq_var,
            command=self.update_tick_frequency
        )
        self.freq_menu.pack(side="left")
        
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
        self.state_frame.grid_propagate(False) # Stop child widgets from resizing this frame
        
        self.state_label = ctk.CTkLabel(self.state_frame, text="State: UNKNOWN", font=("Arial", 24, "bold"))
        self.state_label.pack(pady=(10, 0))
        
        # Make the action label a text box so it can scroll and wrap without breaking the UI layout
        self.action_label = ctk.CTkTextbox(self.state_frame, font=("Arial", 14), wrap="word", height=80, fg_color="transparent")
        self.action_label.pack(pady=10, padx=10, fill="both", expand=True)
        self.action_label.insert("0.0", "Action: None")
        self.action_label.configure(state="disabled")
        
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
        
        # UI Elements - Bottom Row (Filmstrip)
        self.filmstrip_frame = ctk.CTkFrame(self)
        self.filmstrip_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew")
        ctk.CTkLabel(self.filmstrip_frame, text="Last VLM Vision Filmstrip (What the AI saw):", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(5, 0))
        self.filmstrip_label = ctk.CTkLabel(self.filmstrip_frame, text="Waiting for first agent tick...")
        self.filmstrip_label.pack(expand=True, fill="both", pady=5)
        
        # Bind keyboard events for manual control
        self.bind("<KeyPress>", self.on_key_press)
        self.bind("<KeyRelease>", self.on_key_release)
        
        # Start Threads
        threading.Thread(target=emulator_thread, daemon=True).start()
        threading.Thread(target=agent_thread, daemon=True).start()
        
        self.update_ui()

    def update_tick_frequency(self, choice):
        global agent_tick_frequency
        if "5s" in choice:
            agent_tick_frequency = 5
        elif "10s" in choice:
            agent_tick_frequency = 10
        elif "15s" in choice:
            agent_tick_frequency = 15
        elif "30s" in choice:
            agent_tick_frequency = 30
        ui_queue.put({"action_status": f"Tick rate updated to {agent_tick_frequency}s."})

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
                emulator_action_queue.put(("save", save_path))

    def load_state_ui(self):
        global global_emulator
        if global_emulator:
            load_path = filedialog.askopenfilename(filetypes=[("Save States", "*.ss"), ("All Files", "*.*")])
            if load_path:
                emulator_action_queue.put(("load", load_path))

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
                    self.action_label.configure(state="normal")
                    self.action_label.delete("0.0", "end")
                    self.action_label.insert("0.0", f"{data['action_status']}")
                    self.action_label.configure(state="disabled")
                    
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
                if "filmstrip" in data and data["filmstrip"] is not None:
                    img = data["filmstrip"]
                    ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))
                    self.filmstrip_label.configure(image=ctk_img, text="")
                    
        except queue.Empty:
            pass
        
        self.after(20, self.update_ui) # Refresh at ~50Hz

if __name__ == "__main__":
    app = AgentApp()
    app.mainloop()
