#!/usr/bin/env python3
"""Create a working init state for Pokémon Red by skipping the title screen."""
import time
from pathlib import Path

from pyboy import PyBoy

ROM_PATH = Path("synth_ai/environments/examples/red/roms/pokemon_red.gb")
OUTPUT_STATE = Path("synth_ai/environments/examples/red/roms/working_init.state")

def main():
    print(f"Loading ROM: {ROM_PATH}")
    emulator = PyBoy(str(ROM_PATH), window="null")
    
    # Brute-force skip intro by pressing START then A ~200 times
    print("Skipping title screen and intro (brute-force)...")
    button_sequence = [
        ("start", 60),   # Press START to skip title
    ]
    
    # Press A 200 times to get through all dialogue and naming
    # This will cycle through letters but eventually hit confirmation
    for _ in range(200):
        button_sequence.append(("a", 20))
    
    for button, frames_to_hold in button_sequence:
        emulator.button_press(button)
        for _ in range(frames_to_hold):
            emulator.tick()
        emulator.button_release(button)
        emulator.tick()
        print(f"  Pressed {button} for {frames_to_hold} frames")
    
    # Let the game settle and clear any remaining text/menus
    print("Letting game settle and clearing text boxes...")
    # Press B multiple times to close any menus/text
    for _ in range(30):
        emulator.button_press("b")
        for _ in range(25):
            emulator.tick()
        emulator.button_release("b")
        for _ in range(15):
            emulator.tick()
    
    # Final settle
    print("Final settle...")
    for _ in range(180):
        emulator.tick()
    
    # Save state
    print(f"Saving init state to: {OUTPUT_STATE}")
    with open(OUTPUT_STATE, "wb") as f:
        emulator.save_state(f)
    
    emulator.stop()
    print("✓ Init state created successfully!")
    print(f"  Location: {OUTPUT_STATE}")
    print("\nRestart the Pokémon Red task app to use the new init state.")

if __name__ == "__main__":
    main()

