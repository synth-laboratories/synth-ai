#!/usr/bin/env python3
"""Test script that replicates Pokemon Red task app inference request exactly."""
import asyncio
import base64
import json
import os

import httpx

# Try to load from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

async def test_pokemon_inference():
    """Replicate the exact Pokemon Red task app inference request."""
    api_key = os.getenv("SYNTH_API_KEY")
    if not api_key:
        print("ERROR: SYNTH_API_KEY not set")
        return
    
    # Use dev endpoint (not prod!)
    inference_url = "https://synth-laboratories-dev--learning-v2-service-fastapi-app.modal.run/chat/completions"
    
    # Create a Pokemon Red game screenshot (160x144 Game Boy resolution)
    # This is a placeholder - in real evaluation, it would be the actual game frame
    # For now, use a simple test image to verify the API works
    image_base64 = base64.b64decode(
        "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"
    )
    image_data_url = f"data:image/png;base64,{base64.b64encode(image_base64).decode('utf-8')}"
    
    # Build state summary (simulating Pokemon Red observation)
    state_summary = "State summary: {'position': 'Map26:(3,6)', 'badges_earned': 0, 'badges_bitfield': 0, 'hp_status': 'HP: Unknown', 'party_level': 0, 'party_xp': 0, 'in_battle': False, 'step_count': 0, 'reward_last_step': 0.0, 'total_reward': 0.0, 'terminated': False, 'map_id': 38, 'player_x': 3, 'player_y': 6, 'party_count': 0, 'party_pokemon': [], 'battle_outcome': 0, 'text_box_active': True, 'enemy_hp_current': 0, 'enemy_hp_max': 0, 'enemy_hp_percentage': 0.0, 'badges': 0}"
    
    # Build user content (vision mode with text)
    user_content = [
        {"type": "text", "text": state_summary},
        {"type": "image_url", "image_url": {"url": image_data_url}}
    ]
    
    # Exact payload from Pokemon Red task app
    payload = {
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are controlling Pokémon Red, a classic Game Boy game. You can see the game screen in the images provided. "
                    "Your goal is to make progress in the game. Use the execute_sequence tool to press buttons. "
                    "Choose appropriate button presses based on what you see in the game screen. "
                    "Always respond with exactly one tool call in the format: <tool_call>{\"name\": \"execute_sequence\", \"arguments\": {...}}</tool_call>"
                ),
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "execute_sequence",
                    "description": "Execute multiple button presses in sequence. More efficient than separate calls. Recommended: 5-10 actions per call.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "actions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "button": {
                                            "type": "string",
                                            "enum": ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"],
                                            "description": "Game Boy button to press"
                                        },
                                        "frames": {
                                            "type": "integer",
                                            "minimum": 1,
                                            "maximum": 120,
                                            "description": "Number of frames to hold the button (30 frames = 0.5 seconds)"
                                        }
                                    },
                                    "required": ["button", "frames"]
                                },
                                "minItems": 1,
                                "maxItems": 20,
                                "description": "Sequence of button presses to execute"
                            }
                        },
                        "required": ["actions"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "press_button",
                    "description": "Press a single Game Boy button for N frames (use execute_sequence for multiple actions)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "button": {"type": "string", "enum": ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"]},
                            "frames": {"type": "integer", "minimum": 1, "maximum": 120},
                        },
                        "required": ["button"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "execute_sequence"}},
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 2048,
        "thinking_budget": 0
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    print(f"{'='*80}")
    print("POKEMON RED INFERENCE TEST")
    print(f"{'='*80}")
    print(f"Endpoint: {inference_url}")
    print(f"Model: {payload['model']}")
    print("\nPayload (formatted):")
    print(json.dumps(payload, indent=2)[:2000])
    print(f"\n{'='*80}\n")
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            print("Making request...")
            resp = await client.post(inference_url, json=payload, headers=headers)
            print(f"\n{'='*80}")
            print("RESPONSE")
            print(f"{'='*80}")
            print(f"Status: {resp.status_code}")
            
            if resp.status_code != 200:
                print(f"Error: {resp.text[:500]}")
                return
            
            data = resp.json()
            print(f"\nResponse keys: {list(data.keys())}")
            
            choices = data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                print(f"\nMessage keys: {list(message.keys())}")
                print("\nFull message:")
                print(json.dumps(message, indent=2)[:2000])
                
                tool_calls = message.get("tool_calls", [])
                print(f"\n{'='*80}")
                print("TOOL CALLS ANALYSIS")
                print(f"{'='*80}")
                print(f"Number of tool calls: {len(tool_calls)}")
                
                if tool_calls:
                    print("\n✅ SUCCESS! Tool calls received:")
                    for i, tc in enumerate(tool_calls):
                        print(f"\n  Tool call {i+1}:")
                        print(f"    ID: {tc.get('id')}")
                        print(f"    Type: {tc.get('type')}")
                        func = tc.get("function", {})
                        print(f"    Function name: {func.get('name')}")
                        args = func.get('arguments', '')
                        print(f"    Arguments: {args}")
                        try:
                            args_dict = json.loads(args) if isinstance(args, str) else args
                            print(f"    Parsed args: {json.dumps(args_dict, indent=4)}")
                        except:
                            print("    (Could not parse arguments)")
                else:
                    print("\n⚠️  NO TOOL CALLS IN STRUCTURED FORMAT")
                    content = message.get('content', '')
                    print(f"Message content: {content[:500]}")
                    
                    # Try to parse XML tool calls from content
                    import re
                    xml_pattern = r'<tool_call>\s*({.*?})\s*</tool_call>'
                    matches = re.findall(xml_pattern, content, re.DOTALL)
                    if matches:
                        print("\n✅ FOUND XML TOOL CALLS IN CONTENT!")
                        print(f"   Found {len(matches)} tool call(s)")
                        for i, match in enumerate(matches):
                            try:
                                tool_data = json.loads(match)
                                print(f"\n   Tool call {i+1} (parsed from XML):")
                                print(f"     Name: {tool_data.get('name')}")
                                print(f"     Arguments: {json.dumps(tool_data.get('arguments', {}), indent=6)}")
                            except Exception as e:
                                print(f"     Error parsing: {e}")
                                print(f"     Raw: {match[:200]}")
            else:
                print("\nNo choices in response")
                print(f"Full response: {json.dumps(data, indent=2)[:1000]}")
                
        except Exception as e:
            print(f"\nException: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_pokemon_inference())

