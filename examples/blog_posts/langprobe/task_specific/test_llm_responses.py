#!/usr/bin/env python3
"""Test what LLM responses look like for debugging parsing."""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

import httpx

load_dotenv()

API_KEY = os.getenv("ENVIRONMENT_API_KEY") or os.getenv("SYNTH_API_KEY") or "sk_env_30c78a78"

async def test_crafter_response():
    """Test a single Crafter rollout and show the raw response."""
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
    
    payload = {
        "run_id": "test_debug",
        "env": {
            "env_name": "crafter",
            "seed": 0,
            "config": {},
        },
        "policy": {
            "policy_id": "test",
            "config": {
                "model": "openai/gpt-oss-120b",
                "provider": "groq",
                "temperature": 0.0,
                "max_completion_tokens": 512,
            },
        },
        "ops": [],
        "mode": "eval",
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "http://127.0.0.1:8116/rollout",
            json=payload,
            headers=headers,
        )
        
        if response.status_code == 200:
            data = response.json()
            trajectories = data.get("trajectories", [])
            if trajectories:
                steps = trajectories[0].get("steps", [])
                if steps:
                    info = steps[0].get("info", {})
                    response_text = info.get("response_text", "")
                    predicted_action = info.get("predicted_action", None)
                    reward = steps[0].get("reward", 0.0)
                    
                    print("="*80)
                    print("CRAFTER LLM RESPONSE DEBUG")
                    print("="*80)
                    print(f"Response text: {repr(response_text)}")
                    print(f"Predicted action: {predicted_action}")
                    print(f"Reward: {reward}")
                    print("="*80)
                    return response_text, predicted_action, reward
    
    return None, None, 0.0

async def test_sokoban_response():
    """Test a single Sokoban rollout and show the raw response."""
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
    
    payload = {
        "run_id": "test_debug",
        "env": {
            "env_name": "sokoban",
            "seed": 0,
            "config": {},
        },
        "policy": {
            "policy_id": "test",
            "config": {
                "model": "openai/gpt-oss-120b",
                "provider": "groq",
                "temperature": 0.0,
                "max_completion_tokens": 512,
            },
        },
        "ops": [],
        "mode": "eval",
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "http://127.0.0.1:8117/rollout",
            json=payload,
            headers=headers,
        )
        
        if response.status_code == 200:
            data = response.json()
            trajectories = data.get("trajectories", [])
            if trajectories:
                steps = trajectories[0].get("steps", [])
                if steps:
                    info = steps[0].get("info", {})
                    response_text = info.get("response_text", "")
                    predicted_action = info.get("predicted_action", None)
                    reward = steps[0].get("reward", 0.0)
                    
                    print("="*80)
                    print("SOKOBAN LLM RESPONSE DEBUG")
                    print("="*80)
                    print(f"Response text: {repr(response_text)}")
                    print(f"Predicted action: {predicted_action}")
                    print(f"Reward: {reward}")
                    print("="*80)
                    return response_text, predicted_action, reward
    
    return None, None, 0.0

async def test_verilog_response():
    """Test a single Verilog rollout and show the raw response."""
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
    
    payload = {
        "run_id": "test_debug",
        "env": {
            "env_name": "verilog",
            "seed": 0,
            "config": {},
        },
        "policy": {
            "policy_id": "test",
            "config": {
                "model": "openai/gpt-oss-120b",
                "provider": "groq",
                "temperature": 0.0,
                "max_completion_tokens": 512,
            },
        },
        "ops": [],
        "mode": "eval",
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "http://127.0.0.1:8118/rollout",
            json=payload,
            headers=headers,
        )
        
        if response.status_code == 200:
            data = response.json()
            trajectories = data.get("trajectories", [])
            if trajectories:
                steps = trajectories[0].get("steps", [])
                if steps:
                    info = steps[0].get("info", {})
                    response_text = info.get("response_text", "")
                    tool_call = info.get("tool_call", None)
                    reward = steps[0].get("reward", 0.0)
                    
                    print("="*80)
                    print("VERILOG LLM RESPONSE DEBUG")
                    print("="*80)
                    print(f"Response text: {repr(response_text)}")
                    print(f"Tool call: {tool_call}")
                    print(f"Reward: {reward}")
                    print("="*80)
                    return response_text, tool_call, reward
    
    return None, None, 0.0

async def main():
    print("\nTesting LLM responses with gpt-oss-120b...\n")
    
    await test_crafter_response()
    print()
    await test_sokoban_response()
    print()
    await test_verilog_response()

if __name__ == "__main__":
    asyncio.run(main())


