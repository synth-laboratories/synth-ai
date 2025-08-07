#!/usr/bin/env python3
"""
Check API key user details and credits.
"""

import httpx
import os

SYNTH_API_KEY = os.getenv("SYNTH_API_KEY", "sk_live_9592524d-be1b-48b2-aff7-976b277eac95")
SYNTH_API_URL = os.getenv("SYNTH_API_URL", "http://localhost:8000")

async def check_user_details():
    """Check user details for the API key."""
    headers = {
        "Authorization": f"Bearer {SYNTH_API_KEY}",
        "Content-Type": "application/json"
    }
    
    print(f"API Key: {SYNTH_API_KEY[:20]}...")
    print(f"API URL: {SYNTH_API_URL}")
    print("-" * 60)
    
    async with httpx.AsyncClient() as client:
        # Try to get user info via a simple request
        try:
            # Check warmup status (this should show auth details in logs)
            response = await client.get(
                f"{SYNTH_API_URL}/api/warmup/status/test",
                headers=headers
            )
            print(f"Auth check status: {response.status_code}")
            if response.status_code == 200:
                print("✓ API key is valid and authenticated")
            
            # Try to get user/account info if endpoint exists
            response = await client.get(
                f"{SYNTH_API_URL}/api/user/me",
                headers=headers
            )
            if response.status_code == 200:
                print(f"User details: {response.json()}")
            
            # Check credits/billing if endpoint exists
            response = await client.get(
                f"{SYNTH_API_URL}/api/billing/credits",
                headers=headers
            )
            if response.status_code == 200:
                print(f"Credits: {response.json()}")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(check_user_details())