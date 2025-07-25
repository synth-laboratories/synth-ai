#!/usr/bin/env python3
"""Detailed single-step timing test for Crafter environment."""

import asyncio
import time
import aiohttp
import logging
import json

# Set up logging to see service logs
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_single_environment():
    """Test a single environment with detailed timing."""
    service_url = "http://localhost:8901"
    
    async with aiohttp.ClientSession() as session:
        # 1. Health check
        logger.info("=" * 60)
        logger.info("1. HEALTH CHECK")
        start = time.time()
        async with session.get(f"{service_url}/health") as response:
            data = await response.json()
            elapsed = time.time() - start
            logger.info(f"Health check completed in {elapsed:.3f}s")
            logger.info(f"Response: {data}")
        
        # 2. Initialize environment
        logger.info("=" * 60)
        logger.info("2. INITIALIZE ENVIRONMENT")
        init_start = time.time()
        
        payload = {
            "initial_state": {},
            "config": {
                "area": [64, 64],
                "length": 100
            }
        }
        
        async with session.post(
            f"{service_url}/env/CrafterClassic/initialize",
            json=payload
        ) as response:
            data = await response.json()
            init_elapsed = time.time() - init_start
            env_id = data["env_id"]
            logger.info(f"Initialize completed in {init_elapsed:.3f}s")
            logger.info(f"Env ID: {env_id}")
            logger.info(f"Observation keys: {list(data['observation'].keys())}")
            
        # 3. Perform 10 steps with detailed timing
        logger.info("=" * 60)
        logger.info("3. PERFORMING STEPS")
        
        for i in range(10):
            logger.info(f"\n--- Step {i+1}/10 ---")
            step_start = time.time()
            
            # Prepare request
            action = i % 17  # Cycle through actions
            payload = {
                "env_id": env_id,
                "action": {
                    "tool_calls": [{
                        "tool": "interact",
                        "args": {"action": action}
                    }]
                }
            }
            
            # Log request details
            logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")
            
            # Make request
            async with session.post(
                f"{service_url}/env/CrafterClassic/step",
                json=payload
            ) as response:
                step_data = await response.json()
                step_elapsed = time.time() - step_start
                
                # Log response
                logger.info(f"Step {i+1} completed in {step_elapsed:.3f}s")
                logger.info(f"Action: {action}, Done: {step_data.get('done', False)}")
                
                if 'observation' in step_data:
                    obs = step_data['observation']
                    if isinstance(obs, dict):
                        logger.debug(f"Observation keys: {list(obs.keys())}")
                        if 'num_steps_taken' in obs:
                            logger.info(f"Game steps: {obs['num_steps_taken']}")
                            
                # Add a small delay between steps to avoid overwhelming the service
                await asyncio.sleep(0.1)
                
        # 4. Terminate environment
        logger.info("=" * 60)
        logger.info("4. TERMINATE ENVIRONMENT")
        term_start = time.time()
        
        payload = {"env_id": env_id}
        async with session.post(
            f"{service_url}/env/CrafterClassic/terminate",
            json=payload
        ) as response:
            await response.json()
            term_elapsed = time.time() - term_start
            logger.info(f"Terminate completed in {term_elapsed:.3f}s")
            
        logger.info("=" * 60)
        logger.info("TEST COMPLETE")


if __name__ == "__main__":
    asyncio.run(test_single_environment())