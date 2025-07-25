#!/usr/bin/env python3
"""Performance testing script for Crafter environment.

This script tests the performance of the Crafter environment service
to identify bottlenecks and measure response times.
"""

import asyncio
import time
import statistics
import json
import argparse
import logging
from typing import Dict, List, Any, Tuple
import aiohttp
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CrafterPerformanceTester:
    """Performance tester for Crafter environment service."""
    
    def __init__(self, service_url: str = "http://localhost:8901"):
        self.service_url = service_url
        self.timing_data: Dict[str, List[float]] = {
            "health_check": [],
            "initialize": [],
            "step": [],
            "terminate": [],
            "full_episode": [],
            "env_step_only": [],  # Time reported by environment
            "network_overhead": []  # Total time - env time
        }
        
    async def health_check(self, session: aiohttp.ClientSession) -> float:
        """Check service health and measure response time."""
        start_time = time.time()
        try:
            async with session.get(f"{self.service_url}/health") as response:
                data = await response.json()
                elapsed = time.time() - start_time
                logger.debug(f"Health check took {elapsed:.3f}s")
                return elapsed
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise
            
    async def initialize_environment(self, session: aiohttp.ClientSession) -> Tuple[str, float]:
        """Initialize a new Crafter environment and return env_id and timing."""
        start_time = time.time()
        
        payload = {
            "initial_state": {},
            "config": {
                "area": [64, 64],
                "length": 100
            }
        }
        
        try:
            async with session.post(
                f"{self.service_url}/env/CrafterClassic/initialize",
                json=payload
            ) as response:
                data = await response.json()
                elapsed = time.time() - start_time
                logger.debug(f"Initialize took {elapsed:.3f}s")
                return data["env_id"], elapsed
        except Exception as e:
            logger.error(f"Initialize failed: {e}")
            raise
            
    async def step_environment(
        self, 
        session: aiohttp.ClientSession, 
        env_id: str, 
        action: int = 0
    ) -> Tuple[Dict[str, Any], float]:
        """Execute a step in the environment and return response and timing."""
        start_time = time.time()
        
        payload = {
            "env_id": env_id,
            "action": {
                "tool_calls": [{
                    "tool": "interact",
                    "args": {"action": action}
                }]
            }
        }
        
        try:
            async with session.post(
                f"{self.service_url}/env/CrafterClassic/step",
                json=payload
            ) as response:
                data = await response.json()
                elapsed = time.time() - start_time
                logger.debug(f"Step took {elapsed:.3f}s")
                return data, elapsed
        except Exception as e:
            logger.error(f"Step failed: {e}")
            raise
            
    async def terminate_environment(
        self, 
        session: aiohttp.ClientSession, 
        env_id: str
    ) -> float:
        """Terminate an environment and return timing."""
        start_time = time.time()
        
        payload = {"env_id": env_id}
        
        try:
            async with session.post(
                f"{self.service_url}/env/CrafterClassic/terminate",
                json=payload
            ) as response:
                await response.json()
                elapsed = time.time() - start_time
                logger.debug(f"Terminate took {elapsed:.3f}s")
                return elapsed
        except Exception as e:
            logger.error(f"Terminate failed: {e}")
            raise
            
    async def run_episode(
        self, 
        session: aiohttp.ClientSession, 
        num_steps: int = 10
    ) -> Dict[str, Any]:
        """Run a complete episode and collect timing data."""
        episode_start = time.time()
        episode_data = {
            "steps": [],
            "total_time": 0,
            "env_id": None
        }
        
        # Initialize environment
        env_id, init_time = await self.initialize_environment(session)
        episode_data["env_id"] = env_id
        episode_data["init_time"] = init_time
        self.timing_data["initialize"].append(init_time)
        
        # Run steps
        for i in range(num_steps):
            action = np.random.randint(0, 17)  # Random action
            step_data, step_time = await self.step_environment(session, env_id, action)
            
            self.timing_data["step"].append(step_time)
            episode_data["steps"].append({
                "step": i,
                "time": step_time,
                "action": action,
                "done": step_data.get("done", False)
            })
            
            if step_data.get("done", False):
                logger.info(f"Episode terminated early at step {i}")
                break
                
        # Terminate environment
        term_time = await self.terminate_environment(session, env_id)
        episode_data["terminate_time"] = term_time
        self.timing_data["terminate"].append(term_time)
        
        # Calculate total episode time
        episode_time = time.time() - episode_start
        episode_data["total_time"] = episode_time
        self.timing_data["full_episode"].append(episode_time)
        
        return episode_data
        
    async def run_concurrent_episodes(
        self, 
        num_episodes: int = 5,
        num_steps: int = 10
    ) -> List[Dict[str, Any]]:
        """Run multiple episodes concurrently to test service under load."""
        async with aiohttp.ClientSession() as session:
            # Health check first
            health_time = await self.health_check(session)
            self.timing_data["health_check"].append(health_time)
            logger.info(f"✅ Service health check passed ({health_time:.3f}s)")
            
            # Run episodes concurrently
            logger.info(f"Running {num_episodes} concurrent episodes...")
            tasks = [
                self.run_episode(session, num_steps) 
                for _ in range(num_episodes)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out any failed episodes
            successful_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Episode {i} failed: {result}")
                else:
                    successful_results.append(result)
                    
            return successful_results
            
    def analyze_results(self, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze timing data and generate statistics."""
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "num_episodes": len(episodes),
            "timing_statistics": {}
        }
        
        # Calculate statistics for each timing category
        for category, times in self.timing_data.items():
            if times:
                analysis["timing_statistics"][category] = {
                    "count": len(times),
                    "mean": statistics.mean(times),
                    "median": statistics.median(times),
                    "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
                    "min": min(times),
                    "max": max(times),
                    "p95": np.percentile(times, 95) if len(times) > 1 else times[0],
                    "p99": np.percentile(times, 99) if len(times) > 1 else times[0]
                }
                
        # Analyze step timing patterns
        if self.timing_data["step"]:
            # Check if steps are getting slower over time
            step_times = self.timing_data["step"]
            first_quarter = step_times[:len(step_times)//4]
            last_quarter = step_times[-len(step_times)//4:]
            
            if first_quarter and last_quarter:
                analysis["performance_degradation"] = {
                    "first_quarter_mean": statistics.mean(first_quarter),
                    "last_quarter_mean": statistics.mean(last_quarter),
                    "slowdown_factor": statistics.mean(last_quarter) / statistics.mean(first_quarter)
                }
                
        return analysis
        
    def print_analysis(self, analysis: Dict[str, Any]):
        """Print a formatted analysis report."""
        print("\n" + "="*80)
        print("🏁 CRAFTER PERFORMANCE TEST RESULTS")
        print("="*80)
        print(f"Timestamp: {analysis['timestamp']}")
        print(f"Episodes tested: {analysis['num_episodes']}")
        print("\n📊 TIMING STATISTICS (seconds)")
        print("-"*80)
        
        # Print timing stats in a table format
        print(f"{'Operation':<20} {'Mean':>10} {'Median':>10} {'StdDev':>10} {'Min':>10} {'Max':>10} {'P95':>10}")
        print("-"*80)
        
        for category, stats in analysis["timing_statistics"].items():
            if stats["count"] > 0:
                print(f"{category:<20} "
                      f"{stats['mean']:>10.3f} "
                      f"{stats['median']:>10.3f} "
                      f"{stats['std_dev']:>10.3f} "
                      f"{stats['min']:>10.3f} "
                      f"{stats['max']:>10.3f} "
                      f"{stats['p95']:>10.3f}")
                      
        # Print performance degradation if available
        if "performance_degradation" in analysis:
            print("\n⚠️  PERFORMANCE DEGRADATION ANALYSIS")
            print("-"*80)
            pd = analysis["performance_degradation"]
            print(f"First quarter mean: {pd['first_quarter_mean']:.3f}s")
            print(f"Last quarter mean: {pd['last_quarter_mean']:.3f}s")
            print(f"Slowdown factor: {pd['slowdown_factor']:.2f}x")
            
        print("\n" + "="*80)
        
        # Identify potential issues
        print("\n🔍 POTENTIAL ISSUES:")
        issues = []
        
        step_stats = analysis["timing_statistics"].get("step", {})
        if step_stats and step_stats["mean"] > 1.0:
            issues.append(f"- Step operations are slow (mean: {step_stats['mean']:.3f}s)")
            
        if step_stats and step_stats["max"] > 5.0:
            issues.append(f"- Some steps taking very long (max: {step_stats['max']:.3f}s)")
            
        if "performance_degradation" in analysis:
            if analysis["performance_degradation"]["slowdown_factor"] > 1.5:
                issues.append(f"- Significant performance degradation detected ({analysis['performance_degradation']['slowdown_factor']:.2f}x slowdown)")
                
        init_stats = analysis["timing_statistics"].get("initialize", {})
        if init_stats and init_stats["mean"] > 2.0:
            issues.append(f"- Environment initialization is slow (mean: {init_stats['mean']:.3f}s)")
            
        if issues:
            for issue in issues:
                print(issue)
        else:
            print("✅ No significant performance issues detected")
            
        print("\n" + "="*80 + "\n")


async def main():
    """Main function to run performance tests."""
    parser = argparse.ArgumentParser(description="Test Crafter environment performance")
    parser.add_argument(
        "--service-url", 
        default="http://localhost:8901",
        help="URL of the environment service"
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=5,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=20,
        help="Number of steps per episode"
    )
    parser.add_argument(
        "--concurrent", 
        action="store_true",
        help="Run episodes concurrently"
    )
    parser.add_argument(
        "--save-results", 
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create tester
    tester = CrafterPerformanceTester(args.service_url)
    
    # Run tests
    try:
        if args.concurrent:
            logger.info(f"Running {args.episodes} concurrent episodes with {args.steps} steps each...")
            episodes = await tester.run_concurrent_episodes(args.episodes, args.steps)
        else:
            logger.info(f"Running {args.episodes} sequential episodes with {args.steps} steps each...")
            async with aiohttp.ClientSession() as session:
                # Health check
                health_time = await tester.health_check(session)
                tester.timing_data["health_check"].append(health_time)
                logger.info(f"✅ Service health check passed ({health_time:.3f}s)")
                
                # Run episodes sequentially
                episodes = []
                for i in range(args.episodes):
                    logger.info(f"Running episode {i+1}/{args.episodes}...")
                    episode = await tester.run_episode(session, args.steps)
                    episodes.append(episode)
                    
        # Analyze results
        analysis = tester.analyze_results(episodes)
        
        # Save results if requested
        if args.save_results:
            with open(args.save_results, 'w') as f:
                json.dump({
                    "episodes": episodes,
                    "analysis": analysis,
                    "raw_timing_data": tester.timing_data
                }, f, indent=2)
            logger.info(f"Results saved to {args.save_results}")
            
        # Print analysis
        tester.print_analysis(analysis)
        
    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())