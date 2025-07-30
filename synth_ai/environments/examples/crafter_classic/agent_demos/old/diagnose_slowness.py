#!/usr/bin/env python3
"""Diagnose Crafter environment slowness by profiling different components."""

import time
import logging
import crafter
import numpy as np
import asyncio
from typing import Dict, List, Any
import statistics
import cProfile
import pstats
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import patches
import synth_ai.environments.examples.crafter_classic.engine_deterministic_patch
import synth_ai.environments.examples.crafter_classic.engine_serialization_patch_v3
import synth_ai.environments.examples.crafter_classic.world_config_patch_simple


class CrafterProfiler:
    """Profile different aspects of Crafter environment."""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {
            "env_creation": [],
            "env_reset": [],
            "env_step": [],
            "env_render": [],
            "state_extraction": [],
            "serialization": [],
        }
        
    def profile_raw_crafter(self, num_episodes: int = 5, steps_per_episode: int = 50):
        """Profile the raw Crafter environment without any wrappers."""
        logger.info("=" * 60)
        logger.info("PROFILING RAW CRAFTER ENVIRONMENT")
        logger.info("=" * 60)
        
        for episode in range(num_episodes):
            logger.info(f"\nEpisode {episode + 1}/{num_episodes}")
            
            # Time environment creation
            start = time.time()
            env = crafter.Env(area=(64, 64), length=1000)
            self.timings["env_creation"].append(time.time() - start)
            logger.info(f"  Environment created in {self.timings['env_creation'][-1]:.3f}s")
            
            # Time reset
            start = time.time()
            obs = env.reset()
            self.timings["env_reset"].append(time.time() - start)
            logger.info(f"  Environment reset in {self.timings['env_reset'][-1]:.3f}s")
            
            # Time steps
            step_times = []
            render_times = []
            
            for step in range(steps_per_episode):
                # Time step
                action = np.random.randint(0, env.action_space.n)
                start = time.time()
                obs, reward, done, info = env.step(action)
                step_time = time.time() - start
                step_times.append(step_time)
                
                # Time render
                start = time.time()
                rendered = env.render()
                render_time = time.time() - start
                render_times.append(render_time)
                
                if done:
                    logger.info(f"  Episode ended at step {step + 1}")
                    break
                    
            self.timings["env_step"].extend(step_times)
            self.timings["env_render"].extend(render_times)
            
            mean_step = statistics.mean(step_times) if step_times else 0
            mean_render = statistics.mean(render_times) if render_times else 0
            logger.info(f"  Mean step time: {mean_step:.3f}s")
            logger.info(f"  Mean render time: {mean_render:.3f}s")
            
    def profile_with_patches(self, num_episodes: int = 3, steps_per_episode: int = 20):
        """Profile Crafter with all patches applied."""
        logger.info("\n" + "=" * 60)
        logger.info("PROFILING CRAFTER WITH PATCHES")
        logger.info("=" * 60)
        
        for episode in range(num_episodes):
            logger.info(f"\nEpisode {episode + 1}/{num_episodes}")
            
            # Create environment
            start = time.time()
            env = crafter.Env(area=(64, 64), length=1000)
            self.timings["env_creation"].append(time.time() - start)
            logger.info(f"  Environment created in {self.timings['env_creation'][-1]:.3f}s")
            
            # Reset
            start = time.time()
            obs = env.reset()
            self.timings["env_reset"].append(time.time() - start)
            logger.info(f"  Environment reset in {self.timings['env_reset'][-1]:.3f}s")
            
            # Test serialization
            if hasattr(env, 'save'):
                start = time.time()
                state = env.save()
                save_time = time.time() - start
                self.timings["serialization"].append(save_time)
                logger.info(f"  State saved in {save_time:.3f}s")
                
                # Test loading
                start = time.time()
                env.load(state)
                load_time = time.time() - start
                logger.info(f"  State loaded in {load_time:.3f}s")
            
            # Run steps
            for step in range(steps_per_episode):
                action = np.random.randint(0, env.action_space.n)
                start = time.time()
                obs, reward, done, info = env.step(action)
                self.timings["env_step"].append(time.time() - start)
                
                if done:
                    break
                    
    def profile_specific_operations(self):
        """Profile specific operations that might be slow."""
        logger.info("\n" + "=" * 60)
        logger.info("PROFILING SPECIFIC OPERATIONS")
        logger.info("=" * 60)
        
        env = crafter.Env(area=(64, 64), length=1000)
        env.reset()
        
        # Profile world generation
        logger.info("\n1. World Generation:")
        start = time.time()
        # Access internal world generation if available
        if hasattr(env, '_world'):
            logger.info(f"  World object exists: {env._world is not None}")
            logger.info(f"  World size: {env._area}")
        elapsed = time.time() - start
        logger.info(f"  Time to access world: {elapsed:.3f}s")
        
        # Profile rendering operations
        logger.info("\n2. Rendering Operations:")
        render_times = []
        for i in range(10):
            start = time.time()
            obs = env.render()
            render_times.append(time.time() - start)
        logger.info(f"  Mean render time: {statistics.mean(render_times):.3f}s")
        logger.info(f"  Render output shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}")
        
        # Profile achievement checking
        logger.info("\n3. Achievement System:")
        if hasattr(env, '_achievements'):
            start = time.time()
            # Simulate achievement checking
            for _ in range(10):
                env.step(0)  # No-op action
            elapsed = time.time() - start
            logger.info(f"  10 steps with achievement checking: {elapsed:.3f}s")
            
    def run_cprofile_analysis(self):
        """Run cProfile analysis on a typical episode."""
        logger.info("\n" + "=" * 60)
        logger.info("RUNNING CPROFILE ANALYSIS")
        logger.info("=" * 60)
        
        def run_episode():
            env = crafter.Env(area=(64, 64), length=1000)
            env.reset()
            for _ in range(50):
                action = np.random.randint(0, env.action_space.n)
                obs, reward, done, info = env.step(action)
                if done:
                    break
                    
        # Profile the episode
        profiler = cProfile.Profile()
        profiler.enable()
        run_episode()
        profiler.disable()
        
        # Print stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        logger.info("\nTop 20 time-consuming functions:")
        logger.info(s.getvalue())
        
    def print_summary(self):
        """Print timing summary."""
        logger.info("\n" + "=" * 60)
        logger.info("TIMING SUMMARY")
        logger.info("=" * 60)
        
        for operation, times in self.timings.items():
            if times:
                logger.info(f"\n{operation.upper()}:")
                logger.info(f"  Count: {len(times)}")
                logger.info(f"  Mean: {statistics.mean(times):.3f}s")
                logger.info(f"  Median: {statistics.median(times):.3f}s")
                logger.info(f"  Min: {min(times):.3f}s")
                logger.info(f"  Max: {max(times):.3f}s")
                if len(times) > 1:
                    logger.info(f"  Std Dev: {statistics.stdev(times):.3f}s")


async def check_import_time():
    """Check how long it takes to import Crafter environment."""
    logger.info("\n" + "=" * 60)
    logger.info("CHECKING IMPORT TIMES")
    logger.info("=" * 60)
    
    # Time importing the environment module
    start = time.time()
    import synth_ai.environments.examples.crafter_classic.environment
    env_import_time = time.time() - start
    logger.info(f"Importing crafter_classic.environment: {env_import_time:.3f}s")
    
    # Time importing the engine
    start = time.time()
    import synth_ai.environments.examples.crafter_classic.engine
    engine_import_time = time.time() - start
    logger.info(f"Importing crafter_classic.engine: {engine_import_time:.3f}s")
    
    # Time creating environment wrapper
    from synth_ai.environments.examples.crafter_classic.environment import CrafterClassicEnvironment
    from types import SimpleNamespace
    
    start = time.time()
    task = SimpleNamespace(initial_engine_snapshot={})
    env_wrapper = CrafterClassicEnvironment(task)
    wrapper_creation_time = time.time() - start
    logger.info(f"Creating CrafterClassicEnvironment wrapper: {wrapper_creation_time:.3f}s")
    
    # Time initialization
    start = time.time()
    await env_wrapper.initialize()
    init_time = time.time() - start
    logger.info(f"Initializing environment: {init_time:.3f}s")


def main():
    """Run all profiling tests."""
    profiler = CrafterProfiler()
    
    # Test 1: Profile raw Crafter
    profiler.profile_raw_crafter(num_episodes=2, steps_per_episode=20)
    
    # Test 2: Profile with patches
    profiler.profile_with_patches(num_episodes=2, steps_per_episode=20)
    
    # Test 3: Profile specific operations
    profiler.profile_specific_operations()
    
    # Test 4: Run cProfile analysis
    profiler.run_cprofile_analysis()
    
    # Test 5: Check import times
    asyncio.run(check_import_time())
    
    # Print summary
    profiler.print_summary()
    
    # Recommendations
    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 60)
    
    mean_step = statistics.mean(profiler.timings["env_step"]) if profiler.timings["env_step"] else 0
    mean_render = statistics.mean(profiler.timings["env_render"]) if profiler.timings["env_render"] else 0
    
    if mean_step > 0.1:
        logger.info("⚠️  Step operations are slow (>100ms). Consider:")
        logger.info("   - Check if debug mode is enabled")
        logger.info("   - Profile the step() method specifically")
        logger.info("   - Check for unnecessary computations")
        
    if mean_render > 0.05:
        logger.info("⚠️  Render operations are slow (>50ms). Consider:")
        logger.info("   - Caching rendered observations")
        logger.info("   - Optimizing image generation")
        
    if profiler.timings["serialization"] and statistics.mean(profiler.timings["serialization"]) > 0.5:
        logger.info("⚠️  Serialization is slow (>500ms). Consider:")
        logger.info("   - Using more efficient serialization format")
        logger.info("   - Reducing state size")


if __name__ == "__main__":
    main()