#!/usr/bin/env python3
"""
Test script for Gemini fine-tuning data generation
==================================================
Quick test to verify the generation pipeline works correctly.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src"))

from generate_ft_data_gemini import GenerationConfig, generate_trajectory, filter_high_quality_trajectories, convert_to_vertex_ai_format


async def test_single_trajectory():
    """Test generating a single trajectory."""
    print("🧪 Testing single trajectory generation...")
    
    # Create test config
    config = GenerationConfig()
    config.num_rollouts = 1
    config.max_turns = 10  # Short episode for testing
    config.model_name = "gemini-2.5-flash"
    
    # Generate one trajectory
    trajectory = await generate_trajectory(config, 0)
    
    if trajectory:
        print("✅ Successfully generated trajectory")
        print(f"   Model: {trajectory['model']}")
        print(f"   Actions taken: {len(trajectory['actions'])}")
        print(f"   Achievements: {sum(1 for v in trajectory['achievements'].values() if v)}")
        print(f"   Score: {trajectory['final_score']}")
        
        # Test filtering
        filtered = filter_high_quality_trajectories([trajectory], min_score=0, min_achievements=0)
        print(f"   Passed filter: {len(filtered) > 0}")
        
        # Test conversion
        test_output = Path("test_gemini_ft.jsonl")
        num_examples = convert_to_vertex_ai_format([trajectory], test_output)
        print(f"   Generated examples: {num_examples}")
        
        # Cleanup
        if test_output.exists():
            test_output.unlink()
        
        return True
    else:
        print("❌ Failed to generate trajectory")
        return False


async def test_multiple_trajectories():
    """Test generating multiple trajectories."""
    print("\n🧪 Testing multiple trajectory generation...")
    
    # Create test config
    config = GenerationConfig()
    config.num_rollouts = 3
    config.max_turns = 15
    config.model_name = "gemini-2.5-flash"
    
    # Generate trajectories
    from generate_ft_data_gemini import generate_all_trajectories
    trajectories = await generate_all_trajectories(config)
    
    print(f"✅ Generated {len(trajectories)} trajectories")
    
    # Analyze results
    total_achievements = 0
    total_examples = 0
    
    for traj in trajectories:
        achievements = sum(1 for v in traj['achievements'].values() if v)
        total_achievements += achievements
        total_examples += len(traj['llm_calls'])
    
    if trajectories:
        print(f"   Average achievements: {total_achievements / len(trajectories):.1f}")
        print(f"   Total LLM calls: {total_examples}")
        
        # Test filtering
        filtered = filter_high_quality_trajectories(trajectories, min_score=1.0, min_achievements=1)
        print(f"   High quality: {len(filtered)}/{len(trajectories)}")
    
    return len(trajectories) > 0


def test_config_loading():
    """Test configuration loading."""
    print("\n🧪 Testing configuration loading...")
    
    # Test with default config
    config1 = GenerationConfig()
    print(f"✅ Default config loaded")
    print(f"   Model: {config1.model_name}")
    print(f"   Rollouts: {config1.num_rollouts}")
    
    # Test with TOML file if it exists
    toml_path = Path("gemini_ft_config.toml")
    if toml_path.exists():
        config2 = GenerationConfig(str(toml_path))
        print(f"✅ TOML config loaded from {toml_path}")
        print(f"   Model: {config2.model_name}")
        print(f"   Rollouts: {config2.num_rollouts}")
    
    return True


async def main():
    """Run all tests."""
    print("=" * 60)
    print("🚀 GEMINI FINE-TUNING DATA GENERATION TEST SUITE")
    print("=" * 60)
    
    # Check if service is running
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8901/health")
            if response.status_code != 200:
                print("❌ Synth service not running on port 8901")
                print("   Please start the service with: cd ../../.. && python -m synth_ai.environments.service.app")
                return
    except Exception:
        print("❌ Cannot connect to synth service on port 8901")
        print("   Please start the service with: cd ../../.. && python -m synth_ai.environments.service.app")
        return
    
    print("✅ Synth service is running")
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if test_config_loading():
        tests_passed += 1
    
    if await test_single_trajectory():
        tests_passed += 1
    
    if await test_multiple_trajectories():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 TEST SUMMARY: {tests_passed}/{total_tests} passed")
    print("=" * 60)
    
    if tests_passed == total_tests:
        print("\n✅ All tests passed! Ready to generate fine-tuning data.")
        print("\nNext steps:")
        print("1. Generate full dataset:")
        print("   python generate_ft_data_gemini.py --config gemini_ft_config.toml")
        print("\n2. Validate for Vertex AI:")
        print("   python prepare_vertex_ft.py ft_data_gemini/crafter_gemini_ft.jsonl --validate")
        print("\n3. Start fine-tuning:")
        print("   python kick_off_ft_gemini.py ft_data_gemini/crafter_gemini_ft.jsonl \\")
        print("     --project YOUR_PROJECT --bucket YOUR_BUCKET")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    asyncio.run(main())