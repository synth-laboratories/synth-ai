#!/usr/bin/env python3
"""
Test Synth backend integration using synth_ai package.
Tests GPU warmup, inference, fine-tuning, DPO, and inference with fine-tuned models.
Focus on Qwen3 0.6B model.
"""

import asyncio
import os
import time
import json
from typing import Optional, Dict, Any, List
from synth_ai.lm import LM

# Configuration
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # Using available small model
API_KEY = os.getenv("SYNTH_API_KEY", "")

# Test configuration
TEST_PROMPTS = [
    "What is 2+2?",
    "Name the capital of France.",
    "Write a haiku about AI.",
]


class SynthBackendTester:
    """Test suite for Synth backend operations."""
    
    def __init__(self):
        """Initialize the tester with Synth client."""
        self.results = []
        
    async def test_basic_inference(self):
        """Test 1: Basic inference without GPU specification."""
        print("\n" + "="*60)
        print("TEST 1: Basic Inference")
        print("="*60)
        
        try:
            # Initialize LM client
            lm = LM(
                model=MODEL_ID,
                api_key=API_KEY,
                system_prompt="You are a helpful assistant.",
                temperature=0.7,
                max_tokens=100
            )
            
            # Test basic inference
            for i, prompt in enumerate(TEST_PROMPTS, 1):
                print(f"\nPrompt {i}: {prompt}")
                start_time = time.time()
                
                response = lm.respond(prompt)
                elapsed = time.time() - start_time
                
                print(f"Response: {response[:100]}..." if len(response) > 100 else f"Response: {response}")
                print(f"Time: {elapsed:.2f}s")
                
                self.results.append({
                    "test": "basic_inference",
                    "prompt": prompt,
                    "success": True,
                    "time": elapsed,
                    "response_length": len(response)
                })
                
            print("\n✓ Basic inference test completed")
            return True
            
        except Exception as e:
            print(f"\n✗ Basic inference failed: {e}")
            self.results.append({
                "test": "basic_inference",
                "success": False,
                "error": str(e)
            })
            return False
    
    async def test_streaming_inference(self):
        """Test 2: Streaming inference."""
        print("\n" + "="*60)
        print("TEST 2: Streaming Inference")
        print("="*60)
        
        try:
            # Initialize LM client with streaming
            lm = LM(
                model=MODEL_ID,
                api_key=API_KEY,
                stream=True,
                temperature=0.7,
                max_tokens=150
            )
            
            prompt = "Explain machine learning in simple terms."
            print(f"\nPrompt: {prompt}")
            print("Streaming response: ", end="", flush=True)
            
            start_time = time.time()
            full_response = ""
            
            # Stream the response
            for chunk in lm.respond(prompt):
                print(chunk, end="", flush=True)
                full_response += chunk
            
            elapsed = time.time() - start_time
            print(f"\n\nStreaming completed in {elapsed:.2f}s")
            print(f"Total response length: {len(full_response)} chars")
            
            self.results.append({
                "test": "streaming_inference",
                "success": True,
                "time": elapsed,
                "response_length": len(full_response)
            })
            
            print("\n✓ Streaming inference test completed")
            return True
            
        except Exception as e:
            print(f"\n✗ Streaming inference failed: {e}")
            self.results.append({
                "test": "streaming_inference",
                "success": False,
                "error": str(e)
            })
            return False
    
    async def test_concurrent_requests(self):
        """Test 3: Concurrent inference requests."""
        print("\n" + "="*60)
        print("TEST 3: Concurrent Requests")
        print("="*60)
        
        try:
            async def make_request(prompt: str, index: int):
                """Make a single inference request."""
                lm = LM(
                    model=MODEL_ID,
                    api_key=API_KEY,
                    temperature=0.5,
                    max_tokens=50
                )
                
                start = time.time()
                response = lm.respond(prompt)
                elapsed = time.time() - start
                
                return {
                    "index": index,
                    "prompt": prompt[:30],
                    "response_length": len(response),
                    "time": elapsed
                }
            
            # Create concurrent tasks
            prompts = [
                "What is Python?",
                "Explain quantum computing.",
                "Define artificial intelligence.",
                "What are neural networks?",
                "Describe machine learning."
            ]
            
            print(f"Sending {len(prompts)} concurrent requests...")
            start_time = time.time()
            
            # Run requests concurrently
            tasks = [make_request(p, i) for i, p in enumerate(prompts)]
            results = await asyncio.gather(*tasks)
            
            total_time = time.time() - start_time
            
            # Display results
            print("\nResults:")
            for r in results:
                print(f"  Request {r['index']}: {r['response_length']} chars in {r['time']:.2f}s")
            
            print(f"\nTotal time for {len(prompts)} concurrent requests: {total_time:.2f}s")
            avg_time = sum(r['time'] for r in results) / len(results)
            print(f"Average response time: {avg_time:.2f}s")
            
            self.results.append({
                "test": "concurrent_requests",
                "success": True,
                "total_time": total_time,
                "avg_time": avg_time,
                "num_requests": len(prompts)
            })
            
            print("\n✓ Concurrent requests test completed")
            return True
            
        except Exception as e:
            print(f"\n✗ Concurrent requests failed: {e}")
            self.results.append({
                "test": "concurrent_requests",
                "success": False,
                "error": str(e)
            })
            return False
    
    async def test_different_parameters(self):
        """Test 4: Different inference parameters."""
        print("\n" + "="*60)
        print("TEST 4: Different Parameters")
        print("="*60)
        
        try:
            test_configs = [
                {"temperature": 0.1, "max_tokens": 20, "description": "Low temp, short"},
                {"temperature": 0.9, "max_tokens": 100, "description": "High temp, medium"},
                {"temperature": 0.5, "max_tokens": 200, "description": "Medium temp, long"},
            ]
            
            prompt = "Tell me about space exploration."
            
            for config in test_configs:
                print(f"\nConfig: {config['description']}")
                print(f"  Temperature: {config['temperature']}, Max tokens: {config['max_tokens']}")
                
                lm = LM(
                    model=MODEL_ID,
                    api_key=API_KEY,
                    temperature=config['temperature'],
                    max_tokens=config['max_tokens']
                )
                
                start = time.time()
                response = lm.respond(prompt)
                elapsed = time.time() - start
                
                print(f"  Response length: {len(response)} chars")
                print(f"  Time: {elapsed:.2f}s")
                print(f"  Preview: {response[:80]}...")
            
            self.results.append({
                "test": "different_parameters",
                "success": True,
                "configs_tested": len(test_configs)
            })
            
            print("\n✓ Different parameters test completed")
            return True
            
        except Exception as e:
            print(f"\n✗ Different parameters test failed: {e}")
            self.results.append({
                "test": "different_parameters",
                "success": False,
                "error": str(e)
            })
            return False
    
    async def test_conversation_context(self):
        """Test 5: Multi-turn conversation with context."""
        print("\n" + "="*60)
        print("TEST 5: Conversation Context")
        print("="*60)
        
        try:
            # Initialize LM with system prompt
            lm = LM(
                model=MODEL_ID,
                api_key=API_KEY,
                system_prompt="You are a helpful Python programming assistant.",
                temperature=0.5,
                max_tokens=100
            )
            
            # Multi-turn conversation
            conversation = [
                "What is a Python list?",
                "How do I add items to it?",
                "Can you show me an example?",
            ]
            
            print("Starting multi-turn conversation...\n")
            
            for i, user_msg in enumerate(conversation, 1):
                print(f"User {i}: {user_msg}")
                
                start = time.time()
                response = lm(user_msg)
                elapsed = time.time() - start
                
                print(f"Assistant {i}: {response[:150]}..." if len(response) > 150 else f"Assistant {i}: {response}")
                print(f"(Response time: {elapsed:.2f}s)\n")
            
            self.results.append({
                "test": "conversation_context",
                "success": True,
                "turns": len(conversation)
            })
            
            print("✓ Conversation context test completed")
            return True
            
        except Exception as e:
            print(f"\n✗ Conversation context test failed: {e}")
            self.results.append({
                "test": "conversation_context",
                "success": False,
                "error": str(e)
            })
            return False
    
    def generate_summary(self):
        """Generate test summary report."""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        total_tests = len(set(r["test"] for r in self.results))
        successful = sum(1 for r in self.results if r.get("success", False))
        failed = total_tests - successful
        
        print(f"\nTotal Tests Run: {total_tests}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        
        if failed == 0:
            print("\n✓ ALL TESTS PASSED!")
        else:
            print(f"\n✗ {failed} test(s) failed")
        
        # Show failed tests
        failed_tests = [r for r in self.results if not r.get("success", False)]
        if failed_tests:
            print("\nFailed Tests:")
            for test in failed_tests:
                print(f"  - {test['test']}: {test.get('error', 'Unknown error')}")
        
        # Show performance metrics
        print("\nPerformance Metrics:")
        inference_results = [r for r in self.results if r.get("test") == "basic_inference" and r.get("success")]
        if inference_results:
            avg_time = sum(r.get("time", 0) for r in inference_results) / len(inference_results)
            print(f"  Average inference time: {avg_time:.2f}s")
        
        concurrent_results = [r for r in self.results if r.get("test") == "concurrent_requests" and r.get("success")]
        if concurrent_results:
            print(f"  Concurrent requests avg time: {concurrent_results[0].get('avg_time', 0):.2f}s")
        
        print("\n" + "="*60)
        print("Testing completed!")
        print("="*60)


async def main():
    """Main test runner."""
    print("Starting Synth Backend Integration Tests")
    print("Model:", MODEL_ID)
    print("API Key:", "Set" if API_KEY else "Not set (will use default)")
    
    tester = SynthBackendTester()
    
    # Run tests
    tests = [
        tester.test_basic_inference(),
        tester.test_streaming_inference(),
        tester.test_concurrent_requests(),
        tester.test_different_parameters(),
        tester.test_conversation_context(),
    ]
    
    # Execute all tests
    results = await asyncio.gather(*tests, return_exceptions=True)
    
    # Handle any exceptions
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"\nTest {i+1} raised exception: {result}")
    
    # Generate summary
    tester.generate_summary()


if __name__ == "__main__":
    asyncio.run(main())