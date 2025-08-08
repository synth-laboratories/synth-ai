#!/usr/bin/env python3
"""
Integrated test suite for multiple model providers using synth_ai package.
Tests OpenAI (gpt-4o-mini) and other available models.
"""

import os
import time
import asyncio
from typing import Dict, List, Any
from synth_ai.lm import LM

# Model configurations
MODELS = {
    "openai": {
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "description": "OpenAI GPT-4o-mini"
    },
    "claude": {
        "model": "claude-3-haiku-20240307",
        "api_key_env": "ANTHROPIC_API_KEY", 
        "description": "Claude 3 Haiku"
    },
    "together": {
        "model": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        "api_key_env": "TOGETHER_API_KEY",
        "description": "Llama 3.2 3B via Together"
    }
}

# Test prompts
TEST_SUITE = {
    "basic_math": {
        "system": "You are a helpful math assistant. Give brief answers.",
        "user": "What is 15 + 27?",
        "expected_keywords": ["42"]
    },
    "coding": {
        "system": "You are a Python programming assistant. Be concise.",
        "user": "Write a one-line Python function to reverse a string.",
        "expected_keywords": ["def", "return", "[::-1]"]
    },
    "reasoning": {
        "system": "You are a logical reasoning assistant.",
        "user": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "expected_keywords": ["no", "cannot"]
    },
    "creative": {
        "system": "You are a creative writing assistant.",
        "user": "Write a haiku about artificial intelligence.",
        "expected_keywords": []  # No specific keywords for creative tasks
    }
}


class ModelTester:
    """Test suite for different models."""
    
    def __init__(self):
        self.results = {}
        self.available_models = []
        
    def check_available_models(self):
        """Check which models are available based on API keys."""
        print("Checking available models...")
        
        for name, config in MODELS.items():
            api_key = os.getenv(config["api_key_env"], "")
            if api_key:
                self.available_models.append(name)
                print(f"  ✓ {config['description']}: Available")
            else:
                print(f"  ✗ {config['description']}: No API key ({config['api_key_env']})")
        
        if not self.available_models:
            print("\nNo models available. Please set at least one API key:")
            for name, config in MODELS.items():
                print(f"  export {config['api_key_env']}=your_api_key")
            return False
        
        print(f"\nWill test {len(self.available_models)} model(s)")
        return True
    
    def test_model(self, model_name: str) -> Dict[str, Any]:
        """Test a single model with all test cases."""
        config = MODELS[model_name]
        api_key = os.getenv(config["api_key_env"], "")
        
        print(f"\n{'='*60}")
        print(f"Testing {config['description']}")
        print(f"{'='*60}")
        
        model_results = {
            "model": config["model"],
            "description": config["description"],
            "tests": {},
            "total_time": 0,
            "success_count": 0,
            "failure_count": 0
        }
        
        # Initialize LM
        try:
            lm = LM(
                model=config["model"],
                api_key=api_key,
                temperature=0.7,
                max_tokens=150
            )
        except Exception as e:
            print(f"✗ Failed to initialize model: {e}")
            model_results["failure_count"] = len(TEST_SUITE)
            return model_results
        
        # Run each test
        for test_name, test_config in TEST_SUITE.items():
            print(f"\nTest: {test_name}")
            print(f"  Prompt: {test_config['user'][:50]}...")
            
            start_time = time.time()
            
            try:
                # Make the request
                response = lm.respond(
                    system_message=test_config["system"],
                    user_message=test_config["user"]
                )
                
                elapsed = time.time() - start_time
                
                # Extract the response text
                if hasattr(response, 'raw_response'):
                    response_text = response.raw_response
                else:
                    response_text = str(response)
                
                # Check for expected keywords (if any)
                keywords_found = []
                for keyword in test_config["expected_keywords"]:
                    if keyword.lower() in response_text.lower():
                        keywords_found.append(keyword)
                
                success = len(keywords_found) == len(test_config["expected_keywords"]) if test_config["expected_keywords"] else True
                
                model_results["tests"][test_name] = {
                    "success": success,
                    "response": response_text[:200],
                    "time": elapsed,
                    "keywords_found": keywords_found
                }
                
                if success:
                    model_results["success_count"] += 1
                    print(f"  ✓ Success ({elapsed:.2f}s)")
                else:
                    model_results["failure_count"] += 1
                    print(f"  ✗ Failed - missing keywords: {set(test_config['expected_keywords']) - set(keywords_found)}")
                
                print(f"  Response: {response_text[:100]}...")
                
            except Exception as e:
                model_results["tests"][test_name] = {
                    "success": False,
                    "error": str(e),
                    "time": time.time() - start_time
                }
                model_results["failure_count"] += 1
                print(f"  ✗ Error: {e}")
            
            model_results["total_time"] += time.time() - start_time
        
        return model_results
    
    def run_tests(self):
        """Run tests on all available models."""
        if not self.check_available_models():
            return
        
        print(f"\nStarting tests on {len(self.available_models)} model(s)...")
        
        for model_name in self.available_models:
            result = self.test_model(model_name)
            self.results[model_name] = result
        
        self.print_summary()
    
    def print_summary(self):
        """Print test summary."""
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        
        # Overall statistics
        total_tests = len(TEST_SUITE) * len(self.results)
        total_success = sum(r["success_count"] for r in self.results.values())
        total_failed = sum(r["failure_count"] for r in self.results.values())
        
        print(f"\nOverall Results:")
        print(f"  Total tests run: {total_tests}")
        print(f"  Successful: {total_success}")
        print(f"  Failed: {total_failed}")
        print(f"  Success rate: {(total_success/total_tests*100):.1f}%")
        
        # Per-model summary
        print(f"\nPer-Model Performance:")
        for model_name, result in self.results.items():
            print(f"\n  {result['description']}:")
            print(f"    Success: {result['success_count']}/{len(TEST_SUITE)}")
            print(f"    Total time: {result['total_time']:.2f}s")
            print(f"    Avg time per test: {result['total_time']/len(TEST_SUITE):.2f}s")
            
            # Show failed tests
            failed_tests = [name for name, test in result["tests"].items() if not test["success"]]
            if failed_tests:
                print(f"    Failed tests: {', '.join(failed_tests)}")
        
        # Test-specific summary
        print(f"\nTest Case Performance:")
        for test_name in TEST_SUITE.keys():
            successes = sum(1 for r in self.results.values() if r["tests"].get(test_name, {}).get("success", False))
            print(f"  {test_name}: {successes}/{len(self.results)} models passed")
        
        # Response time analysis
        print(f"\nResponse Time Analysis:")
        all_times = []
        for result in self.results.values():
            for test in result["tests"].values():
                if "time" in test:
                    all_times.append(test["time"])
        
        if all_times:
            print(f"  Fastest: {min(all_times):.2f}s")
            print(f"  Slowest: {max(all_times):.2f}s")
            print(f"  Average: {sum(all_times)/len(all_times):.2f}s")
        
        print(f"\n{'='*60}")
        if total_failed == 0:
            print("✓ ALL TESTS PASSED!")
        else:
            print(f"✗ {total_failed} test(s) failed")
        print(f"{'='*60}")


def main():
    """Main test runner."""
    print("Integrated Model Testing Suite")
    print("Testing multiple LLM providers via synth_ai package")
    
    tester = ModelTester()
    tester.run_tests()


if __name__ == "__main__":
    main()