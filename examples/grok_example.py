#!/usr/bin/env python3
"""
Example script demonstrating how to use Grok models with synth-ai.

Before running this script, make sure to:
1. Set the XAI_API_KEY environment variable with your xAI API key
2. Install synth-ai: pip install synth-ai

Usage:
    export XAI_API_KEY="your-api-key-here"
    python examples/grok_example.py
"""

import os
from pydantic import BaseModel
from synth_ai.zyk import LM


class MathProblem(BaseModel):
    """Structured response for math problems."""
    problem: str
    solution: int
    explanation: str


def basic_example():
    """Basic example using Grok with synth-ai."""
    print("🚀 Basic Grok Example")
    print("=" * 50)
    
    # Initialize LM with Grok model
    lm = LM(
        model_name="grok-3-mini-beta",
        formatting_model_name="gpt-4o-mini",
        temperature=0.7,
        synth_logging=False,  # Disable synth logging to avoid dependency
    )
    
    # Simple text response
    response = lm.respond_sync(
        system_message="You are a helpful AI assistant.",
        user_message="What is the capital of France?"
    )
    
    print(f"Response: {response.raw_response}")
    print()


def structured_output_example():
    """Example using structured output with Grok."""
    print("🧮 Structured Output Example")
    print("=" * 50)
    
    # Initialize LM with Grok model
    lm = LM(
        model_name="grok-3-mini-beta",
        formatting_model_name="gpt-4o-mini",
        temperature=0,
        synth_logging=False,  # Disable synth logging to avoid dependency
    )
    
    # Structured response
    response = lm.respond_sync(
        system_message="You are a math tutor.",
        user_message="What is 15 + 27? Provide a structured answer.",
        response_model=MathProblem
    )
    
    print(f"Problem: {response.structured_output.problem}")
    print(f"Solution: {response.structured_output.solution}")
    print(f"Explanation: {response.structured_output.explanation}")
    print()


def reasoning_example():
    """Example demonstrating Grok's reasoning capabilities."""
    print("🧠 Reasoning Example")
    print("=" * 50)
    
    # Initialize LM with Grok model
    lm = LM(
        model_name="grok-3-beta",
        formatting_model_name="gpt-4o-mini",
        temperature=0.8,
        synth_logging=False,  # Disable synth logging to avoid dependency
    )
    
    # Complex reasoning task
    response = lm.respond_sync(
        system_message="You are a logical reasoning assistant.",
        user_message="""
        Alice has 3 times as many apples as Bob. 
        Bob has 2 more apples than Charlie.
        If Charlie has 5 apples, how many apples does Alice have?
        Show your step-by-step reasoning.
        """
    )
    
    print(f"Reasoning: {response.raw_response}")
    print()


def main():
    """Run all examples."""
    print("🤖 Grok Integration Examples")
    print("=" * 50)
    print("Available models:")
    print("  • grok-3-beta      - Latest Grok model")
    print("  • grok-3-mini-beta - Smaller, faster Grok model")
    print("  • grok-beta        - Previous generation")
    print()
    
    # Check if API key is set
    if not os.getenv("XAI_API_KEY"):
        print("❌ Error: XAI_API_KEY environment variable not set")
        print("Please set your xAI API key:")
        print("  export XAI_API_KEY='your-api-key-here'")
        return
    
    try:
        # Run examples
        basic_example()
        structured_output_example()
        reasoning_example()
        
        print("✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure your XAI_API_KEY is valid and you have credits available.")


if __name__ == "__main__":
    main() 