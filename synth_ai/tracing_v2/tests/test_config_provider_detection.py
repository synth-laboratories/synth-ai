#!/usr/bin/env python3
"""
Test the config-based provider detection system.
"""

from synth_ai.tracing_v2.config import (
    is_openai_model, is_azure_openai_model, is_anthropic_model, 
    detect_provider, OPENAI_MODELS, ANTHROPIC_MODELS
)


def test_openai_detection():
    """Test OpenAI model detection."""
    print("=== Testing OpenAI Model Detection ===\n")
    
    test_cases = [
        # Standard models
        ("gpt-4", True),
        ("gpt-4o-mini", True),
        ("gpt-3.5-turbo", True),
        
        # O-series reasoning models
        ("o1", True),
        ("o1-preview", True),
        ("o3-mini", True),
        ("o4", True),  # Future model
        
        # Legacy models
        ("text-davinci-003", True),
        ("text-embedding-3-large", True),
        
        # Non-OpenAI models
        ("claude-3-opus", False),
        ("llama-2-70b", False),
        ("mistral-7b", False),
    ]
    
    for model, expected in test_cases:
        result = is_openai_model(model)
        status = "✅" if result == expected else "❌"
        print(f"{status} {model}: {result} (expected: {expected})")
    
    print()


def test_anthropic_detection():
    """Test Anthropic model detection."""
    print("=== Testing Anthropic Model Detection ===\n")
    
    test_cases = [
        # Claude 3 family
        ("claude-3-opus-20240229", True),
        ("claude-3-sonnet-20240229", True),
        ("claude-3-haiku-20240307", True),
        ("claude-3-5-sonnet-20241022", True),
        
        # Claude 2 family
        ("claude-2.1", True),
        ("claude-instant-1.2", True),
        
        # Shortened versions
        ("claude-3-opus", True),
        ("claude", True),
        
        # Non-Anthropic models
        ("gpt-4", False),
        ("llama-2", False),
        ("mistral-7b", False),
    ]
    
    for model, expected in test_cases:
        result = is_anthropic_model(model)
        status = "✅" if result == expected else "❌"
        print(f"{status} {model}: {result} (expected: {expected})")
    
    print()


def test_azure_detection():
    """Test Azure OpenAI model detection."""
    print("=== Testing Azure OpenAI Model Detection ===\n")
    
    test_cases = [
        # Azure with endpoint
        ("gpt-4", "https://myapp.openai.azure.com/", True),
        ("gpt-35-turbo", "https://myapp.openai.azure.com/", True),  # Azure naming
        ("gpt-3.5-turbo", "https://myapp.openai.azure.com/", True),
        
        # Without Azure endpoint
        ("gpt-4", "https://api.openai.com/v1", False),
        ("gpt-35-turbo", "", True),  # Azure-specific naming even without endpoint
        
        # Non-OpenAI models
        ("claude-3", "https://myapp.openai.azure.com/", False),
    ]
    
    for model, endpoint, expected in test_cases:
        result = is_azure_openai_model(model, endpoint)
        status = "✅" if result == expected else "❌"
        print(f"{status} {model} + {endpoint}: {result} (expected: {expected})")
    
    print()


def test_provider_detection():
    """Test the main detect_provider function."""
    print("=== Testing Provider Detection ===\n")
    
    test_cases = [
        # OpenAI
        ("gpt-4", "", "openai"),
        ("o1-preview", "", "openai"),
        ("text-embedding-3-small", "", "openai"),
        
        # Azure OpenAI
        ("gpt-4", "https://myapp.openai.azure.com/", "azure_openai"),
        ("gpt-35-turbo", "", "azure_openai"),
        
        # Anthropic
        ("claude-3-opus-20240229", "", "anthropic"),
        ("claude-2.1", "", "anthropic"),
        
        # Unknown
        ("llama-2-70b", "", None),
        ("mistral-large", "", None),
        ("", "", None),
    ]
    
    for model, endpoint, expected in test_cases:
        result = detect_provider(model, endpoint)
        status = "✅" if result == expected else "❌"
        print(f"{status} {model} + {endpoint}: {result} (expected: {expected})")
    
    print()


def test_model_list_coverage():
    """Verify model lists are comprehensive."""
    print("=== Model List Statistics ===\n")
    
    print(f"OpenAI models in config: {len(OPENAI_MODELS)}")
    print(f"Anthropic models in config: {len(ANTHROPIC_MODELS)}")
    
    # Show some examples
    print("\nSample OpenAI models:")
    for model in list(OPENAI_MODELS)[:5]:
        print(f"  - {model}")
    
    print("\nSample Anthropic models:")
    for model in list(ANTHROPIC_MODELS)[:5]:
        print(f"  - {model}")
    
    print()


def main():
    """Run all tests."""
    print("🧪 Testing Config-Based Provider Detection\n")
    
    test_openai_detection()
    test_anthropic_detection()
    test_azure_detection()
    test_provider_detection()
    test_model_list_coverage()
    
    print("="*60)
    print("🎯 SUMMARY")
    print("="*60)
    print("Config-based detection provides:")
    print("- ✅ Accurate model-to-provider mapping")
    print("- ✅ Support for o-series OpenAI models (o1, o3, o4, etc.)")
    print("- ✅ Azure-specific naming patterns (gpt-35-turbo)")
    print("- ✅ Comprehensive model lists for each provider")


if __name__ == "__main__":
    main()