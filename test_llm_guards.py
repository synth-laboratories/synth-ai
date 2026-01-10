"""Test script to verify LLM call guards work correctly."""

import sys
import warnings


def test_httpx_guard():
    """Test that httpx guard detects direct provider calls."""
    print("\n=== Testing httpx guard ===")

    import asyncio

    import httpx
    from synth_ai.sdk.task import install_httpx_guard

    install_httpx_guard()

    async def test_direct_call():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            async with httpx.AsyncClient() as client:
                try:
                    # This should trigger a warning (but will fail with connection error)
                    await client.post("https://api.openai.com/v1/chat/completions", json={})
                except Exception:
                    pass  # Expected to fail, we just want the warning

            if w:
                print("✓ Warning triggered for direct OpenAI call:")
                print(f"  {w[0].message}")
                return True
            else:
                print("✗ No warning raised for direct OpenAI call")
                return False

    result = asyncio.run(test_direct_call())

    async def test_interceptor_call():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            async with httpx.AsyncClient() as client:
                try:
                    # This should NOT trigger a warning
                    await client.post(
                        "http://localhost:8000/api/interceptor/v1/test/chat/completions", json={}
                    )
                except Exception:
                    pass  # Expected to fail, we just want to check for warnings

            if w:
                print("✗ Unexpected warning for interceptor call:")
                print(f"  {w[0].message}")
                return False
            else:
                print("✓ No warning for interceptor call (correct)")
                return True

    result2 = asyncio.run(test_interceptor_call())
    return result and result2


def test_openai_guard():
    """Test that OpenAI client guard detects direct instantiation."""
    print("\n=== Testing OpenAI client guard ===")

    try:
        import openai
        from synth_ai.sdk.task import install_openai_guard

        install_openai_guard()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # This should trigger a warning
            client = openai.OpenAI(api_key="test")

            if w:
                print("✓ Warning triggered for direct OpenAI client:")
                print(f"  {w[0].message}")
                return True
            else:
                print("✗ No warning raised for direct OpenAI client")
                return False

    except ImportError:
        print("⊘ OpenAI not installed, skipping test")
        return True


def test_anthropic_guard():
    """Test that Anthropic client guard detects direct instantiation."""
    print("\n=== Testing Anthropic client guard ===")

    try:
        import anthropic
        from synth_ai.sdk.task import install_anthropic_guard

        install_anthropic_guard()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # This should trigger a warning
            client = anthropic.Anthropic(api_key="test")

            if w:
                print("✓ Warning triggered for direct Anthropic client:")
                print(f"  {w[0].message}")
                return True
            else:
                print("✗ No warning raised for direct Anthropic client")
                return False

    except ImportError:
        print("⊘ Anthropic not installed, skipping test")
        return True


def test_url_checker():
    """Test the URL checking function."""
    print("\n=== Testing URL checker ===")

    from synth_ai.sdk.task.llm_call_guards import check_url_for_direct_provider_call

    test_cases = [
        ("https://api.openai.com/v1/chat/completions", True, "OpenAI"),
        ("https://api.anthropic.com/v1/messages", True, "Anthropic"),
        ("https://api.groq.com/openai/v1/chat/completions", True, "Groq"),
        ("http://localhost:8000/api/interceptor/v1/test/chat/completions", False, "Interceptor"),
        ("https://api.usesynth.ai/v1/inference", False, "Synth API"),
    ]

    all_passed = True
    for url, should_warn, name in test_cases:
        result = check_url_for_direct_provider_call(url)
        if result == should_warn:
            print(f"✓ {name}: {'Detected' if result else 'Ignored'} correctly")
        else:
            print(f"✗ {name}: Expected {should_warn}, got {result}")
            all_passed = False

    return all_passed


def test_install_all_guards():
    """Test that install_all_guards doesn't crash."""
    print("\n=== Testing install_all_guards ===")

    try:
        from synth_ai.sdk.task import install_all_guards

        # Should be safe to call multiple times
        install_all_guards()
        install_all_guards()

        print("✓ install_all_guards() executed successfully")
        return True
    except Exception as e:
        print(f"✗ install_all_guards() failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing LLM Call Guards")
    print("=" * 50)

    results = []

    results.append(("URL Checker", test_url_checker()))
    results.append(("httpx Guard", test_httpx_guard()))
    results.append(("OpenAI Guard", test_openai_guard()))
    results.append(("Anthropic Guard", test_anthropic_guard()))
    results.append(("install_all_guards", test_install_all_guards()))

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {name}: {status}")

    all_passed = all(passed for _, passed in results)

    print("\n" + ("=" * 50))
    if all_passed:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)
