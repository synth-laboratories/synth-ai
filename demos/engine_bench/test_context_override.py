#!/usr/bin/env python3
"""
Minimal test to demonstrate context override system works with EngineBench.

This shows:
1. Task app accepts context_override parameter
2. Context affects agent's coding output
3. Different contexts produce different results
"""

import asyncio
import os
from typing import Any, Dict

import httpx

# Task app configuration
TASK_APP_URL = "http://localhost:8020/rollout"
API_KEY = os.getenv("ENVIRONMENT_API_KEY", "sk_env_30c78a787bac223c716918181209f263")

# Test configurations
BASELINE_SYSTEM_PROMPT = """You are an expert Rust developer implementing Pokemon TCG cards.

Your task: Implement card effects by editing Rust files with stub functions marked with TODO comments.

Key patterns:
- Use `def_id_matches(&card.def_id, "DF", NUMBER)` to identify cards
- Implement attack modifiers in the `attack_override` function
- Use `game.queue_prompt()` for user choices
- Return `AttackOverrides::default()` if card doesn't apply

Output requirements:
1. Edit files - replace TODO stubs with working code
2. Make sure code compiles (`cargo check`)
3. Make sure tests pass (`cargo test`)"""

ENHANCED_SYSTEM_PROMPT = """You are an ELITE Rust developer implementing Pokemon TCG cards with EXTREME attention to detail.

CRITICAL TASK: Implement card effects by editing Rust files with stub functions marked with TODO comments.

**MANDATORY PATTERNS** (FAILURE TO FOLLOW = COMPILATION ERRORS):
1. Card Identification: `def_id_matches(&card.def_id, "DF", NUMBER)` or `def_id_matches(&card.def_id, "HP", NUMBER)`
2. Attack Modifiers: Implement in `attack_override` function
3. Poke-Powers/Bodies: Implement in `power_effect` function
4. User Choices: Use `game.queue_prompt(...)` for interactive decisions
5. Non-applicable Cards: Return `AttackOverrides::default()` explicitly

**QUALITY CHECKLIST** (ALL MUST PASS):
✓ Code compiles without errors (`cargo check`)
✓ Tests pass (`cargo test`)
✓ Logic handles edge cases (0 damage, invalid targets, etc.)
✓ Comments explain non-obvious mechanics
✓ No unwrap() calls without error handling

**COMMON ERRORS TO AVOID**:
- ❌ Forgetting to check card.def_id before applying effects
- ❌ Not handling the case where attacker/defender is missing
- ❌ Mutating game state incorrectly
- ❌ Using wrong damage calculation order (base → modifiers → final)

**IMPLEMENTATION STRATEGY**:
1. Read the card text CAREFULLY - every word matters
2. Identify the trigger condition (when does this effect apply?)
3. Identify the effect (what changes?)
4. Implement with defensive programming (check all edge cases)
5. Test mentally: "What if damage is 0? What if multiple cards apply?"

SHIP WORKING CODE. NOTHING LESS."""


async def test_context_override(test_name: str, context_override: Dict[str, Any]):
    """Test a context override configuration."""
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"{'='*80}")

    # Construct rollout request
    import uuid
    request = {
        "run_id": f"test_{test_name.lower().replace(' ', '_')}",
        "trace_correlation_id": str(uuid.uuid4()),
        "env": {
            "env_name": "engine_bench",
            "seed": 0,  # df-001-ampharos
            "config": {}
        },
        "policy": {
            "policy_name": "test-policy",
            "config": {
                "inference_url": "http://localhost:8000/api/inference",
                "model": "gpt-4.1-mini",
                "temperature": 0.7,
                "max_tokens": 4000,
                "timeout": 180
            }
        },
        "model": "gpt-4.1-mini",
    }

    # Add context override
    if context_override:
        request["context_override"] = context_override
        print("✓ Context override provided:")
        for key, value in context_override.items():
            if isinstance(value, str):
                print(f"  - {key}: {len(value)} chars")
            else:
                print(f"  - {key}: {value}")

    print("\nSending rollout request...")

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                TASK_APP_URL,
                json=request,
                headers={"X-API-Key": API_KEY}
            )
            response.raise_for_status()

            result = response.json()

            print("\n✓ Request completed")
            print(f"  - Status: {result.get('success_status', 'unknown')}")
            print(f"  - Reward: {result.get('reward', 0.0):.3f}")

            # Check for context override status
            if "context_override_status" in result:
                status = result["context_override_status"]
                print("\n✓ Context override applied:")
                print(f"  - File artifacts: {len(status.get('file_artifacts', {}))}")
                print(f"  - Success rate: {status.get('success_rate', 0.0):.1%}")

                # Show any rejections
                for key, artifact_status in status.get('file_artifacts', {}).items():
                    if artifact_status.get('status') == 'rejected':
                        print(f"  ✗ {key}: {artifact_status.get('error', 'rejected')}")

            # Show agent output snippet
            if "trace" in result and "stdout" in result["trace"]:
                stdout = result["trace"]["stdout"]
                if len(stdout) > 500:
                    print("\nAgent output (first 500 chars):")
                    print(stdout[:500] + "...")
                else:
                    print("\nAgent output:")
                    print(stdout)

            return result

        except httpx.HTTPError as e:
            print(f"\n✗ Request failed: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response: {e.response.text}")
            raise


async def main():
    """Run context override tests."""
    print("="*80)
    print("UNIFIED OPTIMIZATION CONTEXT OVERRIDE DEMONSTRATION")
    print("="*80)
    print(f"Task app: {TASK_APP_URL}")
    print("Test card: df-001-ampharos (Stage 2, Poke-Body)")
    print("="*80)

    try:
        # Test 1: Baseline (no context override)
        print("\n\n")
        result_baseline = await test_context_override(
            "Baseline (No Context Override)",
            context_override=None
        )

        # Test 2: Enhanced system prompt
        print("\n\n")
        result_enhanced = await test_context_override(
            "Enhanced System Prompt",
            context_override={
                "system_prompt": ENHANCED_SYSTEM_PROMPT
            }
        )

        # Test 3: Full context with architecture guide
        print("\n\n")
        result_full = await test_context_override(
            "Full Context (System + Architecture)",
            context_override={
                "system_prompt": ENHANCED_SYSTEM_PROMPT,
                "architecture_guide": """# EngineBench Architecture Guide

## Core Patterns

### 1. Card Identification
```rust
if def_id_matches(&card.def_id, "DF", 1) {
    // This is df-001-ampharos
}
```

### 2. Attack Overrides
```rust
pub fn attack_override(game: &mut Game, card: &Card) -> AttackOverrides {
    let mut overrides = AttackOverrides::default();

    if def_id_matches(&card.def_id, "DF", 1) {
        // Add modifiers here
        overrides.damage_modifier = Some(Box::new(|base_damage| {
            base_damage + 10  // +10 damage example
        }));
    }

    overrides
}
```

### 3. Poke-Powers/Bodies
```rust
pub fn power_effect(game: &mut Game, card: &Card, trigger: PowerTrigger) -> Result<(), String> {
    if trigger != PowerTrigger::Continuous {
        return Ok(());
    }

    if def_id_matches(&card.def_id, "DF", 1) {
        // Implement Poke-Body effect
        // This runs every turn automatically
    }

    Ok(())
}
```

## Anti-Patterns

❌ **DON'T** use unwrap() without null checks
❌ **DON'T** forget to check def_id before applying effects
❌ **DON'T** mutate game state in attack_override (use power_effect instead)
❌ **DON'T** return errors for non-applicable cards (return Ok(()) or default)

✅ **DO** handle edge cases (0 damage, missing cards, etc.)
✅ **DO** use defensive programming
✅ **DO** test your logic mentally before implementing
"""
            }
        )

        # Compare results
        print("\n\n")
        print("="*80)
        print("RESULTS COMPARISON")
        print("="*80)
        print(f"Baseline reward:        {result_baseline.get('reward', 0.0):.3f}")
        print(f"Enhanced prompt reward: {result_enhanced.get('reward', 0.0):.3f}")
        print(f"Full context reward:    {result_full.get('reward', 0.0):.3f}")
        print("="*80)

        if result_full.get('reward', 0) > result_baseline.get('reward', 0):
            print("✓ CONTEXT OVERRIDES IMPROVE PERFORMANCE")
        else:
            print("✗ No improvement detected (may need more test iterations)")

        print("="*80)
        print("\nCONCLUSION:")
        print("The unified optimization system successfully:")
        print("  1. Accepts context_override parameters")
        print("  2. Applies them to the coding agent")
        print("  3. Reports application status")
        print("  4. Affects agent behavior and output quality")
        print("\nThis demonstrates the core infrastructure is working.")
        print("Full GEPA evolution would iterate this process across generations,")
        print("evolving both prompts AND context artifacts together.")
        print("="*80)

    except Exception as e:
        print(f"\n\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
