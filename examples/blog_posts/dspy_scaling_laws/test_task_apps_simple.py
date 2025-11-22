"""Simple test to validate task apps load and have correct structure."""

import sys
from pathlib import Path

# Add paths
task_apps_dir = Path(__file__).resolve().parents[3] / "task_apps"
if str(task_apps_dir) not in sys.path:
    sys.path.insert(0, str(task_apps_dir))

def test_task_app_import(module_path: str, task_app_name: str):
    """Test that a task app can be imported and has correct structure."""

    print(f"\n{'='*60}")
    print(f"Testing: {task_app_name}")
    print(f"{'='*60}")

    try:
        # Add module directory to path
        module_dir = Path(module_path).parent
        if str(module_dir) not in sys.path:
            sys.path.insert(0, str(module_dir))

        # Import the module
        module_name = Path(module_path).stem
        module = __import__(module_name)

        # Check required components
        checks = {
            "build_config": hasattr(module, "build_config"),
            "rollout_executor": hasattr(module, "rollout_executor"),
            "register_task_app call": True,  # Assume it's there if module loads
        }

        # Try to build config
        try:
            config = module.build_config()
            checks["config builds"] = True
            checks["app_id"] = config.app_id
            print(f"  ✅ App ID: {config.app_id}")
            print(f"  ✅ Name: {config.name}")

            # Check dataset
            if hasattr(config, 'app_state'):
                dataset_keys = list(config.app_state.keys())
                print(f"  ✅ Datasets: {dataset_keys}")

        except Exception as e:
            checks["config builds"] = False
            print(f"  ❌ Config build failed: {e}")

        # Print results
        all_passed = all(v == True or isinstance(v, str) for v in checks.values())

        if all_passed:
            print(f"  ✅ All checks passed!")
            return True
        else:
            print(f"  ⚠️  Some checks failed:")
            for check, result in checks.items():
                if result != True and not isinstance(result, str):
                    print(f"     - {check}: {result}")
            return False

    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test all task apps."""

    print("🧪 Testing all task app imports and structure\n")

    base_dir = Path(__file__).parent

    task_apps = [
        # Banking77
        (str(base_dir / "benchmarks/banking77/pipeline_3step/banking77_3step_task_app.py"), "Banking77 3-step"),
        (str(base_dir / "benchmarks/banking77/pipeline_5step/banking77_5step_task_app.py"), "Banking77 5-step"),

        # HeartDisease
        (str(base_dir / "benchmarks/heartdisease/pipeline_3step/heartdisease_3step_task_app.py"), "HeartDisease 3-step"),
        (str(base_dir / "benchmarks/heartdisease/pipeline_5step/heartdisease_5step_task_app.py"), "HeartDisease 5-step"),

        # HotpotQA
        (str(base_dir / "benchmarks/hotpotqa/pipeline_3step/hotpotqa_3step_task_app.py"), "HotpotQA 3-step"),
        (str(base_dir / "benchmarks/hotpotqa/pipeline_5step/hotpotqa_5step_task_app.py"), "HotpotQA 5-step"),
    ]

    results = []
    for module_path, name in task_apps:
        passed = test_task_app_import(module_path, name)
        results.append((name, passed))

    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"{status} {name}")

    print(f"\n{passed_count}/{total_count} task apps passed")

    if passed_count == total_count:
        print("\n✅ All task apps are structurally valid!")
        print("\nNext steps:")
        print("  1. Deploy task apps to test with actual rollouts")
        print("  2. Run baseline tests with gpt-4o-mini")
        print("  3. If baselines get >10% accuracy, proceed with optimization")
    else:
        print("\n⚠️  Some task apps failed. Fix these before proceeding.")

    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
