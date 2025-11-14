#!/usr/bin/env python3
"""Build and publish synth-ai dev version."""
import subprocess
import sys
from pathlib import Path

def run_cmd(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)

def main():
    synth_ai_root = Path(__file__).parent
    monorepo_root = synth_ai_root.parent / "monorepo"
    
    print("=" * 60)
    print("Building and Publishing synth-ai 0.2.23.dev4")
    print("=" * 60)
    
    # Step 1: Build
    print("\n[1/4] Building synth-ai...")
    exit_code, stdout, stderr = run_cmd(
        ["python3", "-m", "build"],
        cwd=synth_ai_root,
    )
    if exit_code != 0:
        print(f"❌ Build failed:\n{stderr}")
        return 1
    print("✓ Build successful")
    if stdout:
        print(stdout[-500:])  # Last 500 chars
    
    # Step 2: Find wheel
    print("\n[2/4] Finding wheel...")
    dist_dir = synth_ai_root / "dist"
    wheels = list(dist_dir.glob("synth_ai-0.2.23.dev4-*.whl"))
    if not wheels:
        print(f"❌ Wheel not found in {dist_dir}")
        return 1
    wheel_path = wheels[0]
    print(f"✓ Found: {wheel_path.name}")
    
    # Step 3: Load credentials and publish
    print("\n[3/4] Publishing to PyPI...")
    env_file = monorepo_root / ".env.dev"
    if not env_file.exists():
        print(f"❌ .env.dev not found at {env_file}")
        return 1
    
    # Load environment variables from .env.dev
    env_vars = {}
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env_vars[key.strip()] = value.strip().strip('"').strip("'")
    
    # Set environment variables
    import os
    for key, value in env_vars.items():
        if key.startswith("TWINE") or key.startswith("PYPI"):
            os.environ[key] = value
    
    # Publish
    exit_code, stdout, stderr = run_cmd(
        ["python3", "-m", "twine", "upload", str(wheel_path), "--repository", "pypi"],
        cwd=synth_ai_root,
    )
    if exit_code != 0:
        print(f"❌ Publish failed:\n{stderr}")
        if "TWINE" in stderr or "credentials" in stderr.lower():
            print("\n⚠️  Make sure TWINE_USERNAME and TWINE_PASSWORD are set in .env.dev")
        return 1
    print("✓ Published successfully")
    if stdout:
        print(stdout)
    
    # Step 4: Install in backend
    print("\n[4/4] Installing in backend...")
    backend_dir = monorepo_root / "backend"
    
    # Try uv first, then pip
    exit_code, stdout, stderr = run_cmd(
        ["uv", "pip", "install", "--upgrade", "synth-ai>=0.2.23.dev4"],
        cwd=backend_dir,
    )
    if exit_code != 0:
        exit_code, stdout, stderr = run_cmd(
            ["pip", "install", "--upgrade", "synth-ai>=0.2.23.dev4"],
            cwd=backend_dir,
        )
    
    if exit_code != 0:
        print(f"⚠️  Installation warning:\n{stderr}")
        print("You may need to install manually: pip install --upgrade 'synth-ai>=0.2.23.dev4'")
    else:
        print("✓ Installed successfully")
        if stdout:
            print(stdout[-300:])
    
    # Step 5: Verify import
    print("\n[5/5] Verifying import...")
    exit_code, stdout, stderr = run_cmd(
        ["python3", "-c", "import synth_ai.cloudflare; print('✓ synth_ai.cloudflare imported'); print(f'open_quick_tunnel: {hasattr(synth_ai.cloudflare, \"open_quick_tunnel\")}')"],
        cwd=backend_dir,
    )
    if exit_code == 0:
        print(stdout)
    else:
        print(f"⚠️  Import check failed:\n{stderr}")
        print("You may need to restart the backend server")
    
    print("\n" + "=" * 60)
    print("✅ Complete! synth-ai 0.2.23.dev4 is published and installed.")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())

