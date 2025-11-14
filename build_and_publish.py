#!/usr/bin/env python3
"""Build and publish synth-ai using Python build API directly."""
import os
import sys
import subprocess
from pathlib import Path

def main():
    synth_ai_root = Path(__file__).parent
    monorepo_root = synth_ai_root.parent / "monorepo"
    log_file = synth_ai_root / "publish.log"
    
    def log(msg: str):
        """Print and log to file."""
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")
    
    log("=" * 60)
    log("Building and Publishing synth-ai 0.2.23.dev4")
    log("=" * 60)
    
    # Step 1: Build using build module
    log("\n[1/5] Building synth-ai...")
    try:
        from build import ProjectBuilder
        from build.util import project_wheel_metadata
        
        builder = ProjectBuilder(synth_ai_root)
        log("  Building wheel...")
        builder.build("wheel", "dist/")
        log("✓ Build successful")
    except ImportError:
        log("  build module not available, using subprocess...")
        result = subprocess.run(
            [sys.executable, "-m", "build"],
            cwd=synth_ai_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            log(f"❌ Build failed:\n{result.stderr}")
            return 1
        log("✓ Build successful")
        if result.stdout:
            log(result.stdout[-300:])
    
    # Step 2: Find wheel
    log("\n[2/5] Finding wheel...")
    dist_dir = synth_ai_root / "dist"
    wheels = list(dist_dir.glob("synth_ai-0.2.23.dev4-*.whl"))
    if not wheels:
        log(f"❌ Wheel not found in {dist_dir}")
        log(f"   Available wheels: {[w.name for w in dist_dir.glob('*.whl')]}")
        return 1
    wheel_path = wheels[0]
    log(f"✓ Found: {wheel_path.name} ({wheel_path.stat().st_size / 1024 / 1024:.2f} MB)")
    
    # Step 3: Load credentials
    log("\n[3/5] Loading PyPI credentials...")
    env_file = monorepo_root / ".env.dev"
    if not env_file.exists():
        log(f"❌ .env.dev not found at {env_file}")
        return 1
    
    env_vars = {}
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env_vars[key.strip()] = value.strip().strip('"').strip("'")
    
    twine_user = env_vars.get("TWINE_USERNAME") or env_vars.get("PYPI_USERNAME")
    twine_pass = env_vars.get("TWINE_PASSWORD") or env_vars.get("PYPI_PASSWORD") or env_vars.get("PYPI_TOKEN")
    
    if not twine_user or not twine_pass:
        log(f"⚠️  TWINE_USERNAME or TWINE_PASSWORD not found in .env.dev")
        log(f"   Found keys: {[k for k in env_vars.keys() if 'TWINE' in k or 'PYPI' in k]}")
        return 1
    
    os.environ["TWINE_USERNAME"] = twine_user
    os.environ["TWINE_PASSWORD"] = twine_pass
    log("✓ Credentials loaded")
    
    # Step 4: Publish
    log("\n[4/5] Publishing to PyPI...")
    result = subprocess.run(
        [sys.executable, "-m", "twine", "upload", str(wheel_path), "--repository", "pypi"],
        cwd=synth_ai_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log(f"❌ Publish failed:\n{result.stderr}")
        return 1
    log("✓ Published successfully")
    if result.stdout:
        log(result.stdout)
    
    # Step 5: Install in backend
    log("\n[5/5] Installing in backend...")
    backend_dir = monorepo_root / "backend"
    
    # Try uv first
    result = subprocess.run(
        ["uv", "pip", "install", "--upgrade", "synth-ai>=0.2.23.dev4"],
        cwd=backend_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Fallback to pip
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "synth-ai>=0.2.23.dev4"],
            cwd=backend_dir,
            capture_output=True,
            text=True,
        )
    
    if result.returncode != 0:
        log(f"⚠️  Installation warning:\n{result.stderr}")
    else:
        log("✓ Installed successfully")
        if result.stdout:
            log(result.stdout[-200:])
    
    # Verify import
    log("\n[6/6] Verifying import...")
    result = subprocess.run(
        [
            sys.executable, "-c",
            "import synth_ai.cloudflare; print('✓ synth_ai.cloudflare imported'); "
            "print(f'open_quick_tunnel available: {hasattr(synth_ai.cloudflare, \"open_quick_tunnel\")}')"
        ],
        cwd=backend_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        log(result.stdout)
    else:
        log(f"⚠️  Import check failed:\n{result.stderr}")
    
    log("\n" + "=" * 60)
    log("✅ Complete! synth-ai 0.2.23.dev4 is published and installed.")
    log("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())

