"""
Utilities for managing experiments in the tracing system.
"""
import uuid
import random
import subprocess
from datetime import datetime
from typing import Optional, Dict, List, Any


def get_git_info() -> Dict[str, str]:
    """Get current git branch and commit hash."""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], 
            text=True
        ).strip()
        
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            text=True
        ).strip()
        
        return {"branch": branch, "commit": commit}
    except:
        return {"branch": "unknown", "commit": "unknown"}


def generate_pet_name() -> str:
    """Generate a random pet name for experiments."""
    adjectives = [
        "happy", "clever", "brave", "gentle", "wise", "swift", "bright",
        "calm", "eager", "bold", "keen", "quick", "sharp", "smart",
        "nimble", "agile", "alert", "peppy", "jolly", "merry"
    ]
    
    animals = [
        "panda", "dolphin", "eagle", "fox", "owl", "wolf", "bear",
        "tiger", "lion", "hawk", "falcon", "raven", "otter", "seal",
        "whale", "shark", "octopus", "turtle", "koala", "penguin"
    ]
    
    return f"{random.choice(adjectives)}-{random.choice(animals)}"


def create_experiment_context(
    db_manager,
    experiment_name: Optional[str] = None,
    description: str = "",
    system_name: str = "crafter-agent",
    system_description: str = "Crafter ReAct Agent"
) -> Dict[str, Any]:
    """
    Create an experiment with system tracking.
    
    Returns a dict with experiment_id and system info.
    """
    # Generate IDs
    experiment_id = str(uuid.uuid4())
    system_id = str(uuid.uuid4())
    version_id = str(uuid.uuid4())
    
    # Get git info
    git_info = get_git_info()
    
    # Auto-generate experiment name if not provided
    if not experiment_name:
        experiment_name = generate_pet_name()
    
    # Create system
    db_manager.create_system(
        system_id=system_id,
        name=system_name,
        description=system_description
    )
    
    # Create system version
    db_manager.create_system_version(
        version_id=version_id,
        system_id=system_id,
        branch=git_info["branch"],
        commit=git_info["commit"],
        description=f"Version at {datetime.utcnow().isoformat()}"
    )
    
    # Create experiment
    db_manager.create_experiment(
        experiment_id=experiment_id,
        name=experiment_name,
        description=description or f"Experiment {experiment_name}",
        system_versions=[{
            "system_id": system_id,
            "system_version_id": version_id
        }]
    )
    
    return {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "system_id": system_id,
        "system_version_id": version_id,
        "git_branch": git_info["branch"],
        "git_commit": git_info["commit"]
    }