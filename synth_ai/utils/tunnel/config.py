"""
Utility functions for storing tunnel credentials in .env files.
"""
from pathlib import Path
from typing import Optional

from synth_ai.utils.env import write_env_var_to_dotenv


def store_tunnel_credentials(
    tunnel_url: str,
    access_client_id: Optional[str] = None,
    access_client_secret: Optional[str] = None,
    env_file: Optional[Path] = None,
) -> None:
    """
    Store tunnel credentials in .env file for optimizer to use.
    
    Writes:
    - TASK_APP_URL=<tunnel_url>
    - CF_ACCESS_CLIENT_ID=<client_id> (if Access enabled)
    - CF_ACCESS_CLIENT_SECRET=<client_secret> (if Access enabled)
    
    Args:
        tunnel_url: Public tunnel URL (e.g., "https://cust-abc123.usesynth.ai")
        access_client_id: Cloudflare Access client ID (optional)
        access_client_secret: Cloudflare Access client secret (optional)
        env_file: Path to .env file (defaults to .env in current directory)
    """
    write_env_var_to_dotenv(
        "TASK_APP_URL",
        tunnel_url,
        output_file_path=env_file,
        print_msg=True,
        mask_msg=False,
    )
    
    if access_client_id:
        write_env_var_to_dotenv(
            "CF_ACCESS_CLIENT_ID",
            access_client_id,
            output_file_path=env_file,
            print_msg=True,
            mask_msg=True,
        )
    
    if access_client_secret:
        write_env_var_to_dotenv(
            "CF_ACCESS_CLIENT_SECRET",
            access_client_secret,
            output_file_path=env_file,
            print_msg=True,
            mask_msg=True,
        )

