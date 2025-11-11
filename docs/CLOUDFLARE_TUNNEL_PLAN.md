# Cloudflare Tunnel Integration Plan

## Overview

Add Cloudflare Tunnel as a third deployment runtime option alongside `local` (uvicorn) and `modal`. This enables users to keep their task app code and data **local** while allowing Synth's hosted prompt optimizer/RL trainer to reach it via a stable, Access-protected public URL.

**CRITICAL REQUIREMENT**: This implementation MUST support all three training types:
- ✅ **RL Training** (`synth-ai train --type rl`)
- ✅ **GEPA Training** (`synth-ai train --type gepa`)  
- ✅ **MIPRO Training** (`synth-ai train --type mipro`)

All three training types call the `/rollout` endpoint on task apps, and all MUST include Cloudflare Access headers when the task app is behind a tunnel. See **Phase 5: Optimizer Integration** for implementation details.

## Current State Analysis

### Existing Deployment Mechanisms

1. **Local Runtime** (`--runtime local`)
   - Uses `uvicorn` to serve FastAPI task app on `localhost:<port>`
   - Only accessible from local machine
   - Used for development/testing
   - Location: `synth_ai/uvicorn.py`, `synth_ai/cli/deploy.py`

2. **Modal Runtime** (`--runtime modal`)
   - Shell-executes Modal CLI (`modal deploy` or `modal serve`)
   - Creates public `*.modal.run` URLs
   - Requires Modal account and authentication
   - Writes `TASK_APP_URL` to `.env` after deployment
   - Location: `synth_ai/modal.py`, `synth_ai/cli/deploy.py`

### Key Integration Points

- **Deploy Command**: `synth_ai/cli/deploy.py` - dispatches to runtime-specific handlers
- **Config Classes**: `synth_ai/cfgs.py` - `LocalDeployCfg`, `ModalDeployCfg`
- **URL Persistence**: `synth_ai/utils/env.py` - `write_env_var_to_dotenv()` writes `TASK_APP_URL`
- **Task App URL Usage**: Prompt Learning configs expect `task_app_url` in TOML; RL configs use `task_url` parameter

## Architecture

### Two Operating Modes

#### Mode 1: Quick Tunnels (Ephemeral, Zero Setup)
- **Use Case**: Development, demos, quick trials
- **Setup**: None (no Cloudflare account needed)
- **URL Format**: `https://<random>.trycloudflare.com`
- **Lifetime**: Ephemeral (tunnel dies when `cloudflared` stops)
- **Security**: Public (no Access protection)
- **Billing**: Free (no account required)

#### Mode 2: Named Tunnels (Production, Synth-Managed)
- **Use Case**: Production runs, stable hostnames, secure access
- **Setup**: Synth backend provisions tunnel in Synth's Cloudflare account
- **URL Format**: `https://<customer-id>.usesynth.ai` (or `https://<customer-id>.tunnels.usesynth.ai`)
- **Lifetime**: Persistent until explicitly revoked
- **Security**: Cloudflare Access service token protection
- **Billing**: Synth-owned Cloudflare account (costs accrue to Synth)

## Cost Model Summary

Based on Cloudflare pricing analysis:

- **Tunnels**: $0 (free on all plans)
- **Access Service Tokens**: $0 (don't consume seats)
- **TLS for Deep Subdomains** (`*.tunnels.usesynth.ai`): **$10/month per zone** via Advanced Certificate Manager (ACM)
- **TLS for First-Level Subdomains** (`customer-123.usesynth.ai`): $0 (covered by Universal SSL)

**Recommendation**: Use first-level subdomains (`customer-123.usesynth.ai`) to avoid ACM costs, or consolidate all customers under one `tunnels.usesynth.ai` zone and amortize the $10/month ACM cost across all customers.

## Implementation Plan

### Phase 1: SDK-Side Infrastructure

#### 1.1 New Config Class
**File**: `synth_ai/cfgs.py`

Add `CloudflareTunnelDeployCfg`:
- `task_app_path: Path` (same as LocalDeployCfg)
- `env_api_key: str` (same as LocalDeployCfg)
- `host: str = "127.0.0.1"` (local service host)
- `port: int = 8000` (local service port)
- `mode: Literal["quick", "managed"] = "managed"` (tunnel mode)
- `tunnel_token: str | None = None` (for managed mode, fetched from backend)
- `subdomain: str | None = None` (optional custom subdomain)
- `trace: bool = True` (same as LocalDeployCfg)

#### 1.2 Cloudflare Tunnel Client Module
**File**: `synth_ai/tunnel.py` (new)

Core functions:
- `_which_cloudflared() -> str | None`: Detect `cloudflared` binary (check `PATH`, common install locations)
- `_install_cloudflared_hint() -> str`: Return OS-specific install instructions
- `open_quick_tunnel(port: int) -> tuple[str, subprocess.Popen]`: 
  - Spawn `cloudflared tunnel --url http://127.0.0.1:{port}`
  - Parse stdout for `https://*.trycloudflare.com` URL (regex: `r"https://[a-z0-9-]+\.trycloudflare\.com"`)
  - Return `(public_url, process)` tuple
- `open_managed_tunnel(tunnel_token: str) -> subprocess.Popen`:
  - Spawn `cloudflared tunnel run --token {tunnel_token}`
  - Wait for "Connected" message (optional validation)
  - Return process handle
- `stop_tunnel(process: subprocess.Popen) -> None`: Gracefully terminate tunnel process

Error handling:
- `cloudflared` not found → raise with install hint
- Tunnel fails to start → raise with diagnostic info
- URL parsing timeout → raise with suggestion to check logs

#### 1.3 Tunnel Deployment Handler
**File**: `synth_ai/tunnel_deploy.py` (new)

Main function: `deploy_app_tunnel(cfg: CloudflareTunnelDeployCfg) -> str | None`

Flow:
1. **Start local task app** (reuse `deploy_app_uvicorn` logic but in background thread/process)
   - Load task app module
   - Create ASGI app
   - Start uvicorn in background on `cfg.host:cfg.port`
   - Wait for health check (`GET http://{host}:{port}/health`)

2. **Choose tunnel mode**:
   - **Quick mode**: Call `open_quick_tunnel(cfg.port)`
   - **Managed mode**: 
     - Call Synth backend API: `POST /v1/tunnels` with `{"port": cfg.port, "subdomain": cfg.subdomain}`
     - Backend returns `{"tunnel_token": "...", "hostname": "cust-123.usesynth.ai", "access_client_id": "...", "access_client_secret": "..."}`
     - Call `open_managed_tunnel(token)`
     - Construct URL: `https://{hostname}`

3. **Write URL and credentials to `.env`**: 
   - `write_env_var_to_dotenv("TASK_APP_URL", public_url)`
   - If Access enabled: write `CF_ACCESS_CLIENT_ID` and `CF_ACCESS_CLIENT_SECRET`

4. **Return summary string** (for MCP context) or print to stdout

#### 1.4 Backend API Client (SDK Side)
**File**: `synth_ai/api/tunnel.py` (new)

Function: `create_tunnel(synth_api_key: str, port: int, subdomain: str | None = None) -> dict`
- POST to `{BACKEND_BASE_URL}/v1/tunnels`
- Headers: `Authorization: Bearer {synth_api_key}`
- Body: `{"port": port, "subdomain": subdomain, "mode": "managed", "enable_access": True}`
- Returns: `{"tunnel_token": str, "hostname": str, "access_client_id": str | None, "access_client_secret": str | None}`

### Phase 2: CLI Integration

#### 2.1 Extend Deploy Command
**File**: `synth_ai/cli/deploy.py`

Changes:
1. **Update `RuntimeType`**: Add `"tunnel"` to literal union
2. **Add CLI options** (tunnel-specific):
   ```python
   @click.option(
       "--tunnel-mode",
       "tunnel_mode",
       type=click.Choice(["quick", "managed"], case_sensitive=False),
       default="managed",
       help="Tunnel mode: quick (ephemeral) or managed (stable)"
   )
   @click.option(
       "--tunnel-subdomain",
       "tunnel_subdomain",
       type=str,
       default=None,
       help="Custom subdomain for managed tunnel (e.g., 'my-company')"
   )
   ```
3. **Update `deploy_cmd` handler**: Add `case "tunnel":` branch
   - Create `CloudflareTunnelDeployCfg`
   - Call `deploy_app_tunnel(cfg)`

#### 2.2 Standalone Tunnel Command (Optional)
**File**: `synth_ai/cli/tunnel.py` (new)

Commands:
- `synth-ai tunnel open --port 8000 --mode quick|managed`
- `synth-ai tunnel close` (find running tunnel process and stop)
- `synth-ai tunnel status` (show active tunnel URL if running)

Use case: Users who already have task app running locally can just open tunnel without full deploy.

### Phase 3: Backend API (Synth Platform)

#### 3.1 Database Schema

**New Table: `cloudflare_tunnels`**

```sql
CREATE TABLE cloudflare_tunnels (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL REFERENCES synth_customers(org_id),
    tunnel_id TEXT NOT NULL UNIQUE,  -- Cloudflare tunnel ID
    hostname TEXT NOT NULL UNIQUE,    -- e.g., "customer-123.usesynth.ai" or "customer-123.tunnels.usesynth.ai"
    subdomain TEXT,                    -- User-requested subdomain (optional)
    local_port INTEGER NOT NULL,      -- Port tunnel points to (e.g., 8000)
    tunnel_token TEXT NOT NULL,       -- Encrypted token for cloudflared
    access_client_id TEXT,            -- Cloudflare Access client ID (if Access enabled)
    access_client_secret TEXT,       -- Encrypted Access client secret
    is_active BOOLEAN DEFAULT true,
    is_billed BOOLEAN DEFAULT false,  -- Whether this tunnel incurs passthrough costs
    billing_feature_id TEXT,          -- Feature ID in Autumn for billing (e.g., "tunnel:managed")
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    revoked_at TIMESTAMP,
    expires_at TIMESTAMP,             -- Optional: auto-revoke after N days
    metadata JSONB DEFAULT '{}'       -- Store ACM zone info, DNS record IDs, etc.
);

CREATE INDEX idx_cloudflare_tunnels_org_id ON cloudflare_tunnels(org_id);
CREATE INDEX idx_cloudflare_tunnels_hostname ON cloudflare_tunnels(hostname);
CREATE INDEX idx_cloudflare_tunnels_active ON cloudflare_tunnels(is_active) WHERE is_active = true;
```

**New Table: `tunnel_usage_events` (for analytics)**

```sql
CREATE TABLE tunnel_usage_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tunnel_id UUID NOT NULL REFERENCES cloudflare_tunnels(id),
    org_id UUID NOT NULL REFERENCES synth_customers(org_id),
    event_type TEXT NOT NULL,  -- 'created', 'revoked', 'renewed', 'billing_cycle'
    cost_cents NUMERIC(20,6),  -- Passthrough cost for this event
    billing_period_start DATE,
    billing_period_end DATE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_tunnel_usage_org_id ON tunnel_usage_events(org_id);
CREATE INDEX idx_tunnel_usage_billing_period ON tunnel_usage_events(billing_period_start, billing_period_end);
```

#### 3.2 Tunnel Provisioning Endpoint
**File**: `backend/app/api/v1/routes_tunnels.py` (new)

**Endpoint**: `POST /v1/tunnels`

**Request**:
```json
{
  "port": 8000,
  "subdomain": "my-company",
  "mode": "managed",
  "enable_access": true
}
```

**Response**:
```json
{
  "tunnel_token": "eyJ...",
  "hostname": "cust-abc123.usesynth.ai",
  "public_url": "https://cust-abc123.usesynth.ai",
  "access_client_id": "abc123...",
  "access_client_secret": "xyz789...",
  "billing_enabled": false,
  "estimated_monthly_cost_cents": null
}
```

**Backend Implementation**:
1. **Authenticate**: Verify `SYNTH_API_KEY` from Authorization header
2. **Get customer ID**: From API key → customer mapping
3. **Generate hostname**: Use subdomain if provided, else random (see hostname generation logic)
4. **Create Named Tunnel** (Cloudflare API):
   ```
   POST https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/cfd_tunnel
   Headers: Authorization: Bearer {CF_API_TOKEN}
   Body: {"name": "synth-{customer_id}", "config_src": "cloudflare"}
   ```
   Returns: `{id, token}`

5. **Configure Ingress** (Cloudflare API):
   ```
   PUT /accounts/{ACCOUNT_ID}/cfd_tunnel/{tunnel_id}/configurations
   Body: {
     "config": {
       "ingress": [
         {"hostname": "{hostname}", "service": "http://localhost:{port}", "originRequest": {}},
         {"service": "http_status:404"}
       ]
     }
   }
   ```

6. **Create DNS CNAME** (Cloudflare API):
   ```
   POST /zones/{ZONE_ID}/dns_records
   Body: {
     "type": "CNAME",
     "name": "{subdomain_part}",
     "content": "{tunnel_id}.cfargotunnel.com",
     "proxied": true
   }
   ```

7. **Create Access Application** (if enabled):
   ```
   POST /accounts/{ACCOUNT_ID}/access/apps
   Body: {
     "name": "synth-tunnel-{customer_id}",
     "domain": "{hostname}",
     "type": "self_hosted"
   }
   ```

8. **Create Service Token** (if enabled):
   ```
   POST /accounts/{ACCOUNT_ID}/access/service_tokens
   Body: {"name": "synth-optimizer-{customer_id}", "duration": "8760h"}
   ```
   Returns: `{client_id, client_secret}`

9. **Determine billing**: Check if hostname requires ACM (deep subdomain)
10. **Store mapping**: Save tunnel metadata in Synth DB
11. **Return**: `{tunnel_token, hostname, access_client_id, access_client_secret, billing_enabled, estimated_monthly_cost_cents}`

#### 3.3 Hostname Generation Logic
**File**: `backend/app/core/tunnel_hostnames.py` (new)

```python
import secrets
import re
import os
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.models.cloudflare_tunnels import CloudflareTunnel

# Configuration
TUNNEL_DOMAIN = os.getenv("CF_TUNNEL_DOMAIN", "usesynth.ai")  # Use first-level: customer-123.usesynth.ai

def _sanitize_subdomain(subdomain: str) -> str:
    """Sanitize user-provided subdomain to be DNS-safe."""
    sanitized = re.sub(r'[^a-z0-9-]', '', subdomain.lower())
    sanitized = sanitized[:63]  # DNS label max length
    sanitized = sanitized.strip('-')  # Remove leading/trailing hyphens
    return sanitized or None

async def _generate_hostname(
    db: AsyncSession,
    org_id: str,
    requested_subdomain: str | None = None,
) -> str:
    """
    Generate a unique hostname for a tunnel.
    
    Strategy:
    - If user provides subdomain: validate, check uniqueness, use it
    - Else: generate random subdomain (e.g., "cust-abc123")
    """
    if requested_subdomain:
        sanitized = _sanitize_subdomain(requested_subdomain)
        if not sanitized:
            raise ValueError("Invalid subdomain format")
        
        # Check uniqueness
        existing = await db.execute(
            select(CloudflareTunnel).where(
                CloudflareTunnel.hostname == f"{sanitized}.{TUNNEL_DOMAIN}"
            )
        )
        if existing.scalar_one_or_none():
            raise ValueError(f"Subdomain {sanitized} already taken")
        
        return f"{sanitized}.{TUNNEL_DOMAIN}"
    
    # Generate random subdomain
    max_attempts = 10
    for _ in range(max_attempts):
        random_part = secrets.token_hex(6)  # 12 hex chars
        hostname = f"cust-{random_part}.{TUNNEL_DOMAIN}"
        
        # Check uniqueness
        existing = await db.execute(
            select(CloudflareTunnel).where(CloudflareTunnel.hostname == hostname)
        )
        if not existing.scalar_one_or_none():
            return hostname
    
    raise RuntimeError("Failed to generate unique hostname after multiple attempts")
```

#### 3.4 Billing Determination Logic
**File**: `backend/app/core/tunnel_billing.py` (new)

```python
from decimal import Decimal
import os

# Configuration
CF_ACM_COST_CENTS_PER_MONTH = 1000  # $10/month = 1000 cents
CF_TUNNEL_DOMAIN = os.getenv("CF_TUNNEL_DOMAIN", "usesynth.ai")

def _is_deep_subdomain(hostname: str) -> bool:
    """Check if hostname is a deep subdomain (e.g., *.tunnels.usesynth.ai)."""
    parts = hostname.split('.')
    return len(parts) > 3

async def _determine_billing(hostname: str) -> tuple[bool, int | None]:
    """
    Determine if tunnel should be billed and estimated monthly cost.
    
    Returns:
        (billing_enabled: bool, estimated_monthly_cost_cents: int | None)
    """
    # Quick tunnels: never billed
    if hostname.endswith('.trycloudflare.com'):
        return False, None
    
    # First-level subdomains: no ACM cost (Universal SSL covers)
    if not _is_deep_subdomain(hostname):
        return False, None  # Free
    
    # Deep subdomains: require ACM, bill passthrough
    # For MVP: bill $10/month per tunnel if deep subdomain
    return True, CF_ACM_COST_CENTS_PER_MONTH

async def _track_tunnel_billing(
    org_id: str,
    tunnel_id: UUID,
    cost_cents: int,
):
    """
    Track tunnel billing in Autumn billing system.
    
    Creates a recurring monthly charge for the tunnel.
    """
    from app.customer.pricing.service import get_autumn_pricing_service
    
    autumn = await get_autumn_pricing_service()
    
    # Create or update feature mapping for tunnel billing
    feature_id = f"tunnel:managed:{tunnel_id}"
    
    # Track as monthly recurring charge
    await autumn.track(
        customer_id=org_id,
        feature_id=feature_id,
        quantity=1,  # 1 tunnel
        unit_price_cents=cost_cents,
        metadata={
            "tunnel_id": str(tunnel_id),
            "billing_type": "recurring_monthly",
        }
    )
```

#### 3.5 Cloudflare API Integration
**File**: `backend/app/core/cloudflare_api.py` (new)

```python
import httpx
import os
from typing import Dict, Any

CF_API_BASE = "https://api.cloudflare.com/client/v4"
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID")
CF_API_TOKEN = os.getenv("CF_API_TOKEN")
CF_ZONE_ID = os.getenv("CF_ZONE_ID")
CF_TUNNEL_DOMAIN = os.getenv("CF_TUNNEL_DOMAIN", "usesynth.ai")

async def _provision_cloudflare_tunnel(
    hostname: str,
    local_port: int,
    enable_access: bool = True,
) -> Dict[str, Any]:
    """
    Provision a Cloudflare Tunnel via API.
    
    Steps:
    1. Create tunnel
    2. Configure ingress (hostname -> localhost:port)
    3. Create DNS CNAME
    4. (Optional) Create Access application + service token
    """
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}"}
    
    async with httpx.AsyncClient() as client:
        # 1. Create tunnel
        tunnel_name = f"synth-{hostname.split('.')[0]}"
        create_resp = await client.post(
            f"{CF_API_BASE}/accounts/{CF_ACCOUNT_ID}/cfd_tunnel",
            headers=headers,
            json={"name": tunnel_name, "config_src": "cloudflare"},
            timeout=30.0,
        )
        create_resp.raise_for_status()
        tunnel_data = create_resp.json()["result"]
        tunnel_id = tunnel_data["id"]
        tunnel_token = tunnel_data["token"]
        
        # 2. Configure ingress
        ingress_config = {
            "config": {
                "ingress": [
                    {
                        "hostname": hostname,
                        "service": f"http://localhost:{local_port}",
                        "originRequest": {}
                    },
                    {"service": "http_status:404"}  # Catch-all
                ]
            }
        }
        ingress_resp = await client.put(
            f"{CF_API_BASE}/accounts/{CF_ACCOUNT_ID}/cfd_tunnel/{tunnel_id}/configurations",
            headers=headers,
            json=ingress_config,
            timeout=30.0,
        )
        ingress_resp.raise_for_status()
        
        # 3. Create DNS CNAME
        subdomain_part = hostname.split('.')[0]
        dns_resp = await client.post(
            f"{CF_API_BASE}/zones/{CF_ZONE_ID}/dns_records",
            headers=headers,
            json={
                "type": "CNAME",
                "name": subdomain_part,
                "content": f"{tunnel_id}.cfargotunnel.com",
                "proxied": True,
            },
            timeout=30.0,
        )
        dns_resp.raise_for_status()
        
        # 4. Create Access application + service token (if enabled)
        access_client_id = None
        access_client_secret = None
        if enable_access:
            access_data = await _create_access_application(
                client, headers, hostname, tunnel_id
            )
            access_client_id = access_data["client_id"]
            access_client_secret = access_data["client_secret"]
        
        return {
            "tunnel_id": tunnel_id,
            "tunnel_token": tunnel_token,
            "access_client_id": access_client_id,
            "access_client_secret": access_client_secret,
        }

async def _create_access_application(
    client: httpx.AsyncClient,
    headers: Dict[str, str],
    hostname: str,
    tunnel_id: str,
) -> Dict[str, str]:
    """Create Cloudflare Access application and service token."""
    # Create Access app
    app_resp = await client.post(
        f"{CF_API_BASE}/accounts/{CF_ACCOUNT_ID}/access/apps",
        headers=headers,
        json={
            "name": f"synth-tunnel-{tunnel_id}",
            "domain": hostname,
            "type": "self_hosted",
            "session_duration": "24h",
        },
        timeout=30.0,
    )
    app_resp.raise_for_status()
    app_id = app_resp.json()["result"]["id"]
    
    # Create policy: allow service tokens
    policy_resp = await client.post(
        f"{CF_API_BASE}/accounts/{CF_ACCOUNT_ID}/access/apps/{app_id}/policies",
        headers=headers,
        json={
            "name": "Service Token Only",
            "decision": "allow",
            "include": [{"service_token": {}}],
        },
        timeout=30.0,
    )
    policy_resp.raise_for_status()
    
    # Create service token
    token_resp = await client.post(
        f"{CF_API_BASE}/accounts/{CF_ACCOUNT_ID}/access/service_tokens",
        headers=headers,
        json={
            "name": f"synth-optimizer-{tunnel_id}",
            "duration": "8760h",  # 1 year
        },
        timeout=30.0,
    )
    token_resp.raise_for_status()
    token_data = token_resp.json()["result"]
    
    return {
        "client_id": token_data["client_id"],
        "client_secret": token_data["client_secret"],
    }
```

#### 3.6 Tunnel Revocation Endpoint
**Endpoint**: `DELETE /v1/tunnels/{tunnel_id}`

**Backend Implementation**:
1. Verify ownership (customer_id matches tunnel)
2. Delete DNS record
3. Delete tunnel configuration (or mark inactive)
4. Optionally revoke Access service token
5. Stop billing if applicable

#### 3.7 Backend Configuration Requirements

**Environment Variables** (Synth backend):
- `CF_API_TOKEN`: Cloudflare API token with permissions:
  - `Account:Cloudflare Tunnel:Edit`
  - `Zone:DNS:Edit`
  - `Account:Access:Apps and Policies:Edit` (if using Access)
- `CF_ACCOUNT_ID`: Cloudflare account ID
- `CF_ZONE_ID`: Zone ID for `usesynth.ai` (or chosen domain)
- `CF_TUNNEL_DOMAIN`: Domain for tunnel hostnames (e.g., `usesynth.ai` or `tunnels.usesynth.ai`)
- `CF_ACM_COST_CENTS_PER_MONTH`: 1000 (for billing calculations)

### Phase 4: Security & Access Control

#### 4.1 Cloudflare Access Integration

**Why**: Restrict tunnel hostname to only Synth's optimizer service

**Setup** (one-time, in Synth Cloudflare account):
1. Create Access application for tunnel hostnames
2. Create policy: "Service Token" → Allow
3. Create service token per customer (or one shared token for optimizer)

**Optimizer Request Headers**:
```http
CF-Access-Client-Id: <CLIENT_ID>
CF-Access-Client-Secret: <CLIENT_SECRET>
X-API-Key: <ENVIRONMENT_API_KEY>
```

**Alternative**: Single `Authorization` header if Access supports it

#### 4.2 Task App Authentication

**Existing**: Task apps require `X-API-Key: ENVIRONMENT_API_KEY` header

**Keep**: This remains the primary auth mechanism. Cloudflare Access is an additional layer.

#### 4.3 SDK Configuration Storage
**File**: `synth_ai/utils/tunnel_config.py` (new)

```python
import os
from pathlib import Path
from typing import Optional

def store_tunnel_credentials(
    tunnel_url: str,
    access_client_id: Optional[str] = None,
    access_client_secret: Optional[str] = None,
    env_file: Optional[Path] = None,
):
    """
    Store tunnel credentials in .env file for optimizer to use.
    
    Writes:
    - TASK_APP_URL=<tunnel_url>
    - CF_ACCESS_CLIENT_ID=<client_id> (if Access enabled)
    - CF_ACCESS_CLIENT_SECRET=<client_secret> (if Access enabled)
    """
    from synth_ai.utils.env import write_env_var_to_dotenv
    
    write_env_var_to_dotenv("TASK_APP_URL", tunnel_url, output_file_path=env_file)
    
    if access_client_id:
        write_env_var_to_dotenv(
            "CF_ACCESS_CLIENT_ID",
            access_client_id,
            output_file_path=env_file,
        )
    
    if access_client_secret:
        write_env_var_to_dotenv(
            "CF_ACCESS_CLIENT_SECRET",
            access_client_secret,
            output_file_path=env_file,
            mask_msg=True,  # Mask secret in logs
        )
```

### Phase 5: Optimizer Integration

**CRITICAL REQUIREMENT**: All three training types (RL, GEPA, MIPRO) MUST support Cloudflare Tunnel with Access headers.

#### 5.1 Shared HTTP Client Helper
**File**: `monorepo/backend/app/routes/shared/task_app_client.py` (new)

Create a shared helper that ALL training types use for calling task app `/rollout` endpoint:

```python
"""
Shared HTTP client for calling task app endpoints.

Used by:
- RL Training (ClusteredGRPOLudicTrainer)
- GEPA Optimizer
- MIPRO Optimizer

All three training types call /rollout endpoint and need Cloudflare Access headers
when task app is behind a tunnel.
"""
import os
import httpx
from typing import Dict, Any, Optional

def build_task_app_headers(task_app_api_key: Optional[str] = None) -> Dict[str, str]:
    """
    Build HTTP headers for task app requests.
    
    Includes:
    - X-API-Key: Task app authentication (required)
    - Authorization: Bearer token (for compatibility)
    - CF-Access-Client-Id: Cloudflare Access (if tunnel-protected)
    - CF-Access-Client-Secret: Cloudflare Access (if tunnel-protected)
    
    Args:
        task_app_api_key: API key for task app (falls back to ENVIRONMENT_API_KEY env var)
    
    Returns:
        Dict of HTTP headers
    """
    headers: Dict[str, str] = {
        "Content-Type": "application/json",
    }
    
    # Standard API key (required for all task apps)
    api_key = (task_app_api_key or "").strip() or (os.environ.get("ENVIRONMENT_API_KEY") or "").strip()
    if api_key:
        headers["X-API-Key"] = api_key
        headers["Authorization"] = f"Bearer {api_key}"
    
    # Cloudflare Access credentials (for tunnel-protected apps)
    # These are set by SDK when tunnel is created
    access_client_id = os.environ.get("CF_ACCESS_CLIENT_ID")
    access_client_secret = os.environ.get("CF_ACCESS_CLIENT_SECRET")
    
    if access_client_id and access_client_secret:
        headers["CF-Access-Client-Id"] = access_client_id
        headers["CF-Access-Client-Secret"] = access_client_secret
    
    return headers

async def call_task_app_rollout(
    task_app_url: str,
    task_app_api_key: Optional[str],
    payload: Dict[str, Any],
    timeout: float = 300.0,
) -> httpx.Response:
    """
    Call task app /rollout endpoint with proper headers.
    
    This function ensures Cloudflare Access headers are included when task app
    is behind a tunnel, regardless of which training type is calling it.
    
    Args:
        task_app_url: Base URL of task app (e.g., "https://cust-123.usesynth.ai")
        task_app_api_key: API key for authentication
        payload: Rollout request payload
        timeout: Request timeout in seconds
    
    Returns:
        HTTP response from rollout endpoint
    """
    headers = build_task_app_headers(task_app_api_key)
    rollout_url = f"{task_app_url.rstrip('/')}/rollout"
    
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.post(rollout_url, json=payload, headers=headers)
        return response
```

#### 5.2 Update RL Training
**File**: `monorepo/backend/app/routes/clustered_training/core/algorithms/gspo/training/clustered_trainer.py`

**Current code** (line ~2019-2025):
```python
headers = {
    "Content-Type": "application/json",
    "X-API-Key": api_key,
}
async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
    response = await client.post(f"{task_url}/rollout", json=payload, headers=headers)
```

**Updated code**:
```python
from app.routes.shared.task_app_client import build_task_app_headers

headers = build_task_app_headers(api_key)
async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
    response = await client.post(f"{task_url}/rollout", json=payload, headers=headers)
```

**Also update**:
- `monorepo/backend/app/routes/clustered_training/core/algorithms/gspo/evaluation/evaluator.py` (line ~491)
  - Update `_execute_rollout()` function to use `build_task_app_headers()`

#### 5.3 Update GEPA Optimizer
**File**: `monorepo/backend/app/routes/prompt_learning/core/evaluation.py`

**Current code** (line ~699-726):
```python
api_key = (task_app_api_key or "").strip() or (os.environ.get("ENVIRONMENT_API_KEY") or "").strip()
headers = {
    "Content-Type": "application/json",
}
if api_key:
    headers["X-API-Key"] = api_key
    headers["Authorization"] = f"Bearer {api_key}"

rollout_url = f"{task_app_url.rstrip('/')}/rollout"
async with httpx.AsyncClient(timeout=timeout) as client:
    response = await client.post(rollout_url, json=payload, headers=headers)
```

**Updated code**:
```python
from app.routes.shared.task_app_client import build_task_app_headers

headers = build_task_app_headers(task_app_api_key)
rollout_url = f"{task_app_url.rstrip('/')}/rollout"
async with httpx.AsyncClient(timeout=timeout) as client:
    response = await client.post(rollout_url, json=payload, headers=headers)
```

**Also update**:
- `monorepo/backend/app/routes/prompt_learning/core/validation.py` (line ~125-161)
  - Update `fetch_baseline_messages()` to use `build_task_app_headers()`

#### 5.4 Update MIPRO Optimizer
**File**: `monorepo/backend/app/routes/prompt_learning/core/evaluation.py`

MIPRO uses the same `_execute_rollout_request()` function as GEPA (see section 5.3), so updating that function covers both GEPA and MIPRO.

**Verification**: MIPRO calls `evaluate_prompt_template()` which calls `_execute_rollout_request()`, so the fix propagates automatically.

#### 5.5 Testing Requirements

**All three training types MUST be tested with tunnel URLs**:

1. **RL Training Test**:
   ```python
   # Test that RL trainer includes Access headers when calling tunnel URL
   task_url = "https://cust-123.usesynth.ai"  # Tunnel URL
   os.environ["CF_ACCESS_CLIENT_ID"] = "test-client-id"
   os.environ["CF_ACCESS_CLIENT_SECRET"] = "test-secret"
   
   headers = build_task_app_headers("test-api-key")
   assert "CF-Access-Client-Id" in headers
   assert "CF-Access-Client-Secret" in headers
   assert headers["X-API-Key"] == "test-api-key"
   ```

2. **GEPA Test**:
   ```python
   # Test GEPA rollout with tunnel URL
   await _execute_rollout_request(
       seed=0,
       task_app_url="https://cust-123.usesynth.ai",
       task_app_api_key="test-key",
       policy_config={...},
   )
   # Verify headers include Access credentials
   ```

3. **MIPRO Test**:
   ```python
   # Test MIPRO rollout with tunnel URL
   # Same as GEPA - uses same evaluation function
   ```

#### 5.6 Files That MUST Be Updated

**CRITICAL**: The following files MUST be updated to use `build_task_app_headers()` to ensure all training types support tunnels:

1. **RL Training**:
   - `monorepo/backend/app/routes/clustered_training/core/algorithms/gspo/training/clustered_trainer.py`
     - Line ~2019: `_call_task_app_for_rollout()` method
   - `monorepo/backend/app/routes/clustered_training/core/algorithms/gspo/evaluation/evaluator.py`
     - Line ~491: `_execute_rollout()` function

2. **GEPA Training**:
   - `monorepo/backend/app/routes/prompt_learning/core/evaluation.py`
     - Line ~699: `_execute_rollout_request()` function (used by GEPA)
   - `monorepo/backend/app/routes/prompt_learning/core/validation.py`
     - Line ~125: `fetch_baseline_messages()` function

3. **MIPRO Training**:
   - `monorepo/backend/app/routes/prompt_learning/core/evaluation.py`
     - Line ~699: `_execute_rollout_request()` function (used by MIPRO - same as GEPA)

**Verification Checklist**:
- [ ] All files updated to import `build_task_app_headers` from `app.routes.shared.task_app_client`
- [ ] All manual header construction replaced with `build_task_app_headers()` call
- [ ] Unit tests verify Access headers are included when env vars are set
- [ ] Integration tests verify RL/GEPA/MIPRO work with tunnel URLs
- [ ] No hardcoded header dictionaries remain (all use shared helper)

#### 5.7 Backward Compatibility

- **Non-tunnel URLs**: If `CF_ACCESS_CLIENT_ID` and `CF_ACCESS_CLIENT_SECRET` are not set, headers only include `X-API-Key` (existing behavior)
- **Tunnel URLs**: If Access credentials are present in env, they're automatically included
- **No breaking changes**: Existing code continues to work; tunnel support is additive
- **All training types**: RL, GEPA, and MIPRO automatically get tunnel support once shared helper is used

### Phase 6: URL Discovery & Configuration

#### 6.1 URL Writing (Same Pattern as Modal)

After tunnel opens:
- Write `TASK_APP_URL=https://...` to `.env` (via `write_env_var_to_dotenv`)
- Write `CF_ACCESS_CLIENT_ID` and `CF_ACCESS_CLIENT_SECRET` if Access enabled
- Print URL to stdout
- Return URL in MCP context

#### 6.2 Config File Integration

**Prompt Learning Configs** (`task_app_url` in TOML):
- User can set `task_app_url = "https://cust-123.usesynth.ai"` manually
- Or SDK auto-populates from `.env` after tunnel deploy

**RL Configs** (`task_url` parameter):
- Same pattern: read from `.env` or config file

### Phase 7: Error Handling & Diagnostics

#### 7.1 Common Failure Modes

1. **`cloudflared` not installed**:
   - Error message with install instructions
   - macOS: `brew install cloudflare/cloudflare/cloudflared`
   - Linux: Download from Cloudflare or package manager
   - Windows: Download installer

2. **Tunnel fails to connect**:
   - Check local service is running (`/health` endpoint)
   - Verify port matches tunnel config
   - Check firewall (shouldn't be needed, but log warning)

3. **Backend API failure** (managed mode):
   - Fallback suggestion: use `--tunnel-mode quick`
   - Log error details

4. **DNS propagation delay** (managed mode):
   - Warn user: "DNS may take 1-2 minutes to propagate"
   - Provide fallback: use Quick Tunnel for immediate testing

5. **Subdomain taken**:
   - Return clear error with suggestion to try different name

6. **Insufficient credits** (if billing enabled):
   - Check org balance before provisioning
   - Return error with current balance and required amount

#### 7.2 Logging

- Log tunnel process PID
- Log public URL immediately when available
- Log tunnel termination (cleanup)
- Mask sensitive values (tokens, secrets) in logs

### Phase 8: Testing Strategy

#### 8.1 Unit Tests

**File**: `tests/unit/tunnel/test_tunnel.py`

- `test_which_cloudflared()`: Mock `shutil.which`
- `test_open_quick_tunnel()`: Mock subprocess, verify URL parsing
- `test_open_managed_tunnel()`: Mock subprocess, verify token passed
- `test_stop_tunnel()`: Mock process termination
- `test_hostname_sanitization()`: Test subdomain validation

#### 8.2 Integration Tests

**File**: `tests/integration/tunnel/test_tunnel_deploy.py`

- `test_deploy_tunnel_quick()`: End-to-end quick tunnel
  - Start local task app
  - Open quick tunnel
  - Verify URL written to `.env`
  - Verify `/health` accessible via tunnel URL
- `test_deploy_tunnel_managed()`: End-to-end managed tunnel (requires backend mock)
- `test_tunnel_with_existing_task_app()`: Tunnel to already-running app
- `test_custom_subdomain()`: Test subdomain uniqueness validation
- `test_access_protection()`: Verify Access headers required

#### 8.3 Manual Testing Checklist

- [ ] Quick tunnel opens and URL is accessible
- [ ] Managed tunnel provisions via backend API
- [ ] URL written to `.env` correctly
- [ ] Access credentials written to `.env` correctly
- [ ] Task app endpoints (`/rollout`, `/health`) work via tunnel
- [ ] Access protection works (unauthorized requests blocked)
- [ ] **RL Training works with tunnel URL** (test full RL job with tunnel-protected task app)
- [ ] **GEPA Training works with tunnel URL** (test GEPA optimization with tunnel-protected task app)
- [ ] **MIPRO Training works with tunnel URL** (test MIPRO optimization with tunnel-protected task app)
- [ ] All three training types include Access headers in rollout requests
- [ ] Tunnel stops cleanly on Ctrl+C
- [ ] `cloudflared` not found error is helpful
- [ ] Custom subdomain validation works
- [ ] Subdomain uniqueness check works
- [ ] Works on macOS, Linux, Windows (if supported)

### Phase 9: Documentation

#### 9.1 CLI Documentation

**File**: `docs/cli/deploy.md` (update)

Add section:
```markdown
## Deploy via Cloudflare Tunnel

Expose your local task app to the internet via Cloudflare Tunnel:

```bash
uvx synth-ai deploy \
  --task-app task_apps/crafter/task_app.py \
  --runtime tunnel \
  --tunnel-mode managed \
  --port 8000
```

**Modes**:
- `quick`: Ephemeral `*.trycloudflare.com` URL (no account needed)
- `managed`: Stable `*.usesynth.ai` URL (Synth-managed, Access-protected)

**Options**:
- `--tunnel-subdomain`: Custom subdomain (e.g., `my-company` → `my-company.usesynth.ai`)

**Requirements**:
- `cloudflared` installed (see install instructions if missing)
- For managed mode: `SYNTH_API_KEY` configured
```

#### 9.2 Architecture Documentation

**File**: `docs/architecture/tunnels.md` (new)

- Explain Quick vs Managed tunnels
- Security model (Access + X-API-Key)
- Billing model (Synth-owned account)
- Backend provisioning flow

## Pricing Strategy Options

### Option 1: Free (Recommended for MVP)

- All tunnels free (use first-level subdomains, Universal SSL)
- No passthrough billing
- Simple, no billing complexity

### Option 2: Custom Subdomain Fee

- Random subdomains: Free
- Custom subdomain: $10/month (covers ACM if deep subdomain)
- Incentivizes using random subdomains

### Option 3: Zone-Based Amortization

- All tunnels in `tunnels.usesynth.ai` zone share $10/month ACM cost
- Bill each tunnel: `$10 / num_active_tunnels_in_zone`
- More complex, but fairer at scale

**Recommendation**: Start with **Option 1** (free), add Option 2 later if needed.

## Billing Integration

### Monthly Recurring Charge

**File**: `backend/app/workers/tunnel_billing.py` (new)

```python
"""
Worker that runs monthly to bill active tunnels.

Tracks:
- Active tunnels per org
- Amortized ACM costs (if using deep subdomains)
- Records usage events
"""

async def process_monthly_tunnel_billing():
    """
    Run monthly (via cron) to bill active tunnels.
    
    Logic:
    1. Find all active, billed tunnels
    2. Group by zone (if using zone-based amortization)
    3. Calculate cost per tunnel (amortize ACM cost)
    4. Record usage events
    5. Track in Autumn billing
    """
    # TODO: Implement monthly billing cycle
    pass
```

## Environment Variables

### Backend

```bash
# Cloudflare API
CF_ACCOUNT_ID=...
CF_API_TOKEN=...
CF_ZONE_ID=...
CF_TUNNEL_DOMAIN=usesynth.ai  # or tunnels.usesynth.ai

# Billing
CF_ACM_COST_CENTS_PER_MONTH=1000  # $10/month
```

### SDK

```bash
# Standard (already exists)
SYNTH_API_KEY=...
ENVIRONMENT_API_KEY=...

# Tunnel-specific (set by SDK after tunnel creation)
TASK_APP_URL=https://customer-123.usesynth.ai
CF_ACCESS_CLIENT_ID=...  # Only for managed tunnels
CF_ACCESS_CLIENT_SECRET=...  # Only for managed tunnels
```

## Migration & Backward Compatibility

### No Breaking Changes

- Existing `--runtime local` and `--runtime modal` unchanged
- `--runtime tunnel` is additive
- Config files (`task_app_url`) work with tunnel URLs same as Modal URLs

### Gradual Rollout

1. **Phase 1**: SDK-side only (Quick Tunnels)
2. **Phase 2**: Backend API (Managed Tunnels)
3. **Phase 3**: Access integration (security hardening)
4. **Phase 4**: Billing (if needed)

## Open Questions & Decisions Needed

1. **Domain choice**: First-level (`customer-123.usesynth.ai`) or deep (`customer-123.tunnels.usesynth.ai`)?
   - **Recommendation**: First-level (free, simpler)

2. **Billing model**: Free, per-tunnel, or zone-based?
   - **Recommendation**: Free for MVP, add per-tunnel later if needed

3. **Access default**: Always enable Access, or make it optional?
   - **Recommendation**: Always enable for managed tunnels (security)

4. **Tunnel lifetime**: Auto-expire after N days of inactivity, or manual revocation only?
   - **Recommendation**: Manual revocation only (simpler)

5. **Customer isolation**: One tunnel per customer or one per session?
   - **Recommendation**: One per session (simpler, allows multiple concurrent tunnels)

6. **Windows Support**: Full support or macOS/Linux only initially?
   - **Recommendation**: macOS/Linux first, Windows later

7. **`cloudflared` Version**: Pin to specific version or allow any?
   - **Recommendation**: Allow any (simpler), pin later if needed

8. **Fallback Behavior**: If managed tunnel fails, auto-fallback to quick?
   - **Recommendation**: No auto-fallback (explicit user choice)

9. **Subdomain validation**: Allow any DNS-safe string, or restrict format?
   - **Recommendation**: DNS-safe only, max 63 chars, no reserved words

## Success Criteria

- [ ] Users can deploy task app locally and expose via tunnel with one command
- [ ] Tunnel URLs work with existing prompt learning / RL configs
- [ ] **RL Training works with tunnel URLs** (ClusteredGRPOLudicTrainer includes Access headers)
- [ ] **GEPA Training works with tunnel URLs** (GEPA optimizer includes Access headers)
- [ ] **MIPRO Training works with tunnel URLs** (MIPRO optimizer includes Access headers)
- [ ] Managed tunnels are Access-protected (only Synth optimizer can call)
- [ ] Quick tunnels work without any account setup
- [ ] Custom subdomains work with uniqueness validation
- [ ] Billing integration works (if enabled)
- [ ] All `/rollout` endpoint calls include Access headers when tunnel-protected
- [ ] Documentation is clear and complete
- [ ] Error messages guide users to resolution

## Timeline Estimate

- **Phase 1-2** (SDK + CLI): 2-3 days
- **Phase 3** (Backend API): 2-3 days
- **Phase 4** (Access integration): 1-2 days
- **Phase 5** (Optimizer integration): 1 day
- **Phase 6-7** (URL discovery, error handling): 1 day
- **Phase 8** (Testing): 2 days
- **Phase 9** (Documentation): 1 day

**Total**: ~2 weeks for full implementation

## User Workflow: Paying Customer Experience

This section describes how a paying customer would use Cloudflare Tunnel deployment with just their `SYNTH_API_KEY`.

### Prerequisites

1. **Synth API Key**: Customer has `SYNTH_API_KEY` from their Synth dashboard
2. **Environment API Key**: Customer has `ENVIRONMENT_API_KEY` for their task app (can be same as SYNTH_API_KEY or separate)
3. **cloudflared installed**: SDK will check and provide install instructions if missing

### Basic Workflow: Deploy with Tunnel

#### Step 1: Set API Key

```bash
# Option A: Export in shell
export SYNTH_API_KEY="sk-synth-..."

# Option B: Use .env file
echo "SYNTH_API_KEY=sk-synth-..." >> .env
echo "ENVIRONMENT_API_KEY=env-key-..." >> .env
```

#### Step 2: Deploy Task App with Tunnel

```bash
# Deploy task app locally and expose via Cloudflare Tunnel
uvx synth-ai deploy \
  --task-app task_apps/my_app/task_app.py \
  --runtime tunnel \
  --tunnel-mode managed \
  --port 8000
```

**What happens**:
1. SDK validates `SYNTH_API_KEY` (checks format, prompts if missing)
2. SDK calls backend `POST /api/v1/tunnels` with `SYNTH_API_KEY` in Authorization header
3. Backend:
   - Authenticates customer via API key → resolves `org_id`
   - Generates unique hostname (e.g., `cust-abc123.usesynth.ai`)
   - Provisions Cloudflare Tunnel (creates tunnel, configures ingress, DNS, Access)
   - Returns `tunnel_token`, `hostname`, and Access credentials
4. SDK:
   - Starts local task app on port 8000 (uvicorn)
   - Launches `cloudflared tunnel run --token <tunnel_token>`
   - Writes `TASK_APP_URL=https://cust-abc123.usesynth.ai` to `.env`
   - Writes `CF_ACCESS_CLIENT_ID` and `CF_ACCESS_CLIENT_SECRET` to `.env` (for optimizer)
5. Output:
   ```
   ✓ Task app running on http://127.0.0.1:8000
   ✓ Tunnel connected to Cloudflare
   ✓ Public URL: https://cust-abc123.usesynth.ai
   ✓ Wrote TASK_APP_URL to .env
   ```

#### Step 3: Use Tunnel URL in Training Config

The tunnel URL is automatically written to `.env` as `TASK_APP_URL`. Your training configs can reference it:

```toml
# config.toml
[prompt_learning]
algorithm = "mipro"
task_app_url = "${TASK_APP_URL}"  # Reads from .env
task_app_api_key = "${ENVIRONMENT_API_KEY}"  # Reads from .env

[prompt_learning.policy]
model = "gpt-4o-mini"
provider = "openai"
```

Or explicitly set it:

```toml
[prompt_learning]
task_app_url = "https://cust-abc123.usesynth.ai"
task_app_api_key = "env-key-..."
```

#### Step 4: Run Training

```bash
# Training automatically uses tunnel URL from .env
# Works for ALL training types: RL, GEPA, and MIPRO
uvx synth-ai train --type rl --config config.toml      # RL Training
uvx synth-ai train --type gepa --config config.toml   # GEPA Training  
uvx synth-ai train --type mipro --config config.toml  # MIPRO Training
```

**What happens** (same for all training types):
1. Training service reads `TASK_APP_URL` from `.env`
2. Training service reads `CF_ACCESS_CLIENT_ID` and `CF_ACCESS_CLIENT_SECRET` from `.env`
3. **RL/GEPA/MIPRO optimizer** calls tunnel URL with headers:
   ```http
   POST https://cust-abc123.usesynth.ai/rollout
   Content-Type: application/json
   CF-Access-Client-Id: <from .env>
   CF-Access-Client-Secret: <from .env>
   X-API-Key: <ENVIRONMENT_API_KEY>
   Authorization: Bearer <ENVIRONMENT_API_KEY>
   ```
   **Note**: All three training types (RL, GEPA, MIPRO) use the same shared HTTP client helper (`build_task_app_headers()`) which automatically includes Access headers when present in environment. This ensures tunnel support works identically across all training types.
4. Cloudflare edge validates Access token → allows request
5. Task app validates `X-API-Key` → processes request

**Supported Training Types**:
- ✅ **RL Training** (`--type rl`): Uses `ClusteredGRPOLudicTrainer`, calls `/rollout` with Access headers
- ✅ **GEPA Training** (`--type gepa`): Uses `GEPAOptimizer`, calls `/rollout` with Access headers  
- ✅ **MIPRO Training** (`--type mipro`): Uses `MIPROOptimizer`, calls `/rollout` with Access headers

All three types automatically detect tunnel URLs and include Access headers - no special configuration needed!

### Advanced: Custom Subdomain

If customer wants a custom subdomain (e.g., `my-company.usesynth.ai`):

```bash
uvx synth-ai deploy \
  --task-app task_apps/my_app/task_app.py \
  --runtime tunnel \
  --tunnel-mode managed \
  --port 8000 \
  --tunnel-subdomain my-company
```

**What happens**:
1. SDK sends `subdomain: "my-company"` to backend
2. Backend validates subdomain is available (checks uniqueness)
3. If available: provisions tunnel with `my-company.usesynth.ai`
4. If unavailable: returns error with suggestion
5. **Billing**: If using deep subdomain (`*.tunnels.usesynth.ai`), customer may be billed $10/month for ACM

### Quick Tunnel Mode (Free, Ephemeral)

For development/testing, use quick tunnels (no backend API call needed):

```bash
uvx synth-ai deploy \
  --task-app task_apps/my_app/task_app.py \
  --runtime tunnel \
  --tunnel-mode quick \
  --port 8000
```

**What happens**:
1. SDK runs `cloudflared tunnel --url http://127.0.0.1:8000` locally
2. Cloudflare returns random URL: `https://random-abc123.trycloudflare.com`
3. SDK writes `TASK_APP_URL` to `.env`
4. **No billing** (ephemeral, no account needed)
5. **No Access protection** (public URL, anyone can access if they know it)

### Authentication Flow Summary

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐         ┌──────────┐
│   Customer  │         │   Synth SDK  │         │   Backend   │         │Cloudflare│
│   Machine   │         │             │         │             │         │          │
└──────┬──────┘         └──────┬──────┘         └──────┬──────┘         └────┬─────┘
       │                       │                        │                      │
       │ 1. synth-ai deploy    │                        │                      │
       │    --runtime tunnel   │                        │                      │
       ├──────────────────────>│                        │                      │
       │                       │                        │                      │
       │                       │ 2. POST /v1/tunnels   │                      │
       │                       │    Auth: SYNTH_API_KEY │                      │
       │                       ├──────────────────────>│                      │
       │                       │                        │                      │
       │                       │                        │ 3. Create tunnel     │
       │                       │                        ├──────────────────────>│
       │                       │                        │                      │
       │                       │                        │ 4. Tunnel token +   │
       │                       │                        │    Access creds      │
       │                       │                        │<──────────────────────│
       │                       │                        │                      │
       │                       │ 5. tunnel_token +     │                      │
       │                       │    hostname            │                      │
       │                       │<──────────────────────│                      │
       │                       │                        │                      │
       │                       │ 6. Start cloudflared   │                      │
       │                       │    with token          │                      │
       │                       ├────────────────────────┼──────────────────────>│
       │                       │                        │                      │
       │ 7. Tunnel active     │                        │                      │
       │    URL in .env        │                        │                      │
       │<──────────────────────│                        │                      │
       │                       │                        │                      │
       │ 8. Run training      │                        │                      │
       │    (uses TASK_APP_URL)│                        │                      │
       ├──────────────────────>│                        │                      │
       │                       │                        │                      │
       │                       │ 9. Call tunnel URL    │                      │
       │                       │    with Access headers │                      │
       │                       ├────────────────────────┼──────────────────────>│
       │                       │                        │                      │
       │                       │                        │ 10. Validate Access  │
       │                       │                        │     → allow request  │
       │                       │                        │<──────────────────────│
       │                       │                        │                      │
       │                       │ 11. Forward to        │                      │
       │                       │     localhost:8000     │                      │
       │                       ├────────────────────────┼──────────────────────>│
       │                       │                        │                      │
       │                       │ 12. Task app validates│                      │
       │                       │     X-API-Key          │                      │
       │                       │                        │                      │
       │ 13. Training results │                        │                      │
       │<──────────────────────│                        │                      │
```

### Environment Variables Set by SDK

After successful tunnel deployment, `.env` contains:

```bash
# Standard (user-provided)
SYNTH_API_KEY=sk-synth-...
ENVIRONMENT_API_KEY=env-key-...

# Tunnel-specific (set by SDK)
TASK_APP_URL=https://cust-abc123.usesynth.ai
CF_ACCESS_CLIENT_ID=abc123...  # Only for managed tunnels
CF_ACCESS_CLIENT_SECRET=xyz789...  # Only for managed tunnels (encrypted in .env)
```

### Revoking a Tunnel

```bash
# Option 1: Via CLI (if implemented)
uvx synth-ai tunnel revoke --hostname cust-abc123.usesynth.ai

# Option 2: Via API
curl -X DELETE https://api.usesynth.ai/api/v1/tunnels/{tunnel_id} \
  -H "Authorization: Bearer $SYNTH_API_KEY"

# Option 3: Stop cloudflared process manually
# Tunnel will be marked inactive after timeout
```

### Error Handling

**Missing cloudflared**:
```
Error: cloudflared not found. Install it:
  macOS: brew install cloudflare/cloudflare/cloudflared
  Linux: See https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/
```

**Invalid API key**:
```
Error: Invalid SYNTH_API_KEY. Get your key from https://usesynth.ai/dashboard/settings
```

**Subdomain taken**:
```
Error: Subdomain 'my-company' is already taken. Try a different name or use random subdomain.
```

**Insufficient credits** (if billing enabled):
```
Error: Insufficient credits for tunnel provisioning. Required: $10/month. Current balance: $5.00
```

### Cost Transparency

Customers see costs in:
1. **SDK output** (if billing enabled):
   ```
   ✓ Tunnel created: https://cust-abc123.usesynth.ai
   ℹ Estimated monthly cost: $10.00 (ACM certificate for custom subdomain)
   ```

2. **Dashboard** (if implemented):
   - Active tunnels list
   - Monthly tunnel costs
   - Usage breakdown

3. **Billing API** (via Autumn integration):
   - Recurring charges appear in customer's billing dashboard
   - Usage events tracked for transparency

### Security Notes for Customers

- **Access Protection**: Managed tunnels are protected by Cloudflare Access. Only Synth's optimizer can call your tunnel URL (via service token).
- **Task App Auth**: Your `ENVIRONMENT_API_KEY` is still required by the task app itself (defense in depth).
- **Tunnel Token**: The `tunnel_token` is stored securely and never exposed in logs.
- **Local Code**: Your task app code and data stay on your machine; only HTTP traffic flows through Cloudflare.

### Example: Complete Workflow

```bash
# 1. Set up credentials
export SYNTH_API_KEY="sk-synth-abc123..."
export ENVIRONMENT_API_KEY="env-key-xyz789..."

# 2. Deploy with tunnel
uvx synth-ai deploy \
  --task-app examples/task_apps/banking77/task_app.py \
  --runtime tunnel \
  --tunnel-mode managed \
  --port 8000

# Output:
# ✓ Task app running on http://127.0.0.1:8000
# ✓ Tunnel connected: https://cust-def456.usesynth.ai
# ✓ Wrote TASK_APP_URL to .env

# 3. Verify tunnel is accessible
curl https://cust-def456.usesynth.ai/health \
  -H "X-API-Key: $ENVIRONMENT_API_KEY" \
  -H "CF-Access-Client-Id: $(grep CF_ACCESS_CLIENT_ID .env | cut -d= -f2)" \
  -H "CF-Access-Client-Secret: $(grep CF_ACCESS_CLIENT_SECRET .env | cut -d= -f2)"

# 4. Run training (uses tunnel URL automatically)
uvx synth-ai train \
  --config examples/blog_posts/mipro/configs/banking77_pipeline_mipro.toml

# Training config references ${TASK_APP_URL} which is now set to tunnel URL
```

### Key Benefits for Customers

1. **No Infrastructure Setup**: No need to deploy to Modal, AWS, or other cloud providers
2. **Data Stays Local**: Code and data remain on customer's machine (compliance-friendly)
3. **Simple Authentication**: Just `SYNTH_API_KEY` - no Cloudflare account needed
4. **Automatic Security**: Access protection configured automatically
5. **Cost-Effective**: Free for random subdomains, optional billing for custom subdomains
6. **Developer-Friendly**: Works with existing `synth-ai deploy` workflow

## References

- Cloudflare Tunnel API: https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/get-started/create-remote-tunnel-api/
- Quick Tunnels: https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/
- Access Service Tokens: https://developers.cloudflare.com/cloudflare-one/access-controls/service-credentials/service-tokens/
- Current deploy implementation: `synth_ai/cli/deploy.py`, `synth_ai/modal.py`, `synth_ai/uvicorn.py`
Below is a concrete, code‑level integration plan that (1) keeps user task apps *local* while your hosted optimizers reach them through Cloudflare Tunnel, (2) lets you provision named tunnels on customers’ behalf from your own Cloudflare account (so Synth owns the bill), and (3) makes RL, GEPA, and MIPRO all include Cloudflare Access headers whenever the task app is behind a tunnel.

---

## 0) What’s in Synth today (ground truth)

* `synth-ai deploy` currently supports **local (uvicorn)** and **Modal**; Modal is invoked by shelling the Modal CLI; local runs uvicorn. The docs show the exact options/flow and that the CLI loads `ENVIRONMENT_API_KEY` and does health checks. ([Synth AI][1])
* RL/GEPA/MIPRO training CLIs consume TOML configs and call the task app via `/rollout`; RL shows `--task-url` and uses `/rl/verify_task_app`, `/health`, `/task_info`, all authenticated with `ENVIRONMENT_API_KEY`. ([Synth AI][2])
* The **Environment API Key** contract is explicit: backend → task app uses `X-API-Key` (and `Authorization: Bearer`, optional), and task apps validate that header. ([Synth AI][3])

---

## 1) Cloudflare choices and costs (what we’ll rely on)

* **Tunnels are free** (Argo Tunnel was rebranded and the per‑tunnel charge was removed). You *can* add paid Argo Smart Routing, but it’s optional. ([The Cloudflare Blog][4])
* **Quick Tunnels** (`*.trycloudflare.com`) are ephemeral and require **no account**—perfect for dev. ([Cloudflare Docs][5])
* **Access Service Tokens** (Client ID/Secret) authenticate automated systems and **do not consume seats** in Zero Trust. ([Cloudflare Docs][6])
* **TLS on hostnames**: Universal SSL covers `example.com` and `*.example.com` (first‑level subdomains only). If you choose *deep* hostnames like `*.tunnels.example.com`, you’ll need **Advanced Certificate Manager (ACM)** at **$10/month per zone**. We’ll avoid that by using first‑level subdomains (`customer-123.usesynth.ai`). ([Cloudflare Docs][7])

---

## 2) High‑level design

We add a third runtime: **`--runtime tunnel`** with two modes:

* **quick** → spawn `cloudflared tunnel --url http://127.0.0.1:<port>`; parse the ephemeral `https://<rand>.trycloudflare.com` and write it to `.env` as `TASK_APP_URL`. (No Access; dev only.) ([Cloudflare Docs][5])
* **managed** → Synth backend provisions a **named tunnel** and **Access** on **your** Cloudflare account via API; SDK starts `cloudflared` with the returned **tunnel token**; backend also returns **Access Client ID/Secret** (Service Token). SDK writes:

  * `TASK_APP_URL=https://<customer-...>.usesynth.ai`
  * `CF_ACCESS_CLIENT_ID=...`
  * `CF_ACCESS_CLIENT_SECRET=...`
    Optimizers send these headers on **every** `/rollout` across RL, GEPA, MIPRO.

---

## 3) SDK changes (new modules + CLI flags)

> Explanation (one sentence): *These snippets add the tunnel runtime, spawn `cloudflared`, persist URL/Access secrets, and call your backend to mint a named tunnel and service token.*

```python
# synth_ai/cfgs.py  (add)
from dataclasses import dataclass
from typing import Literal, Optional
from pathlib import Path

@dataclass
class CloudflareTunnelDeployCfg:
    task_app_path: Path
    env_api_key: str
    host: str = "127.0.0.1"
    port: int = 8000
    mode: Literal["quick", "managed"] = "managed"
    subdomain: Optional[str] = None  # requested subdomain (managed)
    tunnel_token: Optional[str] = None  # filled in for managed
    trace: bool = True
```

```python
# synth_ai/tunnel.py  (new)
import os, re, shutil, subprocess, sys, time
from typing import Optional, Tuple

_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com", re.I)

def _which_cloudflared() -> str:
    p = shutil.which("cloudflared")
    if p:
        return p
    raise FileNotFoundError(
        "cloudflared not found. Install:\n"
        "  macOS: brew install cloudflare/cloudflare/cloudflared\n"
        "  Linux/Windows: https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/local-management/create-local-tunnel/"
    )

def open_quick_tunnel(port: int, wait_s: float = 10.0) -> Tuple[str, subprocess.Popen]:
    bin_path = _which_cloudflared()
    proc = subprocess.Popen(
        [bin_path, "tunnel", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    start = time.time()
    url = None
    # stream stdout to detect the trycloudflare URL
    while time.time() - start < wait_s:
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.05); continue
        m = _URL_RE.search(line)
        if m:
            url = m.group(0)
            break
    if not url:
        proc.terminate()
        raise RuntimeError("Failed to parse trycloudflare URL from cloudflared output.")
    return url, proc

def open_managed_tunnel(tunnel_token: str) -> subprocess.Popen:
    bin_path = _which_cloudflared()
    # cloudflared v2023.4+ accepts --token for named tunnels
    return subprocess.Popen(
        [bin_path, "tunnel", "run", "--token", tunnel_token],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

def stop_tunnel(proc: Optional[subprocess.Popen]) -> None:
    if proc and proc.poll() is None:
        proc.terminate()
```

```python
# synth_ai/api/tunnel.py  (new)
import os, httpx
from typing import Optional, Dict, Any

BACKEND_BASE_URL = os.getenv("SYNTH_BACKEND_BASE_URL", "https://api.usesynth.ai")

async def create_tunnel(synth_api_key: str, port: int, subdomain: Optional[str]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            f"{BACKEND_BASE_URL}/api/v1/tunnels",
            headers={"Authorization": f"Bearer {synth_api_key}"},
            json={"port": port, "subdomain": subdomain, "mode": "managed", "enable_access": True},
        )
        r.raise_for_status()
        return r.json()
```

```python
# synth_ai/utils/tunnel_config.py  (new)
from pathlib import Path
from typing import Optional
from synth_ai.utils.env import write_env_var_to_dotenv

def store_tunnel_credentials(
    tunnel_url: str,
    access_client_id: Optional[str],
    access_client_secret: Optional[str],
    env_file: Optional[Path] = None,
):
    write_env_var_to_dotenv("TASK_APP_URL", tunnel_url, output_file_path=env_file)
    if access_client_id:
        write_env_var_to_dotenv("CF_ACCESS_CLIENT_ID", access_client_id, output_file_path=env_file)
    if access_client_secret:
        write_env_var_to_dotenv("CF_ACCESS_CLIENT_SECRET", access_client_secret, output_file_path=env_file, mask_msg=True)
```

```python
# synth_ai/tunnel_deploy.py  (new)
import asyncio, os
from typing import Optional
from synth_ai.cfgs import CloudflareTunnelDeployCfg
from synth_ai.tunnel import open_quick_tunnel, open_managed_tunnel
from synth_ai.utils.env import write_env_var_to_dotenv, resolve_env_var
from synth_ai.utils.tunnel_config import store_tunnel_credentials
from synth_ai.uvicorn import deploy_uvicorn_app  # reuse existing helper
from synth_ai.api.tunnel import create_tunnel

async def deploy_app_tunnel(cfg: CloudflareTunnelDeployCfg, env_file: Optional[str] = None) -> str:
    # start local uvicorn (background) using existing deploy helper
    await deploy_uvicorn_app(task_app_path=cfg.task_app_path, host=cfg.host, port=cfg.port, trace=cfg.trace, background=True)

    if cfg.mode == "quick":
        url, proc = open_quick_tunnel(cfg.port)
        store_tunnel_credentials(url, None, None, Path(env_file) if env_file else None)
        return url

    # managed: provision via Synth backend
    synth_api_key = resolve_env_var("SYNTH_API_KEY")
    data = await create_tunnel(synth_api_key, cfg.port, cfg.subdomain)
    token = data["tunnel_token"]
    hostname = data["hostname"]
    access_id = data.get("access_client_id")
    access_secret = data.get("access_client_secret")

    proc = open_managed_tunnel(token)
    url = f"https://{hostname}"
    store_tunnel_credentials(url, access_id, access_secret, Path(env_file) if env_file else None)
    return url
```

```python
# synth_ai/cli/deploy.py  (patch)
import click
from synth_ai.cfgs import CloudflareTunnelDeployCfg
from synth_ai.tunnel_deploy import deploy_app_tunnel

@click.option("--tunnel-mode", "tunnel_mode", type=click.Choice(["quick","managed"]), default="managed")
@click.option("--tunnel-subdomain", "tunnel_subdomain", type=str, default=None)
def deploy(..., runtime, ..., tunnel_mode, tunnel_subdomain, ...):
    ...
    if runtime == "tunnel":
        cfg = CloudflareTunnelDeployCfg(
            task_app_path=task_app,
            env_api_key=env_api_key,
            host=host, port=port, mode=tunnel_mode, subdomain=tunnel_subdomain, trace=trace,
        )
        url = asyncio.run(deploy_app_tunnel(cfg, env_file))
        click.secho(f"✓ Tunnel ready: {url}", fg="green")
        return
```

> Also add a tiny helper so `synth-ai smoke` and CLI health checks automatically include Access headers when present (see §5).

---

## 4) Backend: provision named tunnels on customers’ behalf (Synth pays)

> Explanation (one sentence): *This creates a named tunnel via Cloudflare API, binds DNS, adds Access (service token), and returns the `tunnel_token` + Access credentials back to the SDK.*

```python
# backend/app/api/v1/routes_tunnels.py  (new)
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from .auth import require_org  # your existing auth -> org_id
from app.core.cloudflare_api import provision_tunnel, revoke_tunnel

class CreateTunnelReq(BaseModel):
    port: int
    subdomain: str | None = None
    mode: str = "managed"
    enable_access: bool = True

class CreateTunnelResp(BaseModel):
    tunnel_id: str
    tunnel_token: str
    hostname: str
    access_client_id: str | None = None
    access_client_secret: str | None = None

router = APIRouter()

@router.post("/v1/tunnels", response_model=CreateTunnelResp)
async def create_tunnel(req: CreateTunnelReq, org = Depends(require_org)):
    # generate first-level hostname to avoid ACM cost
    hostname = await generate_hostname(org.id, requested=req.subdomain)  # e.g., cust-<hex>.usesynth.ai
    res = await provision_tunnel(hostname=hostname, local_port=req.port, enable_access=req.enable_access)
    # persist mapping (org_id, tunnel_id, hostname, encrypted secrets, etc.)
    await save_tunnel(org.id, hostname, res["tunnel_id"], res["tunnel_token"], res.get("access_client_id"), res.get("access_client_secret"))
    return CreateTunnelResp(
        tunnel_id=res["tunnel_id"], tunnel_token=res["tunnel_token"],
        hostname=hostname, access_client_id=res.get("access_client_id"), access_client_secret=res.get("access_client_secret")
    )
```

```python
# backend/app/core/cloudflare_api.py  (new)
import os, httpx

CF_API = "https://api.cloudflare.com/client/v4"
CF_ACCOUNT = os.environ["CF_ACCOUNT_ID"]
CF_ZONE = os.environ["CF_ZONE_ID"]
CF_TOKEN = os.environ["CF_API_TOKEN"]

async def provision_tunnel(hostname: str, local_port: int, enable_access: bool = True):
    headers = {"Authorization": f"Bearer {CF_TOKEN}"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1) create named tunnel
        r = await client.post(f"{CF_API}/accounts/{CF_ACCOUNT}/cfd_tunnel", headers=headers, json={"name": f"synth-{hostname}", "config_src": "cloudflare"})
        r.raise_for_status()
        tunnel = r.json()["result"]; tunnel_id, tunnel_token = tunnel["id"], tunnel["token"]

        # 2) bind ingress (hostname -> http://localhost:<port>)
        r = await client.put(f"{CF_API}/accounts/{CF_ACCOUNT}/cfd_tunnel/{tunnel_id}/configurations", headers=headers,
                             json={"config":{"ingress":[{"hostname":hostname,"service":f"http://localhost:{local_port}"},
                                                        {"service":"http_status:404"}]}})
        r.raise_for_status()

        # 3) DNS CNAME to <tunnel_id>.cfargotunnel.com
        sub = hostname.split('.')[0]
        r = await client.post(f"{CF_API}/zones/{CF_ZONE}/dns_records", headers=headers,
                              json={"type":"CNAME","name":sub,"content":f"{tunnel_id}.cfargotunnel.com","proxied":True})
        r.raise_for_status()

        access_id = access_secret = None
        if enable_access:
            # 4) Access self-hosted app + policy (Service Token only) + service token
            app = await client.post(f"{CF_API}/accounts/{CF_ACCOUNT}/access/apps", headers=headers,
                                    json={"name": f"synth-tunnel-{tunnel_id}", "domain": hostname, "type":"self_hosted"})
            app.raise_for_status(); app_id = app.json()["result"]["id"]
            pol = await client.post(f"{CF_API}/accounts/{CF_ACCOUNT}/access/apps/{app_id}/policies", headers=headers,
                                    json={"name":"Service Token Only","decision":"allow","include":[{"service_token":{}}]})
            pol.raise_for_status()
            tok = await client.post(f"{CF_API}/accounts/{CF_ACCOUNT}/access/service_tokens", headers=headers,
                                    json={"name": f"synth-optimizer-{tunnel_id}", "duration":"8760h"})
            tok.raise_for_status()
            token = tok.json()["result"]; access_id, access_secret = token["client_id"], token["client_secret"]

        return {"tunnel_id": tunnel_id, "tunnel_token": tunnel_token, "access_client_id": access_id, "access_client_secret": access_secret}
```

**Why this shape?** Because Cloudflare’s API supports creating a tunnel, attaching ingress, and wiring DNS; Access Service Tokens (Client ID/Secret) are the right primitive for non‑interactive services and do not consume seats. ([Cloudflare Docs][8])

> **DNS/TLS note**: Use **first‑level hostnames** like `cust‑abc123.usesynth.ai` so Universal SSL covers them at $0; avoid `*.tunnels.usesynth.ai` unless you want to pay **ACM $10/month/zone**. ([Cloudflare Docs][7])

---

## 5) Optimizer integration (RL, GEPA, MIPRO all include Access headers)

> Explanation (one sentence): *One shared helper builds headers for `/rollout`, and all callsites import it so tunnels “just work.”*

```python
# backend/app/routes/shared/task_app_client.py  (new)
import os
from typing import Dict, Any
def build_task_app_headers(task_app_api_key: str | None) -> Dict[str,str]:
    headers = {"Content-Type":"application/json"}
    api_key = (task_app_api_key or os.getenv("ENVIRONMENT_API_KEY") or "").strip()
    if api_key:
        headers["X-API-Key"] = api_key
        headers["Authorization"] = f"Bearer {api_key}"
    # Cloudflare Access (Service Token) – added automatically when tunnel-protected
    cid = os.getenv("CF_ACCESS_CLIENT_ID"); csec = os.getenv("CF_ACCESS_CLIENT_SECRET")
    if cid and csec:
        headers["CF-Access-Client-Id"] = cid
        headers["CF-Access-Client-Secret"] = csec
    return headers
```

```python
# RL trainer rollout callsites: import and use build_task_app_headers(...)
# e.g., clustered_trainer.py and evaluator path(s)
from app.routes.shared.task_app_client import build_task_app_headers

headers = build_task_app_headers(api_key)
async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
    resp = await client.post(f"{task_url.rstrip('/')}/rollout", json=payload, headers=headers)
```

```python
# GEPA and MIPRO callsites (evaluation/validation helpers): same import/use
from app.routes.shared.task_app_client import build_task_app_headers

headers = build_task_app_headers(task_app_api_key)
rollout_url = f"{task_app_url.rstrip('/')}/rollout"
async with httpx.AsyncClient(timeout=timeout) as client:
    resp = await client.post(rollout_url, json=payload, headers=headers)
```

**Result:** All three training types (RL, GEPA, MIPRO) automatically send `CF-Access-Client-Id`/`CF-Access-Client-Secret` *when present*—no per‑algorithm code. (Access tokens are the recommended non‑interactive mechanism for Access‑protected apps.) ([Cloudflare Docs][6])

---

## 6) CLI health checks & smoke tests behind Access

Augment the CLI’s existing health verification and `synth-ai smoke` so they *also* include Access headers if present. (Docs show current smoke/health behavior; this is additive.) ([Synth AI][9])

```python
# synth_ai/utils/http.py  (new)
import os
def default_task_app_headers(api_key: str | None) -> dict:
    from app.routes.shared.task_app_client import build_task_app_headers  # reuse exactly
    return build_task_app_headers(api_key)

# wherever CLI does GET /health or /task_info today:
headers = default_task_app_headers(env_api_key)
r = await client.get(f"{url}/health", headers=headers, timeout=10.0)
```

---

## 7) Billing model (passthrough, if/when you want it)

* **Tunnels:** $0 (no per‑tunnel fees). ([The Cloudflare Blog][4])
* **Access Service Tokens:** $0 seats consumed (service tokens are seatless). ([Cloudflare Docs][10])
* **TLS:** If you insist on **deep** hostnames (`*.tunnels.usesynth.ai`), you’ll need **ACM ($10/month/zone)**; otherwise, use **first‑level subdomains** (`customer-123.usesynth.ai`) and pay $0 for TLS. ([Cloudflare Docs][7])

**Recommendation:** Default to first‑level hostnames → zero recurring Cloudflare cost; later, if a customer wants a per‑org deep wildcard, add a monthly line item equal to ACM for the affected zone.

---

## 8) Tests

* **Unit (SDK):** mock `subprocess.Popen` and stdout parsing (Quick); token exec (Managed); robust error messages when `cloudflared` missing.
* **Integration (SDK):** spin a trivial FastAPI task app → Quick tunnel → GET `/health`; then Managed (with a mocked backend).
* **Optimizer (Backend):** assert `build_task_app_headers` injects `CF-Access-*` when env vars exist, and only `X-API-Key` when they don’t.
* **End‑to‑end:** RL, GEPA, and MIPRO configs targeting `TASK_APP_URL` (tunnel) succeed end‑to‑end with Access enforced (401 without Access headers).

---

## 9) Developer UX

* **Dev:** `uvx synth-ai deploy --runtime tunnel --tunnel-mode quick --port 8000` → writes `TASK_APP_URL=https://<rand>.trycloudflare.com` to `.env`. ([Cloudflare Docs][5])
* **Prod‑ish (Synth‑managed):** `uvx synth-ai deploy --runtime tunnel --tunnel-mode managed --tunnel-subdomain acme` → backend mints named tunnel + Access token → SDK starts `cloudflared` with `--token` and writes `TASK_APP_URL=https://acme.usesynth.ai`, `CF_ACCESS_CLIENT_ID/SECRET` to `.env`.
* **Training:** `uvx synth-ai train --type rl|gepa|mipro --config ...` → all `/rollout` calls include Access headers automatically via the shared helper.

---

## 10) Security notes

* Access policy for each tunnel should be **Service Token only**, so only Synth’s optimizer hits the origin. (Cloudflare’s “Service Auth” rule type is the fit.) ([Cloudflare Docs][11])
* Keep `tunnel_token` and Access **Client Secret** encrypted-at-rest; display once, mask in logs.
* Universal SSL suffices when using first‑level hostnames; if you ever use second/third‑level labels, budget for ACM. ([Cloudflare Docs][7])

---

## 11) References you’ll need while implementing

* **Deploy command shapes today (Local/Modal):** shows exactly how your CLI currently works and where to add a third runtime. ([Synth AI][1])
* **Prompt Optimization/RL docs:** confirm `/rollout` contract and task app auth flow. ([Synth AI][12])
* **Cloudflare API – create remote tunnel & ingress, DNS, Access service tokens:** the exact API families we use above. ([Cloudflare Docs][8])
* **Pricing/limits:** Tunnels free; ACM $10/zone; Access tokens seatless. ([The Cloudflare Blog][4])

---

### TL;DR wiring

1. **Add `--runtime tunnel`** to CLI → **Quick** or **Managed**.
2. **Managed**: backend calls Cloudflare API to create **named tunnel + DNS + Access (Service Token)** and returns **tunnel_token + Access creds**. ([Cloudflare Docs][8])
3. SDK **starts `cloudflared`** with `--token`, writes `TASK_APP_URL` and **Access creds** to `.env`.
4. **One shared header helper** makes RL/GEPA/MIPRO always send `CF‑Access‑Client‑Id/Secret` when present.
5. **Use first‑level hostnames** to avoid ACM cost; tunnels & service tokens remain free. ([Cloudflare Docs][7])

If you want, I can convert this into PR‑ready diffs (SDK + backend) in a single patchset next.

[1]: https://docs.usesynth.ai/cli/deploy "Deploy Task Apps - Synth AI"
[2]: https://docs.usesynth.ai/rl/train "Train - Synth AI"
[3]: https://docs.usesynth.ai/sdk/environment-api-key "Environment API Key - Synth AI"
[4]: https://blog.cloudflare.com/tunnel-for-everyone/?utm_source=chatgpt.com "A Boring Announcement: Free Tunnels for Everyone"
[5]: https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/?utm_source=chatgpt.com "Quick Tunnels · Cloudflare One docs"
[6]: https://developers.cloudflare.com/cloudflare-one/access-controls/service-credentials/service-tokens/?utm_source=chatgpt.com "Service tokens · Cloudflare One docs"
[7]: https://developers.cloudflare.com/ssl/edge-certificates/universal-ssl/limitations/?utm_source=chatgpt.com "Limitations for Universal SSL"
[8]: https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/get-started/create-remote-tunnel-api/?utm_source=chatgpt.com "Create a tunnel (API) · Cloudflare One docs"
[9]: https://docs.usesynth.ai/cli/smoke "Smoke - Synth AI"
[10]: https://developers.cloudflare.com/cloudflare-one/team-and-resources/users/seat-management/?utm_source=chatgpt.com "Seat management - Cloudflare One"
[11]: https://developers.cloudflare.com/cloudflare-one/access-controls/policies/?utm_source=chatgpt.com "Access policies · Cloudflare One docs"
[12]: https://docs.usesynth.ai/po/train "Train - Synth AI"
