from __future__ import annotations

import os
from pathlib import Path


class SSLConfig:
    """Centralize SSL verification logic for all HTTP clients."""

    @staticmethod
    def get_verify_setting() -> bool | str:
        """Return the correct verify setting for requests/httpx clients."""
        skip_flag = os.getenv("SYNTH_SKIP_TASK_APP_HEALTH_CHECK", "").strip().lower()
        if skip_flag in {"1", "true", "yes"}:
            return False

        ca_bundle = os.getenv("REQUESTS_CA_BUNDLE") or os.getenv("SSL_CERT_FILE")
        if ca_bundle and Path(ca_bundle).expanduser().is_file():
            return str(Path(ca_bundle).expanduser())

        mitm_ca = Path.home() / ".mitmproxy" / "mitmproxy-ca-cert.pem"
        if mitm_ca.is_file() and os.getenv("HTTPS_PROXY"):
            return str(mitm_ca)

        return True
