"""``client.research.account`` — org account, billing, usage, and BYOK reads.

These backend routes live under ``/api/v1`` (and ``/api/members``), not under
``/smr``, so this namespace calls them through the session client's internal
``_request_json`` layer, which applies the same base URL and
``Authorization: Bearer <api_key>`` header as every other SDK call.

Mutations are limited to BYOK key management and ``subscription.cancel``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from synth_ai.managed_research.models.factories import _optional_datetime
from synth_ai.managed_research.sdk.client import ManagedResearchClient


def _require_account_mapping(payload: object, *, label: str) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} payload must be an object")
    return dict(payload)


@dataclass(frozen=True)
class AccountBalance:
    """Typed ``BalanceResponse`` (org_id, balance_cents, balance_dollars)."""

    org_id: str
    balance_cents: int
    balance_dollars: float
    last_updated: datetime | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> AccountBalance:
        mapping = _require_account_mapping(payload, label="account balance")
        return cls(
            org_id=str(mapping.get("org_id") or ""),
            balance_cents=int(mapping.get("balance_cents") or 0),
            balance_dollars=float(mapping.get("balance_dollars") or 0.0),
            last_updated=_optional_datetime(mapping, "last_updated"),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class AccountTierLimits:
    """Typed ``TierLimitsResponse`` (tier name plus its limits payload)."""

    tier: str
    limits: dict[str, object] = field(default_factory=dict)
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> AccountTierLimits:
        mapping = _require_account_mapping(payload, label="tier limits")
        limits = mapping.get("limits")
        return cls(
            tier=str(mapping.get("tier") or ""),
            limits=dict(limits) if isinstance(limits, Mapping) else {},
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class AccountByokStatus:
    """Typed BYOK status (byok_enabled, plan_type, provider key coverage)."""

    byok_enabled: bool
    plan_type: str = ""
    providers_with_keys: dict[str, bool] = field(default_factory=dict)
    supported_providers: tuple[str, ...] = ()
    missing_providers: tuple[str, ...] = ()
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> AccountByokStatus:
        mapping = _require_account_mapping(payload, label="byok status")
        providers = mapping.get("providers_with_keys")
        return cls(
            byok_enabled=bool(mapping.get("byok_enabled")),
            plan_type=str(mapping.get("plan_type") or ""),
            providers_with_keys={
                str(key): bool(value)
                for key, value in dict(providers if isinstance(providers, Mapping) else {}).items()
            },
            supported_providers=tuple(
                str(item) for item in list(mapping.get("supported_providers") or [])
            ),
            missing_providers=tuple(
                str(item) for item in list(mapping.get("missing_providers") or [])
            ),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class AccountIdentity:
    """Typed ``MeResponse`` (org and user resolution for the API key)."""

    org_id: str
    user_id: str | None = None
    org_name: str | None = None
    user_email: str | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> AccountIdentity:
        mapping = _require_account_mapping(payload, label="account identity")

        def _opt_str(key: str) -> str | None:
            value = mapping.get(key)
            return str(value) if value is not None else None

        return cls(
            org_id=str(mapping.get("org_id") or ""),
            user_id=_opt_str("user_id"),
            org_name=_opt_str("org_name"),
            user_email=_opt_str("user_email"),
            raw=dict(mapping),
        )


class ResearchAccountBalanceAPI:
    """Org credit balance and usage summary."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def get(self) -> AccountBalance:
        """Read the current credit balance. Backend route: ``GET /api/v1/balance/current``."""
        payload = self._session._request_json("GET", "/api/v1/balance/current")
        return AccountBalance.from_wire(payload)

    def usage(self) -> dict[str, Any]:
        """Read the token/GPU usage summary. Backend route: ``GET /api/v1/balance/usage``."""
        return dict(
            _require_account_mapping(
                self._session._request_json("GET", "/api/v1/balance/usage"),
                label="balance usage",
            )
        )


class ResearchAccountCreditsAPI:
    """Credits ledger reads."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def transactions(
        self,
        *,
        cursor: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List purchase/usage transactions, most recent first.

        Backend route: ``GET /api/v1/credits/transactions`` (query ``cursor``, ``limit``).
        """
        params: dict[str, Any] = {"limit": int(limit)}
        if cursor is not None:
            params["cursor"] = str(cursor)
        return dict(
            _require_account_mapping(
                self._session._request_json(
                    "GET",
                    "/api/v1/credits/transactions",
                    params=params,
                ),
                label="credit transactions",
            )
        )


class ResearchAccountTiersAPI:
    """Default rate limits per subscription tier."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def list(self) -> tuple[AccountTierLimits, ...]:
        """List default limits for all tiers. Backend route: ``GET /api/v1/usage/tiers``."""
        payload = self._session._request_json("GET", "/api/v1/usage/tiers")
        if not isinstance(payload, list):
            raise ValueError("tiers payload must be a list")
        return tuple(AccountTierLimits.from_wire(item) for item in payload)

    def get(self, tier: str) -> AccountTierLimits:
        """Read default limits for one tier. Backend route: ``GET /api/v1/usage/tiers/{tier}``."""
        payload = self._session._request_json("GET", f"/api/v1/usage/tiers/{tier}")
        return AccountTierLimits.from_wire(payload)


class ResearchAccountByokAPI:
    """Bring-your-own-key provider key management."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def list(self) -> dict[str, Any]:
        """List stored provider key metadata. Backend route: ``GET /api/v1/byok/keys``."""
        return dict(
            _require_account_mapping(
                self._session._request_json("GET", "/api/v1/byok/keys"),
                label="byok keys",
            )
        )

    def create(self, *, provider: str, encrypted_key_b64: str) -> dict[str, Any]:
        """Store a provider key (ciphertext encrypted with ``crypto.public_key``).

        Backend route: ``POST /api/v1/byok/keys``.
        """
        return dict(
            _require_account_mapping(
                self._session._request_json(
                    "POST",
                    "/api/v1/byok/keys",
                    json_body={
                        "provider": str(provider),
                        "encrypted_key_b64": str(encrypted_key_b64),
                    },
                ),
                label="byok store key",
            )
        )

    def get(self, provider: str) -> dict[str, Any]:
        """Read stored key metadata for one provider.

        Backend route: ``GET /api/v1/byok/keys/{provider}``.
        """
        return dict(
            _require_account_mapping(
                self._session._request_json("GET", f"/api/v1/byok/keys/{provider}"),
                label="byok key",
            )
        )

    def delete(self, provider: str) -> dict[str, Any]:
        """Delete the stored key for one provider.

        Backend route: ``DELETE /api/v1/byok/keys/{provider}``.
        """
        return dict(
            _require_account_mapping(
                self._session._request_json("DELETE", f"/api/v1/byok/keys/{provider}"),
                label="byok delete key",
            )
        )

    def validate(self, provider: str) -> dict[str, Any]:
        """Validate the stored key against the provider's API.

        Backend route: ``POST /api/v1/byok/keys/{provider}/validate``.
        """
        return dict(
            _require_account_mapping(
                self._session._request_json(
                    "POST",
                    f"/api/v1/byok/keys/{provider}/validate",
                ),
                label="byok validate key",
            )
        )

    def status(self) -> AccountByokStatus:
        """Read BYOK enablement and provider key coverage.

        Backend route: ``GET /api/v1/byok/status``.
        """
        return AccountByokStatus.from_wire(
            self._session._request_json("GET", "/api/v1/byok/status")
        )


class ResearchAccountMembersAPI:
    """Org member listing."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def list(self, *, limit: int = 200, offset: int = 0) -> dict[str, Any]:
        """List org members with app-user details.

        Backend route: ``GET /api/members`` (mounted with the ``/api`` prefix,
        not ``/api/v1``; query ``limit``, ``offset``).
        """
        return dict(
            _require_account_mapping(
                self._session._request_json(
                    "GET",
                    "/api/members",
                    params={"limit": int(limit), "offset": int(offset)},
                ),
                label="org members",
            )
        )


class ResearchAccountSubscriptionAPI:
    """Subscription lifecycle (cancel only)."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def cancel(self) -> dict[str, Any]:
        """Cancel the subscription at the end of the current billing period.

        Backend route: ``POST /api/v1/subscription/cancel``.
        """
        return dict(
            _require_account_mapping(
                self._session._request_json("POST", "/api/v1/subscription/cancel"),
                label="subscription cancel",
            )
        )


class ResearchAccountCryptoAPI:
    """Public-key material for client-side key encryption."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def public_key(self) -> dict[str, Any]:
        """Read the app public key used to encrypt BYOK keys.

        Backend route: ``GET /api/v1/crypto/public-key``.
        """
        return dict(
            _require_account_mapping(
                self._session._request_json("GET", "/api/v1/crypto/public-key"),
                label="crypto public key",
            )
        )


class ResearchAccountAPI:
    """Org account namespace: balance, credits, usage, BYOK, members, identity."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session
        self._balance: ResearchAccountBalanceAPI | None = None
        self._credits: ResearchAccountCreditsAPI | None = None
        self._tiers: ResearchAccountTiersAPI | None = None
        self._byok: ResearchAccountByokAPI | None = None
        self._members: ResearchAccountMembersAPI | None = None
        self._subscription: ResearchAccountSubscriptionAPI | None = None
        self._crypto: ResearchAccountCryptoAPI | None = None

    @property
    def balance(self) -> ResearchAccountBalanceAPI:
        """Org credit balance and usage summary."""
        if self._balance is None:
            self._balance = ResearchAccountBalanceAPI(self._session)
        return self._balance

    @property
    def credits(self) -> ResearchAccountCreditsAPI:
        """Credits ledger transactions."""
        if self._credits is None:
            self._credits = ResearchAccountCreditsAPI(self._session)
        return self._credits

    @property
    def tiers(self) -> ResearchAccountTiersAPI:
        """Default per-tier rate limits."""
        if self._tiers is None:
            self._tiers = ResearchAccountTiersAPI(self._session)
        return self._tiers

    @property
    def byok(self) -> ResearchAccountByokAPI:
        """Provider key (BYOK) management."""
        if self._byok is None:
            self._byok = ResearchAccountByokAPI(self._session)
        return self._byok

    @property
    def members(self) -> ResearchAccountMembersAPI:
        """Org member listing."""
        if self._members is None:
            self._members = ResearchAccountMembersAPI(self._session)
        return self._members

    @property
    def subscription(self) -> ResearchAccountSubscriptionAPI:
        """Subscription lifecycle (cancel)."""
        if self._subscription is None:
            self._subscription = ResearchAccountSubscriptionAPI(self._session)
        return self._subscription

    @property
    def crypto(self) -> ResearchAccountCryptoAPI:
        """Public-key material for BYOK encryption."""
        if self._crypto is None:
            self._crypto = ResearchAccountCryptoAPI(self._session)
        return self._crypto

    def usage(self) -> dict[str, Any]:
        """Read current org usage counters. Backend route: ``GET /api/v1/usage``."""
        return dict(
            _require_account_mapping(
                self._session._request_json("GET", "/api/v1/usage"),
                label="usage",
            )
        )

    def user_limits(self) -> dict[str, Any]:
        """Read effective limits for the authenticated user.

        Backend route: ``GET /api/v1/usage/user-limits``.
        """
        return dict(
            _require_account_mapping(
                self._session._request_json("GET", "/api/v1/usage/user-limits"),
                label="user limits",
            )
        )

    def overview(
        self,
        *,
        days: int = 30,
        include_projects: bool = False,
    ) -> dict[str, Any]:
        """Read the entitlements + usage + spend overview.

        Backend route: ``GET /api/v1/usage/overview`` (query ``days`` 1-365,
        ``include_projects``).
        """
        return dict(
            _require_account_mapping(
                self._session._request_json(
                    "GET",
                    "/api/v1/usage/overview",
                    params={
                        "days": int(days),
                        "include_projects": bool(include_projects),
                    },
                ),
                label="usage overview",
            )
        )

    def me(self) -> AccountIdentity:
        """Resolve the org/user identity for the API key.

        Backend route: ``GET /api/v1/me``.
        """
        return AccountIdentity.from_wire(self._session._request_json("GET", "/api/v1/me"))


__all__ = [
    "AccountBalance",
    "AccountByokStatus",
    "AccountIdentity",
    "AccountTierLimits",
    "ResearchAccountAPI",
    "ResearchAccountBalanceAPI",
    "ResearchAccountByokAPI",
    "ResearchAccountCreditsAPI",
    "ResearchAccountCryptoAPI",
    "ResearchAccountMembersAPI",
    "ResearchAccountSubscriptionAPI",
    "ResearchAccountTiersAPI",
]
