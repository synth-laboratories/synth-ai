"""SMR promotion registry SDK namespace."""

from __future__ import annotations

from collections.abc import Mapping

from synth_ai.core.research._legacy.models.promotions import (
    SmrOrgPromotionView,
    SmrPromotionCampaignPublic,
    SmrPromotionMineResponse,
    SmrPromotionPublicCatalog,
    SmrPromotionRetireResult,
    SmrPromotionUpsertRequest,
)
from synth_ai.core.research._legacy.sdk._base import _ClientNamespace


class PromotionsAPI(_ClientNamespace):
    """Public catalog, org-scoped eligibility, claim, and admin campaign helpers."""

    def list_public(self, *, include_scheduled: bool = False) -> SmrPromotionPublicCatalog:
        params = {"include_scheduled": "true"} if include_scheduled else None
        return SmrPromotionPublicCatalog.from_wire(
            self._client._request_json("GET", "/smr/promotions", params=params)
        )

    def mine(self) -> SmrPromotionMineResponse:
        return SmrPromotionMineResponse.from_wire(
            self._client._request_json("GET", "/smr/promotions/mine")
        )

    def claim(self, campaign_id: str) -> SmrOrgPromotionView:
        return SmrOrgPromotionView.from_wire(
            self._client._request_json(
                "POST",
                f"/smr/promotions/{campaign_id}/claim",
            )
        )

    def list_admin_campaigns(self) -> SmrPromotionPublicCatalog:
        return SmrPromotionPublicCatalog.from_wire(
            self._client._request_json("GET", "/smr/promotions/admin/campaigns")
        )

    def upsert_admin_campaign(
        self,
        request: SmrPromotionUpsertRequest | Mapping[str, object],
    ) -> SmrPromotionCampaignPublic:
        payload = (
            request.to_wire() if isinstance(request, SmrPromotionUpsertRequest) else dict(request)
        )
        return SmrPromotionCampaignPublic.from_wire(
            self._client._request_json(
                "POST",
                "/smr/promotions/admin/campaigns",
                json_body=payload,
            )
        )

    def retire_admin_campaign(self, campaign_id: str) -> SmrPromotionRetireResult:
        return SmrPromotionRetireResult.from_wire(
            self._client._request_json(
                "POST",
                f"/smr/promotions/admin/campaigns/{campaign_id}/retire",
            )
        )


__all__ = ["PromotionsAPI"]
