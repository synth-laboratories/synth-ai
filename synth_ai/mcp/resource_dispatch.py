"""Existing public resource operations; names remain owned by the Core catalog."""

from collections.abc import Mapping

from pydantic import BaseModel, JsonValue

from .._generated.models import (
    AutoReloadRequest,
    CheckoutRequest,
    MintApiKeyRequest,
    SetSpendAlertsRequest,
    SpecialFreeRequestInput,
    SubscriptionChangeRequest,
    TermsAcceptanceRequest,
    UpdateApiKeyRequest,
    UploadCreateRequest,
    UsageExportRequest,
)
from ..aclient import AsyncSynthClient
from ..errors import ResponseContractError, local_problem
from ._arguments import optional, page_limit, required

__all__ = ["dispatch_resource"]


async def dispatch_resource(
    client: AsyncSynthClient, operation: str, arguments: Mapping[str, JsonValue]
) -> BaseModel | None:
    match operation:
        case "acceptTerms":
            terms = required(arguments, "body", TermsAcceptanceRequest)
            return await client.account.accept_terms(
                version=terms.version,
                accepted_at=terms.accepted_at,
                idempotency_key=optional(arguments, "idempotency_key", str),
            )
        case "getUsageSummary":
            return await client.account.usage_summary(
                since=optional(arguments, "since", str), until=optional(arguments, "until", str)
            )
        case "listLimits":
            return await client.account.limits()
        case "requestSpecialFree":
            request = required(arguments, "body", SpecialFreeRequestInput)
            return await client.account.request_special_free(
                use_case=request.use_case,
                links=request.links or [],
                idempotency_key=optional(arguments, "idempotency_key", str),
            )
        case "listPlanRequests":
            return await client.account.plan_requests()
        case "revokeOAuthGrant":
            return await client.account.revoke_oauth_grant(required(arguments, "grant_id", str))
        case "withdrawPlanRequest":
            return await client.account.withdraw_plan_request(
                required(arguments, "request_id", str)
            )
        case "createUsageExport":
            body = required(arguments, "body", UsageExportRequest)
            return await client.account.export_usage(
                since=body.since,
                until=body.until,
                idempotency_key=optional(arguments, "idempotency_key", str),
            )
        case "getUsageExport":
            return await client.account.usage_export(required(arguments, "export_id", str))
        case "listApiKeys":
            return await client.account.api_keys.list(
                purpose=optional(arguments, "purpose", str),
                cursor=optional(arguments, "cursor", str),
                limit=page_limit(arguments),
            )
        case "createApiKey":
            key = required(arguments, "body", MintApiKeyRequest)
            return await client.account.api_keys.mint(
                purpose=key.purpose,
                name=key.name,
                scopes=key.scopes,
                expires_at=key.expires_at,
                request_id=key.request_id,
                idempotency_key=optional(arguments, "idempotency_key", str),
            )
        case "updateApiKey":
            grants = required(arguments, "body", UpdateApiKeyRequest)
            return await client.account.api_keys.update_grants(
                required(arguments, "key_id", str),
                scopes=grants.scopes,
                expected_version=grants.expected_version,
                name=grants.name,
            )
        case "rotateApiKey":
            return await client.account.api_keys.rotate(
                required(arguments, "key_id", str),
                idempotency_key=optional(arguments, "idempotency_key", str),
            )
        case "revokeApiKey":
            return await client.account.api_keys.revoke(required(arguments, "key_id", str))
        case "listFiles":
            return await client.swarms.files(
                cursor=optional(arguments, "cursor", str), limit=page_limit(arguments)
            )
        case "getFile":
            return await client.swarms.file(required(arguments, "file_id", str))
        case "createUpload":
            upload = required(arguments, "body", UploadCreateRequest)
            return await client.swarms.prepare_upload(
                name=upload.name,
                media_type=upload.media_type,
                size_bytes=upload.size_bytes,
                digest_sha256=upload.digest_sha256,
                idempotency_key=optional(arguments, "idempotency_key", str),
            )
        case "getUpload":
            return await client.swarms.upload(required(arguments, "upload_id", str))
        case "finalizeUpload":
            return await client.swarms.finalize_upload(
                required(arguments, "upload_id", str),
                idempotency_key=optional(arguments, "idempotency_key", str),
            )
        case "getRunProgram":
            return await client.swarms.program(required(arguments, "run_id", str))
        case "getRunConfiguration":
            return await client.swarms.configuration(required(arguments, "run_id", str))
        case "listPlans":
            return await client.billing.plans()
        case "listBillingHistory":
            return await client.billing.history(
                cursor=optional(arguments, "cursor", str),
                limit=page_limit(arguments),
                cause=optional(arguments, "cause", str),
                run_id=optional(arguments, "run_id", str),
                since=optional(arguments, "since", str),
                until=optional(arguments, "until", str),
            )
        case "listStatements":
            return await client.billing.statements(
                months=optional(arguments, "months", int),
                cursor=optional(arguments, "cursor", str),
                limit=page_limit(arguments),
            )
        case "getCheckout":
            return await client.billing.checkout_status(required(arguments, "id", str))
        case "createCheckout":
            checkout = required(arguments, "body", CheckoutRequest)
            return await client.billing.checkout(
                plan_id=checkout.plan_id,
                pack_id=checkout.pack_id,
                credit=checkout.credit,
                idempotency_key=optional(arguments, "idempotency_key", str),
            )
        case "getSubscription":
            return await client.billing.subscription()
        case "cancelSubscription":
            return await client.billing.cancel_subscription(
                idempotency_key=optional(arguments, "idempotency_key", str)
            )
        case "resumeSubscription":
            return await client.billing.resume_subscription(
                idempotency_key=optional(arguments, "idempotency_key", str)
            )
        case "previewSubscriptionChange":
            return await client.billing.preview_subscription_change(
                required(arguments, "plan_id", str)
            )
        case "changeSubscriptionPlan":
            plan = required(arguments, "body", SubscriptionChangeRequest)
            return await client.billing.change_subscription_plan(
                plan.plan_id, idempotency_key=optional(arguments, "idempotency_key", str)
            )
        case "createPaymentMethodSession":
            return await client.billing.payment_method_session(
                idempotency_key=optional(arguments, "idempotency_key", str)
            )
        case "getAutoReload":
            return await client.billing.auto_reload()
        case "setAutoReload":
            reload = required(arguments, "body", AutoReloadRequest)
            return await client.billing.set_auto_reload(
                enabled=reload.enabled,
                threshold=reload.threshold,
                amount=reload.amount,
                disclosure_version=reload.disclosure_version,
                idempotency_key=optional(arguments, "idempotency_key", str),
            )
        case "getSpendAlerts":
            return await client.billing.spend_alerts()
        case "setSpendAlerts":
            return await client.billing.set_spend_alerts_request(
                required(arguments, "body", SetSpendAlertsRequest),
                idempotency_key=optional(arguments, "idempotency_key", str),
            )
        case _:
            raise ResponseContractError(
                local_problem("mcp_operation_unbound", "mcp_dispatch", detail=operation)
            )
