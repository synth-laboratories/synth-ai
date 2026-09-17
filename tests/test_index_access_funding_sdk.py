from synth_ai.sdk.index.catalog import BillingPolicyUpdate
from synth_ai.sdk.index.client import AccountAPI


def test_access_funding_read_and_policy_update_use_declared_routes():
    calls = []

    def run(call):
        calls.append(call)
        if call.operation_id == "index.me.access_funding":
            return call.parse(
                {
                    "org_id": "org-proof",
                    "can_manage_policy": True,
                    "modes": [],
                    "deep_beta": None,
                    "live_wallet_holds_microcents": 0,
                    "wallet_available_microcents": 0,
                    "generated_at": "2026-09-17T00:00:00Z",
                }
            )
        return {"revision": 1}

    api = AccountAPI(run, False)
    account = api.access_funding()
    assert account.org_id == "org-proof"
    result = api.update_billing_policy(
        "deep",
        BillingPolicyUpdate(
            wallet_enabled=True,
            monthly_cap_cents=500,
            concurrency_limit=1,
            consent_terms_version="index-wallet-v1",
        ),
    )
    assert result == {"revision": 1}
    assert calls[0].operation_id == "index.me.access_funding"
    assert calls[1].operation_id == "index.me.access_funding.update"
    assert calls[1].path_parameters == {"mode": "deep"}
