"""Complete a Stripe TEST-mode hosted Checkout page for the J2 buy_credits step.

Hosted J2 needs `checkout.session.completed` delivered to Core before
`getCheckout` can read `completed`. This drives Stripe's hosted page with a
headless Chromium (Playwright): test card 4242, Link "Save my information"
unticked, submit, wait for the success redirect. It refuses any page that is
not a Stripe test-mode session, so it can never pay in live mode.

The URL is a bearer link to the session; it is never printed or recorded.
Returns None when Stripe redirected to the success URL, else a typed cause.
"""

from __future__ import annotations

# ruff: noqa: UP045 -- Explicit Optional at acceptance contract boundaries.
import re
from typing import TYPE_CHECKING, Optional
from urllib.parse import urlsplit

if TYPE_CHECKING:  # Playwright is an acceptance-host install, not an SDK dependency.
    from playwright.sync_api import Locator, Page

__all__ = ["CHECKOUT_PAGE_TIMEOUT_MS", "complete_test_checkout", "test_mode_url"]

CHECKOUT_PAGE_TIMEOUT_MS = 60_000
FIELD_WAIT_MS = 20_000
TEST_CARD = "4242424242424242"
TEST_EMAIL = "j2-acceptance@example.com"
EMAIL = ("#linkAuthentication-emailInput", "#email", "input[name='email']")
CARD_METHOD = (
    "[data-testid='card-accordion-item-button']",
    "#payment-method-accordion-item-title-card",
    "input[value='card']",
)
NUMBER = ("#cardNumber", "input[name='cardNumber']", "input[name='number']")
EXPIRY = ("#cardExpiry", "input[name='cardExpiry']", "input[name='expiry']")
CVC = ("#cardCvc", "input[name='cardCvc']", "input[name='cvc']")
NAME = ("#billingName", "input[name='billingName']", "input[autocomplete='cc-name']")
POSTAL = (
    "#billingPostalCode",
    "input[name='billingPostalCode']",
    "input[autocomplete='postal-code']",
)
SUBMIT = ("[data-testid='hosted-payment-submit-button']", ".SubmitButton", "button[type='submit']")
SAVE_INFO = "Save my information for faster checkout"


def test_mode_url(url: str) -> bool:
    """True only for an https Stripe-hosted page of a test-mode session."""
    parts = urlsplit(url)
    return (
        parts.scheme == "https"
        and parts.hostname == "checkout.stripe.com"
        and "/cs_test_" in parts.path
    )


def complete_test_checkout(url: str) -> Optional[str]:
    if not test_mode_url(url):
        return "stripe_checkout_not_test_mode"
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError:
        return "stripe_test_completer_unavailable:playwright_absent"
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                return _pay(browser.new_page(viewport={"width": 1280, "height": 1400}), url)
            finally:
                browser.close()
    except PlaywrightError as error:
        # Playwright messages can quote the session URL; record the type only.
        return "stripe_test_checkout_browser_error:" + type(error).__name__


def _find(page: Page, selectors: tuple[str, ...]) -> Optional[Locator]:
    for _ in range(FIELD_WAIT_MS // 500):
        for frame in page.frames:
            for selector in selectors:
                locator = frame.locator(selector)
                if locator.count() and locator.first.is_visible():
                    return locator.first
        page.wait_for_timeout(500)
    return None


def _fill(page: Page, selectors: tuple[str, ...], value: str, required: bool = True) -> bool:
    field = _find(page, selectors)
    if field is None:
        return not required
    field.click()
    field.fill(value)
    return True


def _pay(page: Page, url: str) -> Optional[str]:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    page.goto(url, wait_until="load", timeout=CHECKOUT_PAGE_TIMEOUT_MS)
    if not _fill(page, EMAIL, TEST_EMAIL):
        return "stripe_test_checkout_field_absent:email"
    page.keyboard.press("Tab")
    method = _find(page, CARD_METHOD)
    if method is not None:
        method.click(force=True)
    for name, selectors, value in (
        ("card_number", NUMBER, TEST_CARD),
        ("card_expiry", EXPIRY, "12 / 34"),
        ("card_cvc", CVC, "123"),
    ):
        if not _fill(page, selectors, value):
            return "stripe_test_checkout_field_absent:" + name
    _fill(page, NAME, "J2 Acceptance", required=False)
    _fill(page, POSTAL, "94103", required=False)
    # Link's "Save my information" is ticked by default and then demands a phone.
    for frame in page.frames:
        if frame.locator("input[type='checkbox']:checked").count():
            label = frame.get_by_text(SAVE_INFO)
            if label.count():
                label.first.click()
                break
    submit = _find(page, SUBMIT)
    if submit is None:
        return "stripe_test_checkout_field_absent:submit"
    submit.click()
    try:
        page.wait_for_url(re.compile(r"checkout=success"), timeout=CHECKOUT_PAGE_TIMEOUT_MS)
    except PlaywrightTimeoutError:
        return "stripe_test_checkout_not_redirected"
    return None
