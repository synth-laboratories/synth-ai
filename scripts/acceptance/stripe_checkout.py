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
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional
from urllib.parse import urlsplit

if TYPE_CHECKING:  # Playwright is an acceptance-host install, not an SDK dependency.
    from playwright.sync_api import Frame, Locator, Page

__all__ = [
    "CHECKOUT_PAGE_TIMEOUT_MS",
    "QUALIFIED_PLAYWRIGHT",
    "complete_test_checkout",
    "test_mode_url",
]

# Browser automation is qualified on this Playwright (and its Chromium headless
# shell); see scripts/acceptance/README.md. Declared, not enforced.
QUALIFIED_PLAYWRIGHT = "1.65.2"
CHECKOUT_PAGE_TIMEOUT_MS = 60_000
FIELD_WAIT_MS = 20_000
POLL_MS = 500
TEST_CARD = "4242424242424242"
TEST_EMAIL = "j2-acceptance@example.com"
# Stripe hosted Checkout, 2026-11 (observed on staging TEST sessions, SYN-5696):
# the Payment Element lives in a js.stripe.com `elements-inner-*` frame; card
# inputs render only after the "Card" accordion row (a DIV with role=button,
# not a <button>) is chosen; Link's opt-in is a checkbox in the same frame.
# The legacy page (#cardNumber, no accordion) stays accepted.
CARD_ROLE_NAME = re.compile(r"^Card\b")
EMAIL = ("#linkAuthentication-emailInput", "#email", "input[name='email']")
CARD_METHOD = (
    "[data-testid='card-accordion-item-button']",
    "#payment-method-accordion-item-title-card",
    "input[value='card']",
)
NUMBER = ("#payment-numberInput", "#cardNumber", "input[name='cardNumber']", "input[name='number']")
EXPIRY = ("#payment-expiryInput", "#cardExpiry", "input[name='cardExpiry']", "input[name='expiry']")
CVC = ("#payment-cvcInput", "#cardCvc", "input[name='cardCvc']", "input[name='cvc']")
NAME = (
    "#payment-nameInput",
    "#billingName",
    "input[name='billingName']",
    "input[autocomplete='cc-name']",
)
POSTAL = (
    "#payment-postalCodeInput",
    "#billingPostalCode",
    "input[name='billingPostalCode']",
    "input[autocomplete='postal-code']",
    "input[autocomplete='billing postal-code']",
)
LINK_OPT_IN = ("#payment-linkOptInInput", "#enableStripePass")
SUBMIT = (
    "[data-testid='payment-form-submit-button']",
    "[data-testid='hosted-payment-submit-button']",
    ".SubmitButton",
    "button[type='submit']",
)
SAVE_INFO = "Save my information for faster checkout"
# A challenge is a visible, sizeable hCaptcha iframe in the top page. The
# invisible hCaptcha frames Stripe always loads are not a challenge.
CAPTCHA_FRAME = "iframe[src*='hcaptcha']"
CAPTCHA_MIN_PX = 100


def test_mode_url(url: str) -> bool:
    """True only for an https Stripe-hosted page of a test-mode session."""
    parts = urlsplit(url)
    return (
        parts.scheme == "https"
        and parts.hostname == "checkout.stripe.com"
        and "/cs_test_" in parts.path
    )


class Progress:
    """The step _pay is on and the page it is on, for a secret-free failure cause."""

    def __init__(self) -> None:
        self.stage = "launch"
        self.page: Optional[Page] = None

    def at(self, stage: str) -> None:
        self.stage = stage


def page_location(url: str) -> str:
    """Host + path only; the session id is redacted (the URL is a bearer link)."""
    parts = urlsplit(url)
    path = re.sub(r"/cs_(test|live)_[A-Za-z0-9]+", r"/cs_\1_<redacted>", parts.path)
    return (parts.hostname or "none") + (path or "/")


def failure_kind(error: BaseException) -> str:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    if isinstance(error, PlaywrightTimeoutError):
        return "timeout"
    message = str(error)  # Classified only, never retained: it can quote the URL.
    if "net::" in message or "navigat" in message.lower():
        return "navigation"
    if "closed" in message.lower() or "detached" in message.lower():
        return "target_closed"
    return "error"


def browser_error_cause(error: BaseException, progress: Progress) -> str:
    """stripe_test_checkout_browser_error:<kind>:<stage>:<host/path> -- no query, no message."""
    location = "none"
    if progress.page is not None:
        try:
            location = page_location(progress.page.url)
        except Exception:  # A dead page has no readable URL.
            location = "unreadable"
    return ":".join(
        ("stripe_test_checkout_browser_error", failure_kind(error), progress.stage, location)
    )


def complete_test_checkout(url: str) -> Optional[str]:
    if not test_mode_url(url):
        return "stripe_checkout_not_test_mode"
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError:
        return "stripe_test_completer_unavailable:playwright_absent"
    progress = Progress()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": 1280, "height": 1400})
                progress.page = page
                return _pay(page, url, progress)
            finally:
                browser.close()
    except PlaywrightError as error:
        return browser_error_cause(error, progress)


def _visible(locator: Locator) -> Optional[Locator]:
    for index in range(locator.count()):
        candidate = locator.nth(index)
        if candidate.is_visible():
            return candidate
    return None


def _detached(error: BaseException) -> bool:
    """Stripe reloads its frames while the page settles; a frame can detach mid-walk."""
    return failure_kind(error) == "target_closed"


def _in_frames(page: Page, probe: Callable[[Frame], Optional[Locator]]) -> Optional[Locator]:
    """One walk over a fresh snapshot of the page's frames.

    A frame that detaches while it is probed is skipped; the caller's next poll
    re-enumerates the frames, so the walk never outlives the caller's timeout.
    Any other browser error still propagates.
    """
    from playwright.sync_api import Error as PlaywrightError

    for frame in list(page.frames):
        try:
            found = probe(frame)
        except PlaywrightError as error:
            if not _detached(error):
                raise
            continue
        if found is not None:
            return found
    return None


def _find(
    page: Page, selectors: tuple[str, ...], wait_ms: int = FIELD_WAIT_MS
) -> Optional[Locator]:
    """First visible match in any frame; a hidden first match never masks a later one."""

    def probe(frame: Frame) -> Optional[Locator]:
        for selector in selectors:
            found = _visible(frame.locator(selector))
            if found is not None:
                return found
        return None

    for _ in range(max(1, wait_ms // POLL_MS)):
        found = _in_frames(page, probe)
        if found is not None:
            return found
        page.wait_for_timeout(POLL_MS)
    return None


def _card_method(page: Page) -> Optional[Locator]:
    def probe(frame: Frame) -> Optional[Locator]:
        found = _visible(frame.get_by_role("button", name=CARD_ROLE_NAME))
        for selector in CARD_METHOD:
            if found is not None:
                break
            found = _visible(frame.locator(selector))
        return found

    for _ in range(FIELD_WAIT_MS // POLL_MS):
        found = _in_frames(page, probe)
        if found is not None:
            return found
        page.wait_for_timeout(POLL_MS)
    return None


def _fill(page: Page, selectors: tuple[str, ...], value: str, required: bool = True) -> bool:
    field = _find(page, selectors, FIELD_WAIT_MS if required else 3 * POLL_MS)
    if field is None:
        return not required
    field.click()
    field.fill(value)
    return True


def _choose_card(page: Page) -> bool:
    """Open the Card accordion unless card inputs already show (legacy page)."""
    if _find(page, NUMBER, 3 * POLL_MS) is not None:
        return True
    method = _card_method(page)
    if method is None:
        return False
    if method.get_attribute("aria-expanded") != "true":
        method.click()
    return True


def _decline_link(page: Page) -> None:
    """Link's opt-in, when ticked, demands a phone number before submit."""
    box = _find(page, LINK_OPT_IN, 3 * POLL_MS)
    if box is not None:
        if box.is_checked():
            box.uncheck()
        return
    for frame in page.frames:  # Older pages: only the label is clickable.
        if frame.locator("input[type='checkbox']:checked").count():
            label = frame.get_by_text(SAVE_INFO)
            if label.count():
                label.first.click()
                return


def _captcha_challenge(page: Page) -> bool:
    for frame in page.locator(CAPTCHA_FRAME).all():
        if frame.is_visible():
            box = frame.bounding_box()
            if box and box["width"] >= CAPTCHA_MIN_PX and box["height"] >= CAPTCHA_MIN_PX:
                return True
    return False


def _pay(page: Page, url: str, progress: Optional[Progress] = None) -> Optional[str]:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    at = (progress or Progress()).at
    at("goto")
    page.goto(url, wait_until="load", timeout=CHECKOUT_PAGE_TIMEOUT_MS)
    at("card_method")
    if not _choose_card(page):
        return "stripe_test_checkout_field_absent:card_method"
    at("email")
    email = _find(page, EMAIL)
    if email is None:
        return "stripe_test_checkout_field_absent:email"
    if not email.input_value():
        email.click()
        email.fill(TEST_EMAIL)
    for name, selectors, value in (
        ("card_number", NUMBER, TEST_CARD),
        ("card_expiry", EXPIRY, "12 / 34"),
        ("card_cvc", CVC, "123"),
    ):
        at(name)
        if not _fill(page, selectors, value):
            return "stripe_test_checkout_field_absent:" + name
    at("billing_details")
    _fill(page, NAME, "J2 Acceptance", required=False)
    _fill(page, POSTAL, "94103", required=False)
    at("link_opt_in")
    _decline_link(page)
    at("submit")
    submit = _find(page, SUBMIT)
    if submit is None:
        return "stripe_test_checkout_field_absent:submit"
    submit.click()
    at("success_redirect")
    try:
        page.wait_for_url(re.compile(r"checkout=success"), timeout=CHECKOUT_PAGE_TIMEOUT_MS)
    except PlaywrightTimeoutError:
        # Never solved or bypassed: a visible challenge is reported as such.
        if _captcha_challenge(page):
            return "stripe_test_checkout_captcha_challenge"
        return "stripe_test_checkout_not_redirected"
    return None
