"""SYN-5696: the Stripe TEST completer's page logic against fake Playwright doubles.

The 2026-11 hosted page (observed on staging): card inputs appear only after
the "Card" accordion row, a DIV with role=button, is chosen. These doubles pin
the order, the visible-match rule, the Link untick and the captcha cause; the
live proof is the staging checkout recorded in the SYN-5696 evidence.
"""

from __future__ import annotations

import pytest

from scripts.acceptance import stripe_checkout as sc

pytestmark = pytest.mark.acceptance


class Element:
    def __init__(self, page, name, visible=True, checked=False, box=None, expanded=None):
        self.page, self.name, self.visible = page, name, visible
        self.checked, self.box, self.expanded, self.value = checked, box, expanded, ""

    def is_visible(self):
        return self.visible

    def click(self, **_):
        self.page.log.append(("click", self.name))
        if self.name == "card":
            self.expanded = "true"
            self.page.reveal_card()

    def fill(self, value):
        self.page.log.append(("fill", self.name))
        self.value = value

    def input_value(self):
        return self.value

    def get_attribute(self, name):
        return self.expanded if name == "aria-expanded" else None

    def is_checked(self):
        return self.checked

    def uncheck(self):
        self.page.log.append(("uncheck", self.name))
        self.checked = False

    def bounding_box(self):
        return self.box


class Matches:
    def __init__(self, elements):
        self.elements = elements

    def count(self):
        return len(self.elements)

    def nth(self, index):
        return self.elements[index]

    @property
    def first(self):
        return self.elements[0]

    def all(self):
        return list(self.elements)


class Frame:
    def __init__(self, page):
        self.page = page

    def locator(self, selector):
        return Matches(self.page.dom.get(selector, []))

    def get_by_role(self, role, name):
        assert role == "button" and name is sc.CARD_ROLE_NAME
        return Matches(self.page.dom.get("role:card", []))

    def get_by_text(self, _):
        return Matches([])


class Page:
    """Payment Element page: email visible, card inputs hidden until Card is chosen."""

    def __init__(self, link_checked=True, redirect=True, captcha_box=None):
        self.log, self.redirect = [], redirect
        self.dom = {
            "#linkAuthentication-emailInput": [Element(self, "email")],
            "role:card": [Element(self, "card", expanded="false")],
            "#payment-linkOptInInput": [Element(self, "link", checked=link_checked)],
            "[data-testid='payment-form-submit-button']": [Element(self, "submit")],
            # A hidden decoy first: it must not mask the visible field.
            "input[name='number']": [Element(self, "decoy", visible=False)],
        }
        if captcha_box is not None:
            self.top = {sc.CAPTCHA_FRAME: [Element(self, "captcha", box=captcha_box)]}
        else:
            self.top = {}
        self.frames = [Frame(self)]

    def reveal_card(self):
        for selector, name in (
            ("#payment-numberInput", "number"),
            ("#payment-expiryInput", "expiry"),
            ("#payment-cvcInput", "cvc"),
            ("#payment-nameInput", "name"),
            ("#payment-postalCodeInput", "postal"),
        ):
            self.dom[selector] = [Element(self, name)]
        self.dom["input[name='number']"].append(self.dom["#payment-numberInput"][0])

    def goto(self, *_, **__):
        self.log.append(("goto", ""))

    def wait_for_timeout(self, _):
        pass

    def locator(self, selector):
        return Matches(self.top.get(selector, []))

    def wait_for_url(self, *_, **__):
        if not self.redirect:
            from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

            raise PlaywrightTimeoutError("no redirect")


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    pytest.importorskip("playwright")
    monkeypatch.setattr(sc, "FIELD_WAIT_MS", 1000)


URL = "https://checkout.stripe.com/c/pay/cs_test_fixture"


def test_card_is_chosen_first_then_fields_filled_link_unticked_and_submitted():
    page = Page()
    assert sc._pay(page, URL) is None
    actions = [entry for entry in page.log if entry[0] != "goto"]
    assert actions[0] == ("click", "card")
    assert ("fill", "number") in actions and ("fill", "decoy") not in actions
    assert ("uncheck", "link") in actions
    assert actions[-1] == ("click", "submit")
    assert actions.index(("uncheck", "link")) < actions.index(("click", "submit"))


def test_no_card_method_is_a_typed_cause():
    page = Page()
    page.dom["role:card"] = []
    assert sc._pay(page, URL) == "stripe_test_checkout_field_absent:card_method"


def test_visible_challenge_is_reported_never_solved():
    page = Page(redirect=False, captcha_box={"x": 0, "y": 0, "width": 400, "height": 580})
    assert sc._pay(page, URL) == "stripe_test_checkout_captcha_challenge"


def test_invisible_hcaptcha_frame_is_not_a_challenge():
    page = Page(redirect=False, captcha_box={"x": 0, "y": 0, "width": 0, "height": 0})
    assert sc._pay(page, URL) == "stripe_test_checkout_not_redirected"


SECRET_URL = "https://checkout.stripe.com/c/pay/cs_test_a1B2c3D4e5?x=q#fragmentsecret"


def test_browser_error_cause_names_kind_stage_and_redacted_location():
    from playwright.sync_api import Error, TimeoutError

    page = Page()
    page.url = SECRET_URL
    progress = sc.Progress()
    progress.page = page
    progress.at("card_number")
    cause = sc.browser_error_cause(TimeoutError("Timeout 20000ms " + SECRET_URL), progress)
    assert cause == (
        "stripe_test_checkout_browser_error:timeout:card_number:"
        "checkout.stripe.com/c/pay/cs_test_<redacted>"
    )
    assert "a1B2c3D4e5" not in cause and "fragmentsecret" not in cause and "x=q" not in cause
    navigation = sc.browser_error_cause(Error("net::ERR_ABORTED at " + SECRET_URL), progress)
    assert navigation.split(":")[1] == "navigation"
    other = sc.browser_error_cause(Error("boom"), sc.Progress())
    assert other == "stripe_test_checkout_browser_error:error:launch:none"


def test_pay_records_the_step_it_failed_on():
    from playwright.sync_api import Error

    page = Page()

    def broken(_value):
        raise Error("element is not attached")

    page.reveal_card()
    page.dom["#payment-numberInput"][0].fill = broken
    progress = sc.Progress()
    with pytest.raises(Error):
        sc._pay(page, URL, progress)
    assert progress.stage == "card_number"


class DetachedFrame(Frame):
    """A frame Stripe replaced after page.frames was enumerated."""

    def locator(self, selector):
        from playwright.sync_api import Error

        raise Error("Frame was detached")

    def get_by_role(self, role, name):
        from playwright.sync_api import Error

        raise Error("Target page, context or browser has been closed")


class ReloadingPage(Page):
    """First enumeration returns a frame that detaches; later ones the live frame."""

    def __init__(self, **options):
        super().__init__(**options)
        self.enumerations = 0
        self.stale = DetachedFrame(self)
        self.live = Frame(self)

    @property
    def frames(self):
        self.enumerations += 1
        return [self.stale] if self.enumerations == 1 else [self.stale, self.live]

    @frames.setter
    def frames(self, value):
        pass


def test_card_method_is_found_after_frames_reload_mid_walk():
    page = ReloadingPage()
    assert sc._pay(page, URL) is None
    assert page.enumerations >= 2
    assert ("click", "card") in page.log


def test_detached_frames_with_no_card_method_still_fail_typed():
    page = ReloadingPage()
    page.dom["role:card"] = []
    assert sc._pay(page, URL) == "stripe_test_checkout_field_absent:card_method"


def test_a_non_detach_browser_error_still_propagates():
    from playwright.sync_api import Error

    page = Page()

    class Broken(Frame):
        def get_by_role(self, role, name):
            raise Error("Protocol error: something else")

    page.frames = [Broken(page)]
    with pytest.raises(Error):
        sc._card_method(page)
