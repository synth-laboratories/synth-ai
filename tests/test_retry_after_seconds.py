"""retry_after_seconds keeps an HTTP-date or fractional Retry-After instead of dropping it."""

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

from synth_ai.errors import retry_after_seconds


def test_http_date_is_kept_and_rounded_up():
    when = format_datetime(datetime.now(UTC) + timedelta(seconds=30), usegmt=True)
    assert retry_after_seconds(when) in {30, 31}


def test_fractional_and_integer_delays_round_up():
    assert retry_after_seconds("1.5") == 2
    assert retry_after_seconds("7") == 7


def test_absent_or_unparseable_is_none():
    assert retry_after_seconds(None) is None
    assert retry_after_seconds("soon") is None
