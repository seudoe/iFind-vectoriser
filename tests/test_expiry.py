from datetime import datetime, timedelta, timezone

from vectorizer_hnsw_pipeline import deadline_passed

NOW = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)


def test_none_and_garbage_are_not_expired():
    assert not deadline_passed(None, NOW)
    assert not deadline_passed("", NOW)
    assert not deadline_passed("soon", NOW)


def test_naive_iso_string_as_stored_by_the_scraper():
    assert deadline_passed("2026-11-03T00:00:00", NOW)       # yesterday
    assert not deadline_passed("2026-11-04T00:00:00", NOW)   # date-only = still open until the end of today
    assert not deadline_passed("2026-12-01T00:00:00", NOW)


def test_exact_time_and_dates():
    assert deadline_passed("2026-11-04T10:00:00", NOW)
    assert not deadline_passed("2026-11-04T13:00:00", NOW)
    assert deadline_passed(NOW - timedelta(days=2), NOW)
    assert not deadline_passed(NOW + timedelta(days=2), NOW)
