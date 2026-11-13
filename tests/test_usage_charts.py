from datetime import UTC, datetime, timedelta

from immich_face_api.charts import bar_chart, line_chart
from immich_face_api.store import new_key, series, split_key, totals
from immich_face_api.usage import LATENCY_BOUNDS_MS, N_BUCKETS, UsageBuffer, percentile


def test_buffer_record_drain_restore():
    b = UsageBuffer()
    now = datetime(2026, 1, 1, 10, 30, tzinfo=UTC)
    b.record("web", "/v1/identify", 200, 120, faces=3, matched=2, ambiguous=1, now=now)
    b.record("web", "/v1/identify", 502, 40, now=now)
    b.record("web", "/v1/identify", 429, 1, now=now + timedelta(hours=1))
    data = b.drain()
    assert len(data) == 2 and b.drain() == {}
    c = data[(datetime(2026, 1, 1, 10, tzinfo=UTC), "web", "/v1/identify")]
    assert (c.requests, c.faces, c.matched, c.ambiguous, c.server_errors, c.client_errors) == (2, 3, 2, 1, 1, 0)
    assert sum(c.latency_hist) == 2 and len(c.latency_hist) == N_BUCKETS
    b.restore(data)
    b.record("web", "/v1/identify", 200, 10, now=now)
    again = b.drain()
    assert again[(datetime(2026, 1, 1, 10, tzinfo=UTC), "web", "/v1/identify")].requests == 3


def test_percentile():
    hist = [0] * N_BUCKETS
    assert percentile(hist, 0.5) is None
    hist[LATENCY_BOUNDS_MS.index(1000)] = 10  # all in (750, 1000]
    p50 = percentile(hist, 0.5)
    assert 750 < p50 <= 1000
    hist[-1] = 1
    assert percentile(hist, 1.0) >= LATENCY_BOUNDS_MS[-1]


def test_key_format():
    key, prefix, digest = new_key()
    assert key.startswith(f"ifa_{prefix}_") and len(digest) == 64
    assert split_key(key) == prefix
    for bad in ("ifa_xyz_" + "a" * 40, "ifb_" + prefix + "_" + "a" * 40, "ifa_" + prefix + "_short", "plain"):
        assert split_key(bad) is None


def _row(hour, requests, faces=0, matched=0):
    return {
        "hour": hour,
        "requests": requests,
        "client_errors": 0,
        "server_errors": 0,
        "faces": faces,
        "matched": matched,
        "ambiguous": 0,
        "latency_ms_sum": 0,
        "latency_hist": [requests] + [0] * (N_BUCKETS - 1),
    }


def test_series_fills_gaps_and_totals():
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    rows = [_row(now - timedelta(hours=3), 2, 4, 1), _row(now, 5)]
    pts = series(rows, now - timedelta(hours=5), "hour")
    assert len(pts) == 6 and [c.requests for _, c in pts] == [0, 0, 2, 0, 0, 5]
    t = totals(rows)
    assert (t.requests, t.faces, t.matched) == (7, 4, 1)
    days = series(rows, now - timedelta(days=2), "day")
    assert sum(c.requests for _, c in days) == 7


def test_charts_are_svg_with_tooltips():
    svg = bar_chart(["a", "b", "c"], [0, 3, 10], unit="requests", title="Requests")
    assert svg.startswith("<svg") and svg.count("<title>") == 3 and "b: 3 requests" in svg
    assert "nan" not in svg.lower()
    line = line_chart(
        ["a", "b", "c"], [None, 40.0, 60.0], unit="%", title="Match", fmt=lambda v: f"{v:.0f}%", y_max=100
    )
    assert 'class="line"' in line and "a: no data" in line
    empty = line_chart(["a"], [None], unit="ms", title="p95", fmt=str)
    assert empty.startswith("<svg") and "nan" not in empty.lower()
