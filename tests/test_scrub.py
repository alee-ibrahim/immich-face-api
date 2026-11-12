from immich_face_api.scrub import redact, scrub_breadcrumb, scrub_event

FAKE = "f" * 40  # synthetic, low-entropy stand-in for a key


def test_redact_patterns():
    text = (
        "auth failed for Bearer abcdefghijklmnopqrstuvwxyz0123456789ABCD from jane@example.org "
        "emb [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] img data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ=="
    )
    out = redact(text)
    assert "abcdefghijklmnop" not in out
    assert "jane@example.org" not in out
    assert "0.5, 0.6" not in out
    assert "/9j/4AAQ" not in out
    assert len(redact("x" * 10_000)) <= 2000


def test_scrub_event_drops_request_user_and_locals():
    event = {
        "request": {
            "method": "POST",
            "url": "https://faces.example.org/v1/identify?candidates=3",
            "headers": {"Authorization": "Bearer secret-secret-secret-secret-secret"},
            "data": "binary image",
            "cookies": {"a": "b"},
            "query_string": "candidates=3",
        },
        "user": {"ip_address": "203.0.113.9"},
        "server_name": "pod-1",
        "exception": {
            "values": [
                {
                    "value": "bad token=" + FAKE,
                    "stacktrace": {"frames": [{"function": "f", "vars": {"key": "secret"}}]},
                }
            ]
        },
        "extra": {"note": "contact admin@example.org"},
        "breadcrumbs": {"values": [{"message": "Bearer zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"}]},
    }
    out = scrub_event(event)
    assert out["request"] == {"method": "POST", "url": "https://faces.example.org/v1/identify"}
    assert "user" not in out and "server_name" not in out
    exc = out["exception"]["values"][0]
    assert "vars" not in exc["stacktrace"]["frames"][0]
    assert FAKE not in exc["value"]
    assert "admin@example.org" not in str(out["extra"])
    assert "zzzzzzzz" not in str(out["breadcrumbs"])


def test_scrub_http_breadcrumb():
    crumb = {
        "category": "httplib",
        "data": {"url": "http://ml:3003/predict?x=1", "method": "POST", "status_code": 200},
    }
    assert scrub_breadcrumb(crumb)["data"] == {"method": "POST", "status_code": 200}
