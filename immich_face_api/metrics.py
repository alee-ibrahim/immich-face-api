"""Prometheus metrics, served on a separate port (FACE_API_METRICS_PORT) so the API port never exposes them."""

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("face_api_requests_total", "HTTP requests", ["endpoint", "status", "client"])
LATENCY = Histogram(
    "face_api_request_duration_seconds",
    "Request latency",
    ["endpoint"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 20, 30, 60),
)
STAGE = Histogram(
    "face_api_stage_duration_seconds",
    "Time per identify stage",
    ["stage"],  # decode, ml, db
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
)
FACES_PER_REQUEST = Histogram(
    "face_api_faces_per_request", "Faces detected per identify request", buckets=(0, 1, 2, 3, 5, 8, 13, 21, 34, 55)
)
FACES = Counter("face_api_faces_total", "Faces detected")
FACES_MATCHED = Counter("face_api_faces_matched_total", "Faces matched to a named person")
FACES_AMBIGUOUS = Counter("face_api_faces_ambiguous_total", "Matched faces with a close runner-up")
REJECTED = Counter("face_api_rejected_total", "Requests refused", ["reason"])
ML_ERRORS = Counter("face_api_ml_errors_total", "Machine-learning server errors")
DB_ERRORS = Counter("face_api_db_errors_total", "Database errors")
READY = Gauge("face_api_ready", "1 when the version pin, schema and dependencies are OK")
INFLIGHT = Gauge("face_api_ml_inflight", "Images being processed by the ML server")
REQUEST_LOG_WRITTEN = Counter("face_api_request_log_written_total", "Requests written to the request log")
REQUEST_LOG_DROPPED = Counter(
    "face_api_request_log_dropped_total", "Request log entries dropped (store down or queue full)", ["reason"]
)
CHANGES_CHECKS = Counter("face_api_people_change_checks_total", "People change-log checks", ["mode"])
CHANGES_EVENTS = Counter("face_api_people_change_events_total", "People change events written", ["type"])
CHANGES_ERRORS = Counter("face_api_people_change_errors_total", "Failed people change-log checks")
CHANGES_LAST_CHECK = Gauge("face_api_people_change_last_check_timestamp_seconds", "Last successful change-log check")
VERIFY_CLAIMS = Counter("face_api_verify_claims_total", "Claims checked by /v1/verify", ["verdict"])
