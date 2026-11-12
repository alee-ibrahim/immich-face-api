# syntax=docker/dockerfile:1
# Alpine: small, and none of the Debian packages (libxml2 and friends) this service does not need.
FROM python:3.13-alpine@sha256:9ef1a599f863231c3d91e941389bfdc4a2d6070557243d54df1a4e57015b960e AS build
WORKDIR /app
COPY requirements.txt .
RUN python -m venv /venv \
 && /venv/bin/pip install --no-cache-dir --only-binary=:all: -r requirements.txt \
 && /venv/bin/pip uninstall -y pip setuptools 2>/dev/null || true

FROM python:3.13-alpine@sha256:9ef1a599f863231c3d91e941389bfdc4a2d6070557243d54df1a4e57015b960e
LABEL org.opencontainers.image.title="immich-face-api" \
      org.opencontainers.image.description="Who is in this photo? Face identification against the named people of an Immich library" \
      org.opencontainers.image.licenses="MIT"
# pip is not needed at run time (and is a frequent CVE source)
RUN apk upgrade --no-cache \
 && pip uninstall -y pip setuptools wheel 2>/dev/null || true \
 && rm -rf /usr/local/lib/python3.13/ensurepip
COPY --from=build /venv /venv
COPY immich_face_api /app/immich_face_api
WORKDIR /app
ENV PATH=/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
USER 65532:65532
EXPOSE 8080 9090
ENTRYPOINT ["python", "-m", "immich_face_api"]
