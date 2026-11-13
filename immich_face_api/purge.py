"""``python -m immich_face_api purge``: delete stored request-log data now (with an audit entry).

    purge --crops --yes        all face crops and embeddings (the request log stays)
    purge --all --yes          the whole request log: requests, faces, crops, embeddings
    purge --retention          apply the configured retention now (what the API does every few hours)

Needs FACE_API_STORE_URL (and FACE_API_STORE_PASSWORD[_FILE] if the URL has no password). Without --yes it only
shows what is stored.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import sys

from .config import _env, _int, _secret
from .store import Store


async def run(args: argparse.Namespace) -> int:
    url = _secret("STORE_URL")
    if not url:
        print("FACE_API_STORE_URL is not set", file=sys.stderr)
        return 2
    store = Store(url, _secret("STORE_PASSWORD"), pool_size=1)
    await store.open(migrate=True)
    try:
        stats = await store.log_stats()
        print("stored:", json.dumps(stats, default=str))
        if args.retention:
            out = await store.purge(
                _int("USAGE_RETENTION_DAYS", 400, 1, 3650),
                _int("REQUEST_LOG_RETENTION_DAYS", 90, 1, 3650),
                _int("CROP_RETENTION_DAYS", 30, 1, 3650),
                force=True,
            )
            print("retention applied:", json.dumps(out))
            return 0
        scope = "all" if args.all else "crops" if args.crops else None
        if scope is None:
            print("nothing to do: give --crops, --all or --retention (see --help)")
            return 0
        if not args.yes:
            print(f"would delete scope={scope}; add --yes to do it")
            return 1
        actor = "cli:" + (_env("PURGE_ACTOR") or getpass.getuser() or "unknown")
        print("deleted:", json.dumps(await store.purge_all(scope, actor)))
        return 0
    finally:
        await store.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m immich_face_api purge", description=__doc__.split("\n\n")[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--crops", action="store_true", help="delete all face crops and embeddings")
    g.add_argument("--all", action="store_true", help="delete the whole request log")
    g.add_argument("--retention", action="store_true", help="apply the configured retention now")
    p.add_argument("--yes", action="store_true", help="really delete")
    args = p.parse_args(argv)
    if sys.platform == "win32":  # psycopg async needs a selector loop on Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # pragma: no cover
    return asyncio.run(run(args))
