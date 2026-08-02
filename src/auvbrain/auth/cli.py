"""Command-line utility for managing API keys.

Usage
-----
Create a key::

    python -m auvbrain.auth.cli create --name "operator" --scopes "read write"

List active keys::

    python -m auvbrain.auth.cli list

Revoke a key by UUID::

    python -m auvbrain.auth.cli revoke --id <uuid>
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from datetime import datetime, timezone

from ..config import load_settings
from ..db.engine import get_session, init_db
from ..db.repositories import ApiKeyRepository


async def _create(name: str, scopes: str, expires: str | None) -> None:
    settings = load_settings()
    await init_db(settings)

    expires_at: datetime | None = None
    if expires:
        expires_at = datetime.fromisoformat(expires).replace(tzinfo=timezone.utc)

    async with get_session(settings) as session:
        repo = ApiKeyRepository(session)
        record, raw = await repo.create(name=name, scopes=scopes, expires_at=expires_at)

    print("─" * 60)
    print(f"  Name   : {record.name}")
    print(f"  ID     : {record.id}")
    print(f"  Scopes : {record.scopes}")
    if expires_at:
        print(f"  Expires: {expires_at.isoformat()}")
    print()
    print(f"  API KEY (shown once — store it now):")
    print(f"  {raw}")
    print("─" * 60)


async def _list() -> None:
    settings = load_settings()
    await init_db(settings)

    async with get_session(settings) as session:
        repo = ApiKeyRepository(session)
        keys = await repo.list_active()

    if not keys:
        print("No active API keys.")
        return

    print(f"{'ID':<38} {'Name':<20} {'Scopes':<20} {'Last used'}")
    print("─" * 90)
    for k in keys:
        last = k.last_used_at.isoformat() if k.last_used_at else "never"
        print(f"{str(k.id):<38} {k.name:<20} {k.scopes:<20} {last}")


async def _revoke(key_id: str) -> None:
    settings = load_settings()
    async with get_session(settings) as session:
        repo = ApiKeyRepository(session)
        ok = await repo.revoke(uuid.UUID(key_id))
    if ok:
        print(f"Revoked key {key_id}.")
    else:
        print(f"Key {key_id} not found.", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m auvbrain.auth.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="Create a new API key")
    p_create.add_argument("--name", required=True)
    p_create.add_argument("--scopes", default="read write")
    p_create.add_argument("--expires", default=None, help="ISO 8601 date e.g. 2027-01-01")

    sub.add_parser("list", help="List active API keys")

    p_revoke = sub.add_parser("revoke", help="Revoke a key by UUID")
    p_revoke.add_argument("--id", required=True)

    args = parser.parse_args()

    if args.command == "create":
        asyncio.run(_create(args.name, args.scopes, args.expires))
    elif args.command == "list":
        asyncio.run(_list())
    elif args.command == "revoke":
        asyncio.run(_revoke(args.id))


if __name__ == "__main__":
    main()
