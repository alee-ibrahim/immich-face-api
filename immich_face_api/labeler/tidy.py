"""tidy-names: remove titles from names already in Immich (config ``names.titles``; ``names.keep_titles`` written one
way), whoever wrote them. Only title words at the start or end of a name change, nothing else.

Owner names (``tidy-names``) are written through the Immich API with the owner key; curators' copies
(``tidy-names --copies``) through the narrow name-copy role, and only where the copy without its titles equals the
owner's (already tidy) name; a copy with its own spelling is listed and left alone. Names where a title word is
still left inside the name after tidying are listed as doubtful and never changed. Dry run unless ``--apply``; every
write is in the audit table first (target owner/copy), so ``undo --run <id>`` / ``undo --copies --run <id>`` put the
old names back where nobody has changed them since.
"""

from __future__ import annotations

import logging
from collections import Counter

from .immich import ImmichError
from .names import has_title, tidy_name, title_key
from .settings import NameRules

log = logging.getLogger(__name__)


def removed_titles(old: str, new: str, rules: NameRules) -> list[str]:
    """The title words that tidying took away or rewrote (for the counts)."""
    keys = {title_key(t) for t in [*rules.titles, *rules.keep_titles]}
    new_words = new.split()
    out = [w for w in old.split() if title_key(w.strip("()")) in keys and w not in new_words]
    return out or ["(spacing/casing)"]


def plan_owner(rows: list[tuple[str, str]], rules: NameRules) -> tuple[list[dict], list[dict]]:
    """Owner names to change, and doubtful names (a title word inside the name) that stay as they are."""
    changes, doubtful = [], []
    kept_forms = set(rules.keep_titles.values())
    for pid, name in rows:
        new = tidy_name(name, rules)
        inner = [w for w in new.split()[1 if new.split()[0] in kept_forms else 0 :]] if new else []
        if has_title(" ".join(inner), rules):
            doubtful.append({"person_id": pid, "name": name, "would_be": new, "why": "a title word inside the name"})
            continue
        if new != name:
            changes.append({"person_id": pid, "old": name, "new": new, "titles": removed_titles(name, new, rules)})
    return changes, doubtful


def plan_copies(rows: list[tuple[str, str, str, str]], rules: NameRules) -> tuple[list[dict], list[dict]]:
    """Copies to change (they follow the owner's tidy name) and copies with their own spelling (left alone)."""
    changes, own_spelling = [], []
    for group, user, name, owner_name in rows:
        new = tidy_name(name, rules)
        if new == name:
            continue
        item = {"person_id": group, "user_id": user, "old": name, "new": new, "owner_name": owner_name}
        if new != tidy_name(owner_name, rules):
            own_spelling.append(item)
        else:
            changes.append(item)
    return changes, own_spelling


def summary(changes: list[dict], doubtful: list[dict], sample: int = 40) -> dict:
    by_title = Counter(t for c in changes for t in c.get("titles", []))
    return {
        "changes": len(changes),
        "by_title": dict(by_title),
        "sample": [f"{c['old']} -> {c['new']}" for c in changes[:sample]],
        "doubtful": doubtful,
    }


def apply_owner(api, store, run_id: str, owner: str, changes: list[dict], cap: int) -> Counter:
    c: Counter = Counter()
    for ch in changes:
        if c["applied"] >= cap:
            c["over_cap"] += 1
            continue
        try:
            cur = api.person(ch["person_id"])
        except ImmichError:
            c["skipped_gone"] += 1
            continue
        if cur.get("name") != ch["old"]:
            c["skipped_changed"] += 1
            continue
        aid = store.audit_rename(run_id, ch["person_id"], owner, ch["old"], ch["new"], "owner", "tidy-names")
        try:
            res = api.set_name(ch["person_id"], ch["new"])
        except ImmichError as exc:
            store.audit_set(aid, "failed", str(exc))
            c["failed"] += 1
            continue
        if res.get("name") != ch["new"]:
            store.audit_set(aid, "failed", "Immich answered with a different name")
            c["failed"] += 1
            continue
        store.audit_set(aid, "applied", "tidy-names")
        c["applied"] += 1
    return c


def apply_copies(writer, store, run_id: str, changes: list[dict], cap: int) -> Counter:
    c: Counter = Counter()
    for ch in changes:
        if c["applied"] >= cap:
            c["over_cap"] += 1
            continue
        if ch["owner_name"] != ch["new"]:  # copies follow the owner's tidy name, never lead
            c["skipped_owner_not_tidy"] += 1
            continue
        aid = store.audit_rename(run_id, ch["person_id"], ch["user_id"], ch["old"], ch["new"], "copy", "tidy-names")
        if writer.rename(ch["user_id"], ch["person_id"], ch["old"], ch["new"]):
            store.audit_set(aid, "applied", "tidy-names")
            c["applied"] += 1
        else:
            store.audit_set(aid, "skipped", "copy changed since")
            c["skipped_changed"] += 1
    return c
