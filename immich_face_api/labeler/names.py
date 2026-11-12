"""Turning folder names into person names, and telling whether two spellings can be the same name."""

from __future__ import annotations

import re
import unicodedata

from .settings import NameRules

_SPACES = re.compile(r"\s+")
_LETTER = re.compile(r"[^\W\d_]")


def clean_name(raw: str, rules: NameRules) -> str | None:
    """Folder name -> display name, or None when it is not a person's name."""
    s = unicodedata.normalize("NFC", raw).replace("_", " ")
    s = _SPACES.sub(" ", s).strip()
    for _ in range(3):  # strip patterns can uncover each other ("12 - MP X" -> "MP X" -> "X")
        before = s
        for pat in rules.strip_patterns:
            s = re.sub(pat, " ", s, flags=re.I)
            s = _SPACES.sub(" ", s).strip(" .,-_()[]")
        if s == before:
            break
    if len(_LETTER.findall(s)) < 3:
        return None
    low = s.casefold()
    if low in {e.casefold() for e in rules.exclude}:
        return None
    if any(re.search(p, s, flags=re.I) for p in rules.exclude_patterns):
        return None
    if s.isupper() or s.islower():
        s = " ".join(w[:1].upper() + w[1:].lower() for w in s.split(" "))
    return s


def _skeleton(token: str) -> str:
    """Spelling-tolerant key of one name token: transliterations of the same name differ mostly in vowels and in
    "dh"/"d", "th"/"t" ("Rasheedh" / "Rasheed", "Musthofa" / "Musthafa")."""
    t = unicodedata.normalize("NFKD", token.casefold())
    t = "".join(c for c in t if c.isalpha())
    if not t:
        return ""
    for a, b in (("dh", "d"), ("th", "t"), ("sh", "s"), ("gh", "g"), ("kh", "k"), ("ph", "f"), ("q", "g"), ("x", "z")):
        t = t.replace(a, b)
    if len(t) > 2:
        t = t.rstrip("h")
    head, rest = t[0], re.sub(r"[aeiouyw]", "", t[1:])
    out = head
    for c in rest:
        if c != out[-1]:
            out += c
    return out


def name_tokens(name: str) -> set[str]:
    return {k for k in (_skeleton(w) for w in re.split(r"[\s.\-]+", name)) if len(k) >= 2}


def _soft(token: str) -> str:
    t = unicodedata.normalize("NFKD", token.casefold())
    t = "".join(c for c in t if c.isalpha())
    for a, b in (
        ("dh", "d"),
        ("th", "t"),
        ("gh", "g"),
        ("kh", "k"),
        ("ph", "f"),
        ("q", "g"),
        ("x", "z"),
        ("ee", "i"),
        ("oo", "u"),
        ("aa", "a"),
        ("ou", "u"),
    ):
        t = t.replace(a, b)
    if len(t) > 2:
        t = t.rstrip("h")
    return re.sub(r"(.)\1+", r"\1", t)


def name_key(name: str) -> str:
    """Same-name key (homonyms): word order and vowels matter ("Hassan" != "Hussain"), spelling variants of one
    name do not ("Rasheedh" == "Rasheed" == "Rashid" is NOT folded: i vs ee is, a vs u is not)."""
    return " ".join(k for k in (_soft(w) for w in re.split(r"[\s.\-]+", name)) if k)


def compatible(a: str, b: str) -> bool:
    """Could ``a`` and ``b`` be spellings of the same person's name? Every token of the shorter one must appear in
    the longer one (spelling-tolerant): "Jabir" / "Abdulla Jabir" yes, "Ali Rifau" / "Abdulla Rifau" no."""
    ta, tb = name_tokens(a), name_tokens(b)
    if not ta or not tb:
        return False
    return ta <= tb or tb <= ta
