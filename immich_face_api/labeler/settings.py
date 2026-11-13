"""Labeler settings: connection details from environment variables (prefix LABELER_), the name sources and
thresholds from a JSON file (LABELER_CONFIG_FILE). Secrets only from files or the environment, never from argv."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path

PREFIX = "LABELER_"


class ConfigError(ValueError):
    pass


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(PREFIX + name)
    return default if value is None or value == "" else value


def _secret(name: str) -> str | None:
    path = _env(name + "_FILE")
    if path:
        return Path(path).read_text(encoding="utf-8").strip()
    return _env(name)


def _num(name: str, default, kind=float):
    raw = _env(name)
    if raw is None:
        return default
    try:
        return kind(raw)
    except ValueError as exc:
        raise ConfigError(f"{PREFIX}{name}: not a number") from exc


@dataclass
class Thresholds:
    """Distances are cosine distances (0 = same direction). Centroid-to-centroid distances are much smaller than
    face-to-face distances for the same person, because averaging removes pose and lighting noise."""

    face_distance: float = 0.5  # Immich's facialRecognition.maxDistance: "same person" for two single faces
    identity_merge_distance: float = 0.35  # two references with centroids this close are one identity
    same_name_merge_distance: float = 0.55  # ... or this close when their names are compatible (ageing portraits)
    match_max_distance: float = 0.55  # no proposal beyond this (centroid of cluster vs identity)
    high_max_distance: float = 0.30
    high_min_margin: float = 0.12  # distance to the next other identity minus distance to the best one
    high_min_consistency: float = 0.8  # share of the cluster's faces within face_distance of the identity
    high_min_cluster_faces: int = 3
    high_min_reference_faces: int = 2  # photos behind the identity (all its references together)
    high_min_name_tokens: int = 2  # "Jane" alone is not a name to apply automatically
    medium_max_distance: float = 0.40
    medium_min_margin: float = 0.06
    merge_distance: float = 0.30  # two clusters with centroids this close are a merge candidate
    folder_min_coverage: float = 0.5  # the dominant identity must be in at least this share of the folder's photos
    folder_ambiguity_ratio: float = 0.7  # second identity coverage / first >= this -> the folder is ambiguous
    folder_max_faces: int = 3000  # sample bigger folders (by photo) to bound the work
    cluster_sample_faces: int = 200  # faces per candidate cluster used for the consistency check
    min_cluster_faces: int = 1  # clusters with fewer faces are ignored
    # named-neighbour vote (named_people: true): unnamed clusters that are more clusters of an already-named person
    vote_high_share: float = 0.9
    vote_medium_share: float = 0.8
    vote_low_share: float = 0.5
    vote_max_median: float = 0.4
    vote_unnamed_sample: int = 50  # faces per unnamed cluster
    vote_named_sample: int = 300  # faces per named person


@dataclass
class FolderRule:
    """Folders under ``root`` whose relative path matches ``pattern`` from the start. The named group ``name`` is the
    person's name; the matched prefix is the folder (photos in its subfolders count too)."""

    root: str
    pattern: str
    label: str = ""
    regex: re.Pattern = field(init=False, repr=False)

    def __post_init__(self):
        try:
            self.regex = re.compile(self.pattern)
        except re.error as exc:
            raise ConfigError(f"folder pattern {self.pattern!r}: {exc}") from exc
        if "name" not in self.regex.groupindex:
            raise ConfigError(f"folder pattern {self.pattern!r} has no (?P<name>...) group")


@dataclass
class WebPage:
    url: str
    lang: str


@dataclass
class WebSource:
    """A source of (name, portrait) cards.

    ``kind: "listing"``: a listing page (one per language) with one card per person. ``card_pattern`` is a regex over
    the HTML with named groups ``id`` and ``name`` (or a ``name_template`` built from other groups) and ``image`` on
    the page of the primary language; ``image_template`` (Python format string over the groups) builds the image URL,
    e.g. for an archived page. ``alt_name_groups`` maps a language to a group holding the name in that language.
    Other named groups are kept as details (e.g. a role).

    ``kind: "wikidata"``: a SPARQL ``query`` for query.wikidata.org returning ``?item ?itemLabel ?image`` (optional
    ``?<lang>Label`` for other languages, e.g. ``?dvLabel``, and ``?role``). The licence and author of every
    Wikimedia Commons image are looked up and kept with the reference.

    ``licence``: what the source's images may be used for (recorded in the report and audit)."""

    key: str
    label: str
    pages: list[WebPage] = field(default_factory=list)
    card_pattern: str = ""
    primary_lang: str = "en"
    kind: str = "listing"
    query: str = ""
    image_template: str = ""
    name_template: str = ""
    alt_name_groups: dict = field(default_factory=dict)
    immutable: bool = False  # pages never change (archived snapshots): fetch once
    licence: str = ""
    regex: re.Pattern | None = field(init=False, repr=False, default=None)

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,40}", self.key):
            raise ConfigError(f"web source key {self.key!r}: use [A-Za-z0-9_.-]")
        if self.kind == "wikidata":
            if "?item" not in self.query or "?image" not in self.query:
                raise ConfigError(f"web source {self.key}: a wikidata query must select ?item and ?image")
            return
        if self.kind != "listing":
            raise ConfigError(f"web source {self.key}: kind must be listing or wikidata")
        self.pages = [p if isinstance(p, WebPage) else WebPage(**p) for p in self.pages]
        try:
            self.regex = re.compile(self.card_pattern, re.S)
        except re.error as exc:
            raise ConfigError(f"web source {self.key}: {exc}") from exc
        if "id" not in self.regex.groupindex:
            raise ConfigError(f"web source {self.key}: card_pattern needs a (?P<id>...) group")
        if "name" not in self.regex.groupindex and not self.name_template:
            raise ConfigError(f"web source {self.key}: card_pattern needs a (?P<name>...) group or a name_template")
        if self.primary_lang not in {p.lang for p in self.pages}:
            raise ConfigError(f"web source {self.key}: no page for primary language {self.primary_lang}")


@dataclass
class NameRules:
    strip_patterns: list[str] = field(default_factory=list)  # removed (case-insensitive), in order, repeatedly
    exclude: list[str] = field(default_factory=list)  # cleaned names that are not people (case-insensitive)
    exclude_patterns: list[str] = field(default_factory=list)
    homonym_detail: str = ""  # web card detail (named group) appended to tell same-name people apart: "Name (detail)"


@dataclass
class SourcesConfig:
    folders: list[FolderRule] = field(default_factory=list)
    web: list[WebSource] = field(default_factory=list)
    names: NameRules = field(default_factory=NameRules)
    thresholds: Thresholds = field(default_factory=Thresholds)
    user_agent: str = "immich-people-labeler"
    request_delay_seconds: float = 2.0
    page_refresh_hours: float = 168.0
    immich_web_url: str = ""  # for links in the report (the URL people open Immich with)
    # people already named in Immich (by anyone) as references: finds the person's other, unnamed clusters
    named_people: bool = False
    # sources whose evidence may be WRITTEN ("folder", "named" or a web source key); others only propose (held).
    # Unset = every source may be written.
    write_sources: list[str] | None = None

    @classmethod
    def from_dict(cls, data: dict) -> SourcesConfig:
        if not isinstance(data, dict):
            raise ConfigError("config: expected a JSON object")
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"config: unknown keys {sorted(unknown)}")
        th = data.get("thresholds", {})
        tkeys = {f.name for f in fields(Thresholds)}
        if set(th) - tkeys:
            raise ConfigError(f"config thresholds: unknown keys {sorted(set(th) - tkeys)}")
        try:
            return cls(
                folders=[FolderRule(**f) for f in data.get("folders", [])],
                web=[WebSource(**w) for w in data.get("web", [])],
                names=NameRules(**data.get("names", {})),
                thresholds=Thresholds(**th),
                user_agent=data.get("user_agent", cls.user_agent),
                request_delay_seconds=float(data.get("request_delay_seconds", cls.request_delay_seconds)),
                page_refresh_hours=float(data.get("page_refresh_hours", cls.page_refresh_hours)),
                immich_web_url=data.get("immich_web_url", "").rstrip("/"),
                named_people=bool(data.get("named_people", False)),
                write_sources=data.get("write_sources"),
            )
        except TypeError as exc:
            raise ConfigError(f"config: {exc}") from exc

    @classmethod
    def load(cls, path: str) -> SourcesConfig:
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"config {path}: invalid JSON: {exc}") from exc


@dataclass
class LabelerSettings:
    immich_url: str
    api_key: str
    db_host: str
    db_user: str
    db_password: str | None
    db_port: int = 5432
    db_name: str = "immich"
    db_sslmode: str = "prefer"
    db_sslrootcert: str | None = None
    state_db_name: str = "immich_labeler"
    state_db_host: str | None = None
    state_db_user: str | None = None
    state_db_password: str | None = None
    ml_url: str = "http://immich-machine-learning:3003"
    face_model: str = "buffalo_l"
    min_score: float = 0.7
    max_edge: int = 1440
    expected_immich_version: str = ""
    owner_email: str | None = None
    config_file: str = ""
    output_dir: str | None = None
    max_apply: int = 500  # hard cap of names written per command
    max_copies: int = 20000  # cap of share-names copies per run
    keep_reports: int = 30  # older runs keep their proposals and audit, but not the report blob
    http_timeout: float = 60.0
    sources: SourcesConfig = field(default_factory=SourcesConfig)

    @classmethod
    def from_env(cls) -> LabelerSettings:
        required = {"IMMICH_URL": _env("IMMICH_URL"), "DB_HOST": _env("DB_HOST"), "DB_USER": _env("DB_USER")}
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ConfigError("missing " + ", ".join(PREFIX + m for m in missing))
        key = _secret("API_KEY")
        if not key or key == "REPLACE_ME":
            raise ConfigError(f"{PREFIX}API_KEY(_FILE): the Immich API key of the library owner is not set")
        config_file = _env("CONFIG_FILE", "")
        s = cls(
            immich_url=required["IMMICH_URL"].rstrip("/"),
            api_key=key,
            db_host=required["DB_HOST"],
            db_user=required["DB_USER"],
            db_password=_secret("DB_PASSWORD"),
            db_port=_num("DB_PORT", 5432, int),
            db_name=_env("DB_NAME", "immich"),
            db_sslmode=_env("DB_SSLMODE", "prefer"),
            db_sslrootcert=_env("DB_SSLROOTCERT"),
            state_db_name=_env("STATE_DB_NAME", "immich_labeler"),
            state_db_host=_env("STATE_DB_HOST"),
            state_db_user=_env("STATE_DB_USER"),
            state_db_password=_secret("STATE_DB_PASSWORD"),
            ml_url=_env("ML_URL", "http://immich-machine-learning:3003").rstrip("/"),
            face_model=_env("FACE_MODEL", "buffalo_l"),
            min_score=_num("MIN_SCORE", 0.7),
            max_edge=_num("MAX_EDGE", 1440, int),
            expected_immich_version=_env("EXPECTED_IMMICH_VERSION", ""),
            owner_email=_env("OWNER_EMAIL"),
            config_file=config_file,
            output_dir=_env("OUTPUT_DIR"),
            max_apply=_num("MAX_APPLY", 500, int),
            max_copies=_num("MAX_COPIES", 20000, int),
            keep_reports=_num("KEEP_REPORTS", 30, int),
            http_timeout=_num("HTTP_TIMEOUT", 60.0),
            sources=SourcesConfig.load(config_file) if config_file else SourcesConfig(),
        )
        if not 1 <= s.max_apply <= 100000:
            raise ConfigError(f"{PREFIX}MAX_APPLY: 1..100000")
        return s
