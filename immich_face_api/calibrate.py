"""``python -m immich_face_api calibrate``: fit the main-subject model to curator labels.

    calibrate                                 report how the current model does on the labels (no output file)
    calibrate --out subject-model.json        fit, cross-validate, write a new versioned model
    calibrate --labels photos.jsonl --out …   from an exported file instead of the store
    calibrate --export photos.jsonl           write the labelled photos (signals + labels, no images) to a file

The labels come from the admin page "Label subjects" (store table ``subject_sample``; needs FACE_API_STORE_URL and
FACE_API_STORE_PASSWORD[_FILE]). A JSONL file has one photo per line: ``{"signals": [{...}, ...], "labels": [...]}``.

How it fits:

1. **Prominence weights**: L2-regularised logistic regression (Newton's method, on standardised features) of
   "this face is a subject" (``main`` or ``co_subject``) against ``background``, on the model's features
   (``subject.FEATURES``) computed from the stored signals.
2. **Role thresholds**: with those weights, a grid search over ``main_min``, ``co_min``, ``co_margin`` and
   ``tie_dominance`` for the best mean of the F1 scores of "main" and of "subject" (main or co-subject). The hard
   rules (tiny and blurred faces, photo type) keep their values from the current model.
3. **Honest numbers**: k-fold cross-validation by photo (steps 1-2 on the other folds, measured on the held-out
   fold); then the final model is fitted on all photos.

Precision/recall of "main": of the faces the model calls main, how many the curators called main; of the faces the
curators called main, how many the model found. The model is written with a new ``version`` (default
``cal-<UTC date and time>``) that the API reports as ``subject_model_version``.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import random
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import labels as lb
from .config import _secret
from .subject import (
    FEATURES,
    ROLES,
    SubjectModel,
    assign_roles,
    feature_vector,
    is_gated,
    load_model,
    model_inputs,
    photo_type_of,
    raw_prominences,
    score,
)

MIN_PHOTOS = 30
MIN_PER_CLASS = 5


@dataclass
class Metrics:
    main_tp: int = 0
    main_fp: int = 0
    main_fn: int = 0
    subj_tp: int = 0
    subj_fp: int = 0
    subj_fn: int = 0
    correct: int = 0
    faces: int = 0

    def add(self, predicted: list[str], truth: list[str]) -> None:
        for p, t in zip(predicted, truth, strict=True):
            self.faces += 1
            self.correct += int(p == t)
            self.main_tp += int(p == "main" and t == "main")
            self.main_fp += int(p == "main" and t != "main")
            self.main_fn += int(p != "main" and t == "main")
            ps, ts = p != "background", t != "background"
            self.subj_tp += int(ps and ts)
            self.subj_fp += int(ps and not ts)
            self.subj_fn += int(ts and not ps)

    @staticmethod
    def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    def main(self) -> tuple[float, float, float]:
        return self._prf(self.main_tp, self.main_fp, self.main_fn)

    def subject(self) -> tuple[float, float, float]:
        return self._prf(self.subj_tp, self.subj_fp, self.subj_fn)

    def objective(self) -> float:
        return (self.main()[2] + self.subject()[2]) / 2

    def as_dict(self) -> dict:
        mp, mr, mf = self.main()
        sp, sr, sf = self.subject()
        return {
            "faces": self.faces,
            "accuracy": round(self.correct / self.faces, 4) if self.faces else None,
            "main": {"precision": round(mp, 4), "recall": round(mr, 4), "f1": round(mf, 4)},
            "subject": {"precision": round(sp, 4), "recall": round(sr, 4), "f1": round(sf, 4)},
        }


def validate(photos: list[dict]) -> list[dict]:
    out = []
    for p in photos:
        sig, lab = p.get("signals"), p.get("labels")
        if not isinstance(sig, list) or not isinstance(lab, list) or len(sig) != len(lab) or not sig:
            continue
        if any(x not in ROLES for x in lab):
            continue
        out.append({"signals": sig, "labels": lab})
    return out


def design(photos: list[dict], rules: dict) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix and "is a subject" target over all faces of the photos."""
    rows, ys = [], []
    for p in photos:
        sigs = p["signals"]
        _, count = photo_type_of(sigs, rules)
        gated = [is_gated(sig, rules, len(sigs)) for sig in sigs]
        for x, lab in zip(model_inputs(sigs, gated, count), p["labels"], strict=True):
            rows.append(feature_vector(x))
            ys.append(1.0 if lab != "background" else 0.0)
    return np.asarray(rows, dtype=np.float64).reshape(-1, len(FEATURES)), np.asarray(ys)


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> tuple[np.ndarray, float]:
    """(weights, intercept) in the raw feature space. Newton's method on standardised features, L2 on the weights."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-9] = 1.0
    z = np.hstack([np.ones((len(x), 1)), (x - mean) / std])
    beta = np.zeros(z.shape[1])
    reg = np.full(z.shape[1], l2)
    reg[0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(z @ beta, -35, 35)))
        grad = z.T @ (p - y) + reg * beta
        hess = (z * (p * (1 - p))[:, None]).T @ z + np.diag(reg) + 1e-9 * np.eye(len(beta))
        step = np.linalg.solve(hess, grad)
        beta -= step
        if np.abs(step).max() < 1e-8:
            break
    w = beta[1:] / std
    b = beta[0] - float((beta[1:] * mean / std).sum())
    return w, b


def with_weights(
    base: SubjectModel, w: np.ndarray, b: float, rules: dict | None = None, version: str = ""
) -> SubjectModel:
    return SubjectModel(
        version=version or base.version,
        intercept=float(b),
        weights={f: float(v) for f, v in zip(FEATURES, w, strict=True)},
        rules={**base.rules, **(rules or {})},
        description=base.description,
    )


def evaluate(model: SubjectModel, photos: list[dict]) -> Metrics:
    m = Metrics()
    for p in photos:
        res = score(p["signals"], model)
        m.add([f.role for f in res.faces], p["labels"])
    return m


GRID = {
    "main_min": [round(0.30 + 0.05 * i, 2) for i in range(13)],  # 0.30 .. 0.90
    "co_min": [round(0.15 + 0.05 * i, 2) for i in range(12)],  # 0.15 .. 0.70
    "co_margin": [0.1, 0.15, 0.2, 0.25, 0.3],
    "tie_dominance": [0.8, 0.85, 0.9, 0.95, 1.01],
}


def tune_rules(model: SubjectModel, photos: list[dict]) -> tuple[dict, Metrics]:
    """The grid point with the best objective; prominences are computed once (they do not depend on the rules)."""
    pre = [(raw_prominences(p["signals"], model), p) for p in photos]
    best, best_m = None, None
    for main_min, co_min, margin, tie in itertools.product(*GRID.values()):
        if co_min > main_min:
            continue
        rules = {**model.rules, "main_min": main_min, "co_min": co_min, "co_margin": margin, "tie_dominance": tie}
        m = Metrics()
        for (prom, gated, photo_type, _), p in pre:
            roles, _ = assign_roles(prom, p["signals"], gated, photo_type, rules)
            m.add(roles, p["labels"])
        if best_m is None or m.objective() > best_m.objective() + 1e-12:
            best, best_m = {k: rules[k] for k in GRID}, m
    return best or {}, best_m or Metrics()


def fit(base: SubjectModel, photos: list[dict], l2: float, version: str = "") -> tuple[SubjectModel, Metrics]:
    x, y = design(photos, base.rules)
    w, b = fit_logistic(x, y, l2=l2)
    model = with_weights(base, w, b, version=version)
    rules, m = tune_rules(model, photos)
    return with_weights(base, w, b, rules, version=version), m


def cross_validate(base: SubjectModel, photos: list[dict], folds: int, l2: float, seed: int = 7) -> Metrics:
    order = list(range(len(photos)))
    random.Random(seed).shuffle(order)  # noqa: S311 - a reproducible split, not a secret
    total = Metrics()
    for k in range(folds):
        test = [photos[i] for j, i in enumerate(order) if j % folds == k]
        train = [photos[i] for j, i in enumerate(order) if j % folds != k]
        if not test or not train:
            continue
        model, _ = fit(base, train, l2)
        m = evaluate(model, test)
        for field in vars(total):
            setattr(total, field, getattr(total, field) + getattr(m, field))
    return total


def check_enough(photos: list[dict]) -> str | None:
    counts = {r: sum(p["labels"].count(r) for p in photos) for r in ROLES}
    if len(photos) < MIN_PHOTOS:
        return f"need at least {MIN_PHOTOS} labelled photos, have {len(photos)}"
    if counts["main"] < MIN_PER_CLASS or counts["background"] < MIN_PER_CLASS:
        return f"need at least {MIN_PER_CLASS} main and {MIN_PER_CLASS} background faces, have {counts}"
    return None


def report(title: str, m: Metrics) -> None:
    d = m.as_dict()
    print(
        f"{title}: {d['faces']} faces, accuracy {d['accuracy']}, "
        f"main P {d['main']['precision']:.3f} R {d['main']['recall']:.3f} F1 {d['main']['f1']:.3f}, "
        f"subject P {d['subject']['precision']:.3f} R {d['subject']['recall']:.3f} F1 {d['subject']['f1']:.3f}"
    )


async def load_from_store() -> list[dict]:
    from .store import Store

    url = _secret("STORE_URL")
    if not url:
        raise SystemExit("FACE_API_STORE_URL is not set (or give --labels FILE)")
    store = Store(url, _secret("STORE_PASSWORD"), pool_size=1)
    await store.open(migrate=True)
    try:
        return await lb.training_photos(store)
    finally:
        await store.close()


def run(args: argparse.Namespace) -> int:
    base = load_model(args.model or _model_file_from_env())
    if args.labels:
        lines = Path(args.labels).read_text(encoding="utf-8").splitlines()
        photos = [json.loads(line) for line in lines if line.strip()]
    else:
        photos = asyncio.run(load_from_store())
    photos = validate(photos)
    counts = {r: sum(p["labels"].count(r) for p in photos) for r in ROLES}
    print(f"{len(photos)} labelled photos; faces: {counts}")
    if args.export:
        with open(args.export, "w", encoding="utf-8") as fh:
            for p in photos:
                fh.write(json.dumps(p) + "\n")
        print(f"exported to {args.export}")
        return 0
    if not photos:
        print("no labelled photos", file=sys.stderr)
        return 2
    report(f"current model {base.version}", evaluate(base, photos))
    if not args.out:
        return 0
    problem = check_enough(photos)
    if problem and not args.force:
        print(f"not calibrating: {problem} (--force to try anyway)", file=sys.stderr)
        return 2
    folds = max(2, min(args.folds, len(photos)))
    cv = cross_validate(base, photos, folds, args.l2)
    report(f"cross-validated new model ({folds} folds)", cv)
    version = args.version or "cal-" + datetime.now(UTC).strftime("%Y%m%d-%H%M")
    model, fitted = fit(base, photos, args.l2, version=version)
    report("new model on all labels (optimistic)", fitted)
    out = model.to_dict()
    out["description"] = f"fitted on {len(photos)} photos / {sum(counts.values())} faces"
    out["calibration"] = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "base_version": base.version,
        "photos": len(photos),
        "faces": counts,
        "folds": folds,
        "l2": args.l2,
        "cross_validated": cv.as_dict(),
        "current_model": evaluate(base, photos).as_dict(),
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} (version {version}); weights:")
    for f in FEATURES:
        print(f"  {f:16s} {model.weights[f]:+.3f}")
    print(f"  {'intercept':16s} {model.intercept:+.3f}")
    print("  rules:", {k: model.rules[k] for k in GRID})
    return 0


def _model_file_from_env() -> str | None:
    from .config import _env

    return _env("SUBJECT_MODEL_FILE")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m immich_face_api calibrate", description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", help="JSONL file of labelled photos (default: the store)")
    ap.add_argument("--export", help="write the labelled photos to this JSONL file and stop")
    ap.add_argument("--out", help="write the fitted model here (without it: only report the current model)")
    ap.add_argument("--model", help="the current model JSON (default FACE_API_SUBJECT_MODEL_FILE or the built-in)")
    ap.add_argument("--version", help="version name of the new model (default cal-<date-time>)")
    ap.add_argument("--folds", type=int, default=5, help="cross-validation folds (default 5)")
    ap.add_argument("--l2", type=float, default=1.0, help="L2 regularisation strength (default 1.0)")
    ap.add_argument("--force", action="store_true", help="fit even with few labels")
    return run(ap.parse_args(argv))
