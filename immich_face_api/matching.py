"""Turn nearest-neighbour rows into one decision per face (pure logic, no I/O).

Decision rule ("nearest face wins"):
  1. Take the k nearest **named** faces of the owner (exact cosine distance, see db.py).
  2. Drop every neighbour farther than ``max_distance`` (Immich's own facial-recognition maxDistance).
  3. Group the rest by person. A person's score is their **best (smallest) distance**; ties are broken by the
     number of neighbours (votes) they have among the k.
  4. The best person is the match; the next ones are the candidates.

Why not a majority vote over the k neighbours: Immich assigns a new face to the person of its nearest matching
face, so this mirrors what curators already see in Immich. A vote favours people with many photos (a member who
appears in thousands of photos fills the k slots even when someone else is closer) and punishes people who
have just been named with a handful of faces.

``ambiguous`` is set when the runner-up's best distance is within ``ambiguity_margin`` of the winner's: the
website should then show the candidates instead of pre-selecting one.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Neighbour:
    person_id: str
    name: str
    distance: float


@dataclass(frozen=True)
class Candidate:
    person_id: str
    name: str | None
    distance: float
    votes: int

    @property
    def confidence(self) -> float:
        """Cosine similarity (1 - cosine distance), clamped to [0, 1]. A hint for staff, not a probability."""
        return round(max(0.0, min(1.0, 1.0 - self.distance)), 4)


@dataclass(frozen=True)
class Decision:
    best: Candidate | None
    candidates: list[Candidate]
    ambiguous: bool


def decide(
    neighbours: list[Neighbour], *, max_distance: float, max_candidates: int, ambiguity_margin: float
) -> Decision:
    by_person: dict[str, list[Neighbour]] = {}
    for n in neighbours:
        if n.distance <= max_distance:
            by_person.setdefault(n.person_id, []).append(n)
    ranked = sorted(
        (
            Candidate(
                person_id=pid,
                name=(rows[0].name or None),
                distance=round(min(r.distance for r in rows), 4),
                votes=len(rows),
            )
            for pid, rows in by_person.items()
        ),
        key=lambda c: (c.distance, -c.votes, c.person_id),
    )
    if not ranked:
        return Decision(best=None, candidates=[], ambiguous=False)
    ambiguous = len(ranked) > 1 and (ranked[1].distance - ranked[0].distance) <= ambiguity_margin
    return Decision(best=ranked[0], candidates=ranked[:max_candidates], ambiguous=ambiguous)


def normalise_box(box: dict, sent_w: int, sent_h: int) -> dict:
    """ML box (x1,y1,x2,y2 in pixels of the image it got) -> {x,y,w,h} as fractions 0..1, clamped."""

    def clamp(v: float) -> float:
        return max(0.0, min(1.0, v))

    x1 = clamp(float(box["x1"]) / sent_w)
    y1 = clamp(float(box["y1"]) / sent_h)
    x2 = clamp(float(box["x2"]) / sent_w)
    y2 = clamp(float(box["y2"]) / sent_h)
    return {
        "x": round(x1, 5),
        "y": round(y1, 5),
        "w": round(max(0.0, x2 - x1), 5),
        "h": round(max(0.0, y2 - y1), 5),
    }
