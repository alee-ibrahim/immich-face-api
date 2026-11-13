"""Tiny server-side SVG charts for the admin UI (no JavaScript, no CDN).

One series per chart (so no legend; the heading names it), colours from CSS variables (light and dark themes),
thin marks with 4px rounded data ends on bars, 2px lines, a recessive grid, a hover tooltip per bucket through a
full-height transparent hit area with a <title>, and only a few axis labels. Every chart has a table view next to it
on the page.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from html import escape

W, H = 420, 190
PAD_L, PAD_R, PAD_T, PAD_B = 46, 6, 10, 26


def _nice_max(v: float) -> float:
    if v <= 0:
        return 1.0
    exp = 10 ** math.floor(math.log10(v))
    for m in (1, 2, 2.5, 5, 10):
        if v <= m * exp:
            return m * exp
    return 10 * exp


def _fmt_num(v: float) -> str:
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if v >= 1000:
        return f"{v / 1000:.1f}k".replace(".0k", "k")
    return f"{v:g}"


def _frame(
    labels: Sequence[str], y_max: float, title: str, tick_fmt: Callable[[float], str]
) -> tuple[list[str], float]:
    n = max(len(labels), 1)
    plot_w = W - PAD_L - PAD_R
    step = plot_w / n
    parts = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="{escape(title)}">']
    for frac in (0, 0.5, 1):
        y = PAD_T + (H - PAD_T - PAD_B) * (1 - frac)
        parts.append(f'<line class="grid" x1="{PAD_L}" x2="{W - PAD_R}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(
            f'<text class="tick" x="{PAD_L - 6}" y="{y + 4:.1f}" text-anchor="end">{escape(tick_fmt(y_max * frac))}</text>'
        )
    every = max(1, math.ceil(n / 5))
    for i, label in enumerate(labels):
        last_regular = (n - 1) // every * every
        if i % every == 0 or (i == n - 1 and n - 1 - last_regular >= max(2, every // 2 + 1)):
            x = PAD_L + step * (i + 0.5)
            parts.append(f'<text class="tick" x="{x:.1f}" y="{H - 6}" text-anchor="middle">{escape(label)}</text>')
    return parts, step


def bar_chart(labels: Sequence[str], values: Sequence[float], *, unit: str, title: str) -> str:
    y_max = _nice_max(max(values, default=0))
    parts, step = _frame(labels, y_max, title, _fmt_num)
    base = H - PAD_B
    plot_h = H - PAD_T - PAD_B
    bar_w = max(1.0, min(step - 2, 18))  # 2px surface gap between bars
    for i, (label, v) in enumerate(zip(labels, values, strict=True)):
        x0 = PAD_L + step * i
        cx = x0 + step / 2
        tip = f"{label}: {v:,.0f} {unit}"
        parts.append(f'<g class="hit"><title>{escape(tip)}</title>')
        parts.append(f'<rect class="hit-area" x="{x0:.1f}" y="{PAD_T}" width="{step:.1f}" height="{plot_h}"/>')
        if v > 0:
            h = max(2.0, plot_h * v / y_max)
            r = min(4.0, bar_w / 2, h)
            x, top = cx - bar_w / 2, base - h
            parts.append(
                f'<path class="bar" d="M{x:.1f},{base} V{top + r:.1f} Q{x:.1f},{top:.1f} {x + r:.1f},{top:.1f} '
                f'H{x + bar_w - r:.1f} Q{x + bar_w:.1f},{top:.1f} {x + bar_w:.1f},{top + r:.1f} V{base} Z"/>'
            )
        parts.append("</g>")
    parts.append(f'<line class="axis" x1="{PAD_L}" x2="{W - PAD_R}" y1="{base}" y2="{base}"/></svg>')
    return "".join(parts)


def line_chart(
    labels: Sequence[str],
    values: Sequence[float | None],
    *,
    unit: str,
    title: str,
    fmt: Callable[[float], str],
    y_max: float | None = None,
) -> str:
    present = [v for v in values if v is not None]
    top = y_max or _nice_max(max(present, default=0))
    parts, step = _frame(labels, top, title, (lambda v: fmt(v)) if present else _fmt_num)
    base = H - PAD_B
    plot_h = H - PAD_T - PAD_B

    def xy(i: int, v: float) -> tuple[float, float]:
        return PAD_L + step * (i + 0.5), base - plot_h * min(v, top) / top

    runs: list[list[tuple[float, float]]] = [[]]
    for i, v in enumerate(values):
        if v is None:
            if runs[-1]:
                runs.append([])
            continue
        runs[-1].append(xy(i, v))
    for run in runs:
        if len(run) >= 2:
            d = " ".join(f"{'M' if j == 0 else 'L'}{x:.1f},{y:.1f}" for j, (x, y) in enumerate(run))
            parts.append(f'<path class="line" d="{d}"/>')
        elif len(run) == 1:
            parts.append(f'<circle class="dot" cx="{run[0][0]:.1f}" cy="{run[0][1]:.1f}" r="4"/>')
    for i, (label, v) in enumerate(zip(labels, values, strict=True)):
        x0 = PAD_L + step * i
        tip = f"{label}: {fmt(v) if v is not None else 'no data'}"
        parts.append(f'<g class="hit"><title>{escape(tip)}</title>')
        parts.append(f'<rect class="hit-area" x="{x0:.1f}" y="{PAD_T}" width="{step:.1f}" height="{plot_h}"/>')
        if v is not None:
            x, y = xy(i, v)
            parts.append(f'<circle class="marker" cx="{x:.1f}" cy="{y:.1f}" r="4"/>')
        parts.append("</g>")
    parts.append(f'<line class="axis" x1="{PAD_L}" x2="{W - PAD_R}" y1="{base}" y2="{base}"/></svg>')
    return "".join(parts)
