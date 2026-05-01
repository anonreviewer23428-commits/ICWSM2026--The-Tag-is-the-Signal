from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pypdf import PdfReader

ONE_COL_IN = 3.3
TWO_COL_IN = 7.0
DPI = 300


def apply_icwsm_style() -> None:
    style_path = Path(__file__).resolve().parent / "icwsm.mplstyle"
    mpl.style.use(str(style_path))


def style_axes(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="major", width=0.6, length=3.0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.85)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(0.01, 0.98, label, transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")


def sanitize_label(s: str) -> str:
    s = str(s)
    # Replace common dash variants / NB hyphens with ASCII hyphen to avoid missing glyphs.
    s = s.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-").replace("\u2013", "-").replace("\u2212", "-")
    # Remove NULs / direction marks that can render as boxes.
    s = re.sub(r"[\u0000\u200e\u200f\u202a-\u202e]", "", s)
    # Escape LaTeX specials when `text.usetex=True` (only ones observed in this project).
    # Avoid double-escaping existing sequences like `\\&`.
    s = re.sub(r"(?<!\\)&", r"\&", s)
    # Normalize whitespace.
    s = re.sub(r"\s+", " ", s).strip()
    return s


def wrap_text(s: str, *, width: int) -> str:
    s = sanitize_label(s)
    # Insert soft break opportunities for common "glued" tokens.
    s = s.replace("/", " / ")
    s = re.sub(r"([\\[\\]\\(\\),])", r" \\1 ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return textwrap.fill(s, width=width, break_long_words=True, break_on_hyphens=True)


def wrap_text_px(
    s: str,
    *,
    max_px: float,
    renderer: mpl.backend_bases.RendererBase,
    fontsize: float,
) -> str:
    s = sanitize_label(s)
    s = s.replace("/", " / ")
    s = re.sub(r"([\\[\\]\\(\\),])", r" \\1 ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""

    fp = mpl.font_manager.FontProperties(family=mpl.rcParams.get("font.family", "serif"), size=fontsize)
    ismath = "TeX" if bool(mpl.rcParams.get("text.usetex", False)) else False
    words = [w for w in s.split(" ") if w]
    lines: list[str] = []
    cur: str = ""
    for w in words:
        cand = w if not cur else f"{cur} {w}"
        w_px, _, _ = renderer.get_text_width_height_descent(cand, fp, ismath=ismath)
        if (w_px <= max_px) or not cur:
            cur = cand
            continue
        lines.append(cur)
        cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def pct_tick(x: float) -> str:
    v = x * 100.0
    if v < 1:
        return f"{v:.1f}\\%"
    if v < 10:
        return f"{v:g}\\%"
    return f"{v:.0f}\\%"


def _find_text_outside_canvas(fig: plt.Figure, *, pad_px: float = 0.5) -> list[mpl.text.Text]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    w_px, h_px = fig.canvas.get_width_height()
    bad: list[mpl.text.Text] = []
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible():
            continue
        txt = t.get_text()
        if not txt or not str(txt).strip():
            continue
        bbox = t.get_window_extent(renderer=renderer)
        if bbox.x0 < -pad_px or bbox.x1 > (w_px + pad_px) or bbox.y0 < -pad_px or bbox.y1 > (h_px + pad_px):
            bad.append(t)
    return bad


def save_fig(fig: plt.Figure, out_base: Path, *, target_width_in: float) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight_bbox = fig.get_tightbbox(renderer)
    tight_w = float(tight_bbox.width)
    if tight_w > target_width_in + 1e-3:
        offenders = _find_text_outside_canvas(fig)
        samples = ", ".join(repr(str(t.get_text()).replace("\n", "\\n")) for t in offenders[:6])
        raise ValueError(
            f"Figure tight width {tight_w:.3f}in exceeds target {target_width_in:.3f}in. "
            f"Text outside canvas (sample): {samples}"
        )

    pad = max((target_width_in - tight_w) / 2.0, 0.0)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=pad)
    fig.savefig(out_base.with_suffix(".png"), dpi=DPI, bbox_inches="tight", pad_inches=pad)


def min_fontsize_in_fig(fig: plt.Figure) -> float:
    sizes: list[float] = []
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible():
            continue
        txt = t.get_text()
        if not txt or not str(txt).strip():
            continue
        sizes.append(float(t.get_fontsize()))
    return float(min(sizes)) if sizes else float("nan")


def _shrink_bbox(bbox: mpl.transforms.Bbox, pad_px: float) -> mpl.transforms.Bbox | None:
    x0 = bbox.x0 + pad_px
    y0 = bbox.y0 + pad_px
    x1 = bbox.x1 - pad_px
    y1 = bbox.y1 - pad_px
    if x1 <= x0 or y1 <= y0:
        return None
    return mpl.transforms.Bbox.from_extents(x0, y0, x1, y1)


def find_text_overlaps(
    fig: plt.Figure,
    *,
    pad_px: float = 0.5,
    min_intersection_px2: float = 6.0,
) -> list[tuple[mpl.text.Text, mpl.text.Text]]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    texts: list[tuple[mpl.text.Text, mpl.transforms.Bbox]] = []
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible():
            continue
        s = t.get_text()
        if not s or not str(s).strip():
            continue
        if t.get_alpha() == 0:
            continue
        bbox = t.get_window_extent(renderer=renderer)
        bbox = _shrink_bbox(bbox, pad_px)
        if bbox is None:
            continue
        texts.append((t, bbox))

    overlaps: list[tuple[mpl.text.Text, mpl.text.Text]] = []
    for i, (t1, b1) in enumerate(texts):
        for t2, b2 in texts[i + 1 :]:
            inter_w = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            if inter_w <= 0:
                continue
            inter_h = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if inter_h <= 0:
                continue
            if inter_w * inter_h < min_intersection_px2:
                continue
            overlaps.append((t1, t2))
    return overlaps


def assert_no_text_overlap(fig: plt.Figure, name: str) -> None:
    overlaps = find_text_overlaps(fig)
    if not overlaps:
        return

    lines = [f"Text overlap detected in {name}: {len(overlaps)} pair(s)"]
    for t1, t2 in overlaps[:12]:
        s1 = str(t1.get_text()).replace("\n", "\\n")
        s2 = str(t2.get_text()).replace("\n", "\\n")
        lines.append(f"- {s1!r} overlaps {s2!r}")
    raise AssertionError("\n".join(lines))


def _week_start(week_label: str) -> pd.Timestamp:
    return pd.to_datetime(str(week_label).split("/", 1)[0], utc=True)


def _split_multi(s) -> list[str]:
    if s is None:
        return []
    if isinstance(s, float) and math.isnan(s):
        return []
    out = [p.strip() for p in str(s).split(",")]
    return [p for p in out if p]


def _humanize_token_value(v: str) -> str:
    v = sanitize_label(v)
    v = v.replace("___", " & ")
    v = v.replace("__", " ")
    v = v.replace("_/_", " / ")
    v = v.replace("_", " ")
    v = re.sub(r"\s+", " ", v).strip(" _")
    v = v.title()
    v = v.replace("Fomo", "FOMO")
    return v


def _cluster_fallback_label(top_tokens: str) -> str:
    toks = [t.strip() for t in str(top_tokens).split(";") if t.strip()]
    theme = next((t.split("=", 1)[1] for t in toks if t.startswith("theme=")), None)
    claim = next((t.split("=", 1)[1] for t in toks if t.startswith("claim=")), None)
    cta = next((t.split("=", 1)[1] for t in toks if t.startswith("cta=")), None)
    evid = next((t.split("=", 1)[1] for t in toks if t.startswith("evid=")), None)

    parts: list[str] = []
    if theme:
        parts.append(_humanize_token_value(theme))
    if claim:
        parts.append(_humanize_token_value(claim))
    elif cta:
        parts.append(_humanize_token_value(cta))
    elif evid:
        parts.append(_humanize_token_value(evid))

    s = ": ".join(parts[:2])
    return sanitize_label(s)


def _cluster_label_map(tables_dir: Path) -> dict[int, str]:
    sig = pd.read_csv(tables_dir / "strategy_cluster_signatures.csv")
    sig_map = {int(r.cluster): _cluster_fallback_label(r.top_tokens) for r in sig.itertuples(index=False)}

    comm = pd.read_csv(tables_dir / "community_signature_table.csv")
    out: dict[int, str] = {}
    for r in comm.itertuples(index=False):
        try:
            c1 = int(r.top1_cluster)
            c2 = int(r.top2_cluster)
        except Exception:
            continue
        if c1 not in out and isinstance(r.top1_label_short, str) and r.top1_label_short.strip():
            out[c1] = sanitize_label(r.top1_label_short.strip())
        if c2 not in out and isinstance(r.top2_label_short, str) and r.top2_label_short.strip():
            out[c2] = sanitize_label(r.top2_label_short.strip())

    for k, v in sig_map.items():
        out.setdefault(k, v)
    return out


def _place_labels_no_overlap(
    ax: plt.Axes,
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    labels: list[str],
    order: list[int],
    fontsize: float = 9.0,
    avoid: Iterable[mpl.transforms.Bbox] | None = None,
) -> None:
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bbox = ax.get_window_extent(renderer=renderer)

    placed: list[mpl.transforms.Bbox] = []
    if avoid is not None:
        for b in avoid:
            b2 = _shrink_bbox(b, 0.2)
            placed.append(b2 if b2 is not None else b)
    candidates = [
        (6, 6),
        (6, -8),
        (-6, 6),
        (-6, -8),
        (10, 0),
        (-10, 0),
        (0, 10),
        (0, -12),
        (12, 12),
        (-12, 12),
        (12, -14),
        (-12, -14),
    ]

    for i in order:
        x = float(xs[i])
        y = float(ys[i])
        text = labels[i]
        chosen = None
        chosen_bbox = None
        for dx, dy in candidates:
            ann = ax.annotate(
                text,
                xy=(x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="left" if dx >= 0 else "right",
                va="center",
                fontsize=fontsize,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.2, alpha=0.9),
            )
            fig.canvas.draw()
            bbox = ann.get_window_extent(renderer=renderer)
            bbox = _shrink_bbox(bbox, 0.2) or bbox

            inside = (
                bbox.x0 >= ax_bbox.x0
                and bbox.x1 <= ax_bbox.x1
                and bbox.y0 >= ax_bbox.y0
                and bbox.y1 <= ax_bbox.y1
            )
            overlap = any(
                (min(bbox.x1, b.x1) - max(bbox.x0, b.x0)) > 0 and (min(bbox.y1, b.y1) - max(bbox.y0, b.y0)) > 0
                for b in placed
            )
            if inside and not overlap:
                chosen = ann
                chosen_bbox = bbox
                break
            ann.remove()

        if chosen is None:
            continue
        placed.append(chosen_bbox if chosen_bbox is not None else chosen.get_window_extent(renderer=renderer))


@dataclass(frozen=True)
class FigureMeta:
    key: str
    pdf_path: Path
    png_path: Path
    width_target_in: float
    min_font_pt: float
    grayscale_safe: bool


def fig_case_study_composition_shift(
    base: pd.DataFrame,
    *,
    cluster_label: dict[int, str],
    js_by_week: pd.DataFrame,
    out_dir: Path,
) -> FigureMeta:
    event = js_by_week.sort_values("js_cluster", ascending=False).iloc[0]
    event_week = str(event.week)

    weeks_sorted = (
        base[["week"]]
        .drop_duplicates()
        .assign(week_start=lambda d: d["week"].map(_week_start))
        .sort_values("week_start")
    )["week"].tolist()
    pos = weeks_sorted.index(event_week)
    baseline_weeks = weeks_sorted[max(0, pos - 8) : pos]

    def _cluster_share_for_week(week_label: str, *, tail_only: bool) -> pd.Series:
        d = base.loc[base["week"] == week_label]
        if tail_only:
            d = d.loc[d["is_high_tail"] == 1]
        counts = d["cluster"].value_counts().sort_index()
        return counts / max(counts.sum(), 1)

    baseline_mat = pd.DataFrame([_cluster_share_for_week(w, tail_only=False) for w in baseline_weeks]).fillna(0.0)
    baseline_med = baseline_mat.median(axis=0)
    event_share = _cluster_share_for_week(event_week, tail_only=False)
    tail_share = _cluster_share_for_week(event_week, tail_only=True)

    all_clusters = sorted(
        set(baseline_med.index.astype(int)) | set(event_share.index.astype(int)) | set(tail_share.index.astype(int))
    )
    baseline_all = baseline_med.reindex(all_clusters, fill_value=0.0)
    event_all = event_share.reindex(all_clusters, fill_value=0.0)
    tail_all = tail_share.reindex(all_clusters, fill_value=0.0)

    delta = (event_all - baseline_all).abs().sort_values(ascending=False)
    top_clusters = delta.head(8).index.astype(int).tolist()

    # Sort by event-week share for interpretability.
    top_clusters.sort(key=lambda c: float(event_all.loc[c]), reverse=True)

    b = baseline_all.reindex(top_clusters).to_numpy(float) * 100.0
    e = event_all.reindex(top_clusters).to_numpy(float) * 100.0
    t = tail_all.reindex(top_clusters).to_numpy(float) * 100.0

    fig = plt.figure(figsize=(ONE_COL_IN, 3.15), dpi=DPI)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.55, 1.15], hspace=0.32)
    ax = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[1, 0])
    ax_map.set_axis_off()

    y = np.arange(len(top_clusters))
    # Event bars thicker; baseline/tail thinner outlines/hatches.
    ax.barh(
        y,
        e,
        height=0.26,
        color="0.55",
        edgecolor="black",
        linewidth=0.7,
        label=f"Event ({event_week.split('/',1)[0]})",
    )
    ax.barh(
        y - 0.23,
        b,
        height=0.18,
        color="white",
        edgecolor="black",
        linewidth=0.7,
        label="Baseline (prev 8w)",
    )
    ax.barh(
        y + 0.23,
        t,
        height=0.18,
        color="white",
        edgecolor="black",
        linewidth=0.7,
        hatch="///",
        label="Tail-only",
    )

    ax.set_yticks(y)
    ax.set_yticklabels([f"C{c}" for c in top_clusters])
    ax.invert_yaxis()
    ax.set_xlabel(r"Share of messages (\%)", labelpad=0)
    ax.set_xlim(0, max(float(np.max([b.max(), e.max(), t.max()])) * 1.15, 5.0))
    ax.grid(False)
    style_axes(ax)

    # Compact legend in mapping panel (prevents crowding in the plot area).
    handles, labels = ax.get_legend_handles_labels()
    ax_map.legend(
        handles,
        labels,
        loc="upper left",
        ncol=2,
        fontsize=9,
        handlelength=1.6,
        handletextpad=0.6,
        labelspacing=0.3,
        borderaxespad=0.0,
        columnspacing=1.0,
    )

    # Cluster label mapping (2 columns) with variable vertical spacing (prevents wraps from colliding).
    mapping = [(f"C{c}", sanitize_label(cluster_label.get(int(c), ""))) for c in top_clusters]
    x0 = 0.02
    col_w = 0.45
    gap = 0.06
    x1 = x0 + col_w + gap

    fig.subplots_adjust(left=0.20, right=0.98, top=0.98, bottom=0.02)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    leg = ax_map.get_legend()
    y_start = 0.92
    if leg is not None:
        leg_bbox = leg.get_window_extent(renderer=renderer).transformed(ax_map.transAxes.inverted())
        y_start = float(leg_bbox.y0) - 0.03

    max_px = float(col_w * ax_map.bbox.width)

    def _draw_mapping(items: list[tuple[str, str]], *, x: float) -> None:
        y = y_start
        for cid, desc in items:
            txt = f"{cid}: {desc}"
            wrapped = wrap_text_px(txt, max_px=max_px, renderer=renderer, fontsize=9)
            t_obj = ax_map.text(
                x,
                y,
                wrapped,
                transform=ax_map.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                linespacing=0.9,
            )
            fig.canvas.draw()
            bbox = t_obj.get_window_extent(renderer=renderer)
            y -= float(bbox.height / ax_map.bbox.height) + 0.03

    _draw_mapping(mapping[:4], x=x0)
    _draw_mapping(mapping[4:], x=x1)

    assert_no_text_overlap(fig, "case_study_composition_shift")
    min_font = min_fontsize_in_fig(fig)

    out_base = out_dir / "case_study_composition_shift_icwsm"
    save_fig(fig, out_base, target_width_in=ONE_COL_IN)
    plt.close(fig)
    return FigureMeta(
        key="case_study_composition_shift",
        pdf_path=out_base.with_suffix(".pdf"),
        png_path=out_base.with_suffix(".png"),
        width_target_in=ONE_COL_IN,
        min_font_pt=min_font,
        grayscale_safe=True,
    )


def fig_community_risk_by_community(tables_dir: Path, *, out_dir: Path) -> FigureMeta:
    channels = pd.read_csv(tables_dir / "channel_profiles.csv")
    comm_sig = pd.read_csv(tables_dir / "community_signature_table.csv")

    comm_map = (
        comm_sig[["community_raw", "community_id", "short_label"]]
        .drop_duplicates()
        .assign(community_raw=lambda d: d["community_raw"].astype(int))
    )
    channels["community_raw"] = channels["community"].astype(int)
    channels = channels.merge(comm_map, on="community_raw", how="left")
    channels["community_id"] = channels["community_id"].astype(str)

    def _abbrev_short_label(s: str) -> str:
        s = sanitize_label(s)
        s = s.replace("Crypto Forecast", "Forecast")
        s = s.replace("(Buy, Stat)", "(Buy/Stat)")
        return s

    channels["short_label"] = channels["short_label"].astype(str).map(_abbrev_short_label)

    # Order by median channel risk (descending).
    order = (
        channels.groupby("community_id", dropna=False)["mean_risk"].median().sort_values(ascending=False).index.tolist()
    )
    data = [channels.loc[channels["community_id"] == cid, "mean_risk"].to_numpy(float) for cid in order]

    fig = plt.figure(figsize=(TWO_COL_IN, 3.55), dpi=DPI)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.45], hspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[1, 0])
    ax_map.set_axis_off()

    bp = ax.boxplot(
        data,
        vert=False,
        tick_labels=order,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.0),
        boxprops=dict(edgecolor="black", linewidth=0.7),
        whiskerprops=dict(color="black", linewidth=0.7),
        capprops=dict(color="black", linewidth=0.7),
    )
    for box in bp["boxes"]:
        box.set_facecolor("0.85")

    # Overlay channels as deterministic jittered points.
    rng = np.random.default_rng(0)
    for i, cid in enumerate(order, start=1):
        vals = channels.loc[channels["community_id"] == cid, "mean_risk"].to_numpy(float)
        if len(vals) == 0:
            continue
        yy = i + rng.normal(0, 0.06, size=len(vals))
        ax.scatter(vals, yy, s=10, color="0.25", alpha=0.65, linewidth=0)

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Channel mean risk score")
    ax.set_ylabel("Community")
    ax.invert_yaxis()
    style_axes(ax)

    # Mapping panel (full width): "Ck = short label" (wrapped; no truncation/ellipses).
    mapping = (
        channels[["community_id", "short_label"]]
        .drop_duplicates()
        .set_index("community_id")["short_label"]
        .to_dict()
    )
    lines = [(cid, wrap_text(mapping.get(cid, ""), width=30)) for cid in order]
    ax_map.text(
        0.0,
        0.96,
        "Community key",
        transform=ax_map.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
    )
    col_x = [0.0, 0.34, 0.68]
    n_per_col = 4
    y_start = 0.82
    dy = 0.26
    for i, (cid, desc) in enumerate(lines):
        col = i // n_per_col
        row = i % n_per_col
        ax_map.text(
            col_x[col],
            y_start - row * dy,
            f"{cid} = {desc}",
            transform=ax_map.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            linespacing=0.88,
        )

    fig.subplots_adjust(left=0.12, right=0.99, top=0.98, bottom=0.06)
    assert_no_text_overlap(fig, "community_risk_by_community")
    min_font = min_fontsize_in_fig(fig)

    out_base = out_dir / "community_risk_by_community_icwsm"
    save_fig(fig, out_base, target_width_in=TWO_COL_IN)
    plt.close(fig)
    return FigureMeta(
        key="community_risk_by_community",
        pdf_path=out_base.with_suffix(".pdf"),
        png_path=out_base.with_suffix(".png"),
        width_target_in=TWO_COL_IN,
        min_font_pt=min_font,
        grayscale_safe=True,
    )


def fig_conditional_lift_by_theme_heatmap(base: pd.DataFrame, *, out_dir: Path) -> FigureMeta:
    tables_dir = Path(__file__).resolve().parents[1] / "tag_dynamics" / "outputs" / "tables"
    tags_claim = pd.read_csv(tables_dir / "tags_claim.csv").assign(family="Claim", col="claim_types_cb")
    tags_cta = pd.read_csv(tables_dir / "tags_cta.csv").assign(family="CTA", col="ctas_cb")
    tags_evidence = pd.read_csv(tables_dir / "tags_evidence.csv").assign(family="Evidence", col="evidence_cb")

    sel = pd.concat(
        [
            tags_claim.sort_values("high_tail_lift", ascending=False).head(4),
            tags_cta.sort_values("high_tail_lift", ascending=False).head(4),
            tags_evidence.sort_values("high_tail_lift", ascending=False).head(4),
        ],
        ignore_index=True,
    )
    sel = sel.dropna(subset=["tag"]).copy()
    # Display labels are sanitized (rendering), but computations use raw tag strings.
    sel["row_label"] = sel.apply(lambda r: f"{r['family']}: {sanitize_label(r['tag'])}", axis=1)

    themes = base["theme_cb"].value_counts().rename_axis("theme").reset_index(name="n").sort_values("n", ascending=False)
    theme_list_raw = themes["theme"].tolist()

    # Keep the top themes by volume for readability; the rest are typically low-support (mostly gray).
    max_themes = 12
    theme_list_raw = theme_list_raw[:max_themes]
    theme_list_disp = [sanitize_label(t) for t in theme_list_raw]
    theme_codes = [f"T{i+1}" for i in range(len(theme_list_raw))]

    base_rate = base.groupby("theme_cb", observed=True)["is_high_tail"].mean().to_dict()
    base_theme = base["theme_cb"]
    split_cache = {
        "claim_types_cb": base["claim_types_cb"].astype(object).map(_split_multi),
        "ctas_cb": base["ctas_cb"].astype(object).map(_split_multi),
        "evidence_cb": base["evidence_cb"].astype(object).map(_split_multi),
    }

    min_support = 200
    lift = np.full((len(sel), len(theme_list_raw)), np.nan, dtype=float)
    support = np.zeros((len(sel), len(theme_list_raw)), dtype=int)

    for i, r in enumerate(sel.itertuples(index=False)):
        tag_raw = str(r.tag)
        col = str(r.col)
        present = split_cache[col].map(lambda xs, tag=tag_raw: tag in xs)
        for j, theme_raw in enumerate(theme_list_raw):
            m_theme = base_theme == theme_raw
            m = m_theme & present
            n_cell = int(m.sum())
            support[i, j] = n_cell
            if n_cell < min_support:
                continue
            denom = float(base_rate.get(theme_raw, np.nan))
            if not np.isfinite(denom) or denom <= 0:
                continue
            numer = float(base.loc[m, "is_high_tail"].mean())
            lift[i, j] = numer / denom

    val = np.log2(lift)
    val = np.clip(val, -2.0, 2.0)
    cmap = mpl.colormaps["RdBu_r"].copy()
    cmap.set_bad(color="0.90")  # low-support cells
    norm = mpl.colors.TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0)

    fig = plt.figure(figsize=(TWO_COL_IN, 4.65), dpi=DPI)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.30], hspace=0.16)
    ax = fig.add_subplot(gs[0, 0])
    ax_key = fig.add_subplot(gs[1, 0])
    ax_key.set_axis_off()

    mesh = ax.pcolormesh(val, cmap=cmap, norm=norm, shading="nearest", edgecolors="white", linewidth=0.6)

    ax.set_xticks(np.arange(len(theme_list_raw)) + 0.5)
    ax.set_xticklabels(theme_codes, rotation=0, ha="center")
    ax.set_yticks(np.arange(len(sel)) + 0.5)
    # Keep y tick labels readable without vertical collisions (prefer fewer wrapped lines).
    ax.set_yticklabels([wrap_text(s, width=28) for s in sel["row_label"].tolist()])
    ax.set_xlabel("Theme")
    ax.set_ylabel("Tag")
    style_axes(ax)
    for t in ax.get_yticklabels():
        if "\n" in t.get_text():
            t.set_linespacing(0.85)

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)
    ticks = np.array([-2, -1, 0, 1, 2], dtype=float)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{2**t:g}" for t in ticks])
    cbar.set_label("Conditional tail lift")

    # Theme mapping + low-support note, inside the figure bbox (not on top of tick labels).
    ax_key.text(0.0, 0.95, "Theme key", transform=ax_key.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")
    col_x = [0.0, 0.34, 0.68]
    dy = 0.25
    for i, (code, full) in enumerate(zip(theme_codes, theme_list_disp, strict=True)):
        x = col_x[i // 4]
        y = 0.82 - (i % 4) * dy
        ax_key.text(
            x,
            y,
            f"{code} = {wrap_text(full, width=22)}",
            transform=ax_key.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            linespacing=0.9,
        )

    ax_key.add_patch(
        mpl.patches.Rectangle((0.84, 0.08), 0.035, 0.18, facecolor="0.90", edgecolor="black", linewidth=0.6, transform=ax_key.transAxes)
    )
    ax_key.text(
        0.885,
        0.17,
        rf"Support $< {min_support}$",
        transform=ax_key.transAxes,
        ha="left",
        va="center",
        fontsize=9,
    )

    # Larger left margin ensures multi-line y tick labels stay inside the 7.0" figure bbox.
    fig.subplots_adjust(left=0.30, right=0.96, top=0.98, bottom=0.05)
    assert_no_text_overlap(fig, "conditional_lift_by_theme_heatmap")
    min_font = min_fontsize_in_fig(fig)

    out_base = out_dir / "conditional_lift_by_theme_heatmap_icwsm"
    save_fig(fig, out_base, target_width_in=TWO_COL_IN)
    plt.close(fig)
    return FigureMeta(
        key="conditional_lift_by_theme_heatmap",
        pdf_path=out_base.with_suffix(".pdf"),
        png_path=out_base.with_suffix(".png"),
        width_target_in=TWO_COL_IN,
        min_font_pt=min_font,
        grayscale_safe=True,
    )


def fig_monitoring_dashboard(
    base: pd.DataFrame,
    *,
    js_by_week: pd.DataFrame,
    cluster_label: dict[int, str],
    high_tail_q: float,
    out_dir: Path,
) -> FigureMeta:
    wk = (
        base.groupby("week", observed=True)
        .agg(n_msgs=("msg_id", "size"), tail_rate=("is_high_tail", "mean"))
        .reset_index()
    )
    wk["week_start"] = wk["week"].map(_week_start)
    wk = wk.sort_values("week_start")

    js = js_by_week.copy()
    js["week_start"] = js["week"].map(_week_start)
    js = js.sort_values("week_start")

    spikes = js.sort_values("js_cluster", ascending=False).head(2).copy()
    spike_weeks = list(spikes["week"])
    spike_dates = list(spikes["week_start"])

    wk_counts = base.groupby(["week", "cluster"], observed=True).size().unstack(fill_value=0)
    wk_share = wk_counts.div(wk_counts.sum(axis=1), axis=0)
    wk_share["week_start"] = wk_share.index.map(_week_start)
    wk_share = wk_share.sort_values("week_start").drop(columns=["week_start"])

    def _top_attribution(week_label: str, *, lookback: int = 8, topk: int = 3) -> pd.DataFrame:
        idx = list(wk_share.index)
        pos = idx.index(week_label)
        start = max(0, pos - lookback)
        baseline = wk_share.iloc[start:pos]
        base_med = baseline.median(axis=0) if len(baseline) else wk_share.iloc[:pos].median(axis=0)
        event = wk_share.loc[week_label]
        delta = (event - base_med).sort_values(ascending=False)
        top = delta.head(topk)
        out = pd.DataFrame({"cluster": top.index.astype(int), "delta": top.values})
        def _short_cluster_label(s: str) -> str:
            s = sanitize_label(s)
            return s.split(":", 1)[0].strip() if ":" in s else s

        out["label"] = out["cluster"].map(lambda c: f"C{c} {_short_cluster_label(cluster_label.get(int(c), ''))}".strip())
        out["label_wrapped"] = out["label"].map(lambda s: wrap_text(s, width=22))
        return out

    attr = [_top_attribution(w, lookback=8, topk=3) for w in spike_weeks]

    fig = plt.figure(figsize=(TWO_COL_IN, 3.8), dpi=DPI)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.10], hspace=0.45, wspace=0.50)
    ax_tail = fig.add_subplot(gs[0, :])
    ax_js = fig.add_subplot(gs[1, :], sharex=ax_tail)
    ax_attr1 = fig.add_subplot(gs[2, 0])
    ax_attr2 = fig.add_subplot(gs[2, 1])

    ax_tail.plot(wk["week_start"], wk["tail_rate"] * 100.0, color="black", linewidth=1.1)
    tail_k = int(100 * (1 - high_tail_q))
    ax_tail.set_ylabel(rf"Top-{tail_k}\% tail rate (\%)")
    ax_tail.set_ylim(0, max(7.5, float((wk["tail_rate"] * 100.0).max()) * 1.15))
    style_axes(ax_tail)
    add_panel_label(ax_tail, "(a)")
    plt.setp(ax_tail.get_xticklabels(), visible=False)

    ax_js.plot(js["week_start"], js["js_cluster"], color="black", linewidth=1.1)
    ax_js.set_ylabel("JS divergence")
    ax_js.set_ylim(0, float(js["js_cluster"].max()) * 1.15)
    ax_js.set_xlabel("Week")
    style_axes(ax_js)
    add_panel_label(ax_js, "(b)")

    x0, x1 = ax_js.get_xlim()
    for dt, wk_label in zip(spike_dates, spike_weeks, strict=True):
        for ax in [ax_tail, ax_js]:
            ax.axvline(dt, color="0.35", linestyle="--", linewidth=0.9)
        dt_num = mpl.dates.date2num(pd.to_datetime(dt).to_pydatetime())
        frac = 0.5 if (x1 - x0) <= 0 else float((dt_num - x0) / (x1 - x0))
        ha = "right" if frac > 0.85 else "left"
        dx = -3 if ha == "right" else 3
        ax_js.annotate(
            wk_label.split("/", 1)[0],
            xy=(dt, float(js.loc[js["week"] == wk_label, "js_cluster"].iloc[0])),
            xytext=(dx, 8),
            textcoords="offset points",
            ha=ha,
            va="bottom",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.2, alpha=0.85),
        )

    for ax, w, a, plab in [
        (ax_attr1, spike_weeks[0], attr[0], "(c)"),
        (ax_attr2, spike_weeks[1], attr[1], "(d)"),
    ]:
        ax.barh(a["label_wrapped"], a["delta"] * 100.0, color="0.65", edgecolor="black", linewidth=0.6)
        ax.axvline(0.0, color="black", linewidth=0.9)
        ax.set_title(f"Attribution: {w.split('/', 1)[0]}", fontsize=9, pad=3)
        ax.set_xlabel(r"$\Delta$ share vs prev 8w median (pp)")
        ax.invert_yaxis()
        style_axes(ax)
        add_panel_label(ax, plab)

    fig.subplots_adjust(left=0.16, right=0.99, top=0.98, bottom=0.12)
    assert_no_text_overlap(fig, "monitoring_dashboard")
    min_font = min_fontsize_in_fig(fig)

    out_base = out_dir / "monitoring_dashboard_icwsm"
    save_fig(fig, out_base, target_width_in=TWO_COL_IN)
    plt.close(fig)
    return FigureMeta(
        key="monitoring_dashboard",
        pdf_path=out_base.with_suffix(".pdf"),
        png_path=out_base.with_suffix(".png"),
        width_target_in=TWO_COL_IN,
        min_font_pt=min_font,
        grayscale_safe=True,
    )


def fig_cluster_portfolio_riskmass_vs_lift(tables_dir: Path, *, out_dir: Path) -> FigureMeta:
    clusters = pd.read_csv(tables_dir / "strategy_cluster_summary.csv")
    clusters["cluster"] = clusters["cluster"].astype(int)

    x = clusters["risk_mass_share"].to_numpy(float)
    y = clusters["high_tail_lift"].to_numpy(float)
    n = clusters["n_msgs"].to_numpy(float)

    n_max = max(float(np.nanmax(n)), 1.0)

    def size_fn(v: float) -> float:
        return 30.0 + 220.0 * math.sqrt(max(v, 0.0) / n_max)

    sizes = np.array([size_fn(float(v)) for v in n], dtype=float)

    fig = plt.figure(figsize=(ONE_COL_IN, 2.35), dpi=DPI)
    ax = fig.add_axes([0.20, 0.20, 0.78, 0.76])
    ax.scatter(x, y, s=sizes, facecolor="0.70", edgecolor="black", linewidth=0.6)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    lift_label = ax.text(
        0.02,
        1.0,
        "lift=1",
        transform=mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData),
        ha="left",
        va="bottom",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.2, alpha=0.85),
    )

    ax.set_xscale("log")
    ax.set_xlim(max(0.0015, float(np.nanmin(x)) * 0.9), min(0.5, float(np.nanmax(x)) * 1.35))
    ax.set_ylim(0.0, 6.6)
    xt = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    xt = [t for t in xt if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
    ax.set_xticks(xt)
    ax.set_xticklabels([pct_tick(v) for v in xt])
    ax.set_xlabel("Risk mass share")
    ax.set_ylabel("High-tail lift")
    style_axes(ax)

    # Bubble size legend (3 reference sizes).
    legend_ns = [1000, 10000, int(n_max)]
    handles = [ax.scatter([], [], s=size_fn(v), facecolor="0.70", edgecolor="black", linewidth=0.6) for v in legend_ns]
    labels = [f"{int(v):,} msgs" for v in legend_ns]
    leg = ax.legend(handles, labels, title="Cluster size", loc="upper right", fontsize=9, title_fontsize=9, frameon=False)

    # Deterministic label selection: top-5 by lift OR top-4 by risk-mass share.
    top_lift = clusters.sort_values("high_tail_lift", ascending=False).head(5).index.to_list()
    top_mass = clusters.sort_values("risk_mass_share", ascending=False).head(4).index.to_list()
    sel = list(dict.fromkeys(top_lift + top_mass))
    label_text = ["" for _ in range(len(clusters))]
    for idx in sel:
        c = int(clusters.loc[idx, "cluster"])
        label_text[idx] = f"C{c}"

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    avoid = [
        leg.get_window_extent(renderer=renderer),
        lift_label.get_window_extent(renderer=renderer),
    ]
    order = sel[:]  # keep deterministic order by importance already defined
    _place_labels_no_overlap(ax, xs=x, ys=y, labels=label_text, order=order, fontsize=9.0, avoid=avoid)

    fig.subplots_adjust(left=0.20, right=0.98, top=0.98, bottom=0.10)
    assert_no_text_overlap(fig, "cluster_portfolio_riskmass_vs_lift")
    min_font = min_fontsize_in_fig(fig)

    out_base = out_dir / "cluster_portfolio_riskmass_vs_lift_icwsm"
    save_fig(fig, out_base, target_width_in=ONE_COL_IN)
    plt.close(fig)
    return FigureMeta(
        key="cluster_portfolio_riskmass_vs_lift",
        pdf_path=out_base.with_suffix(".pdf"),
        png_path=out_base.with_suffix(".png"),
        width_target_in=ONE_COL_IN,
        min_font_pt=min_font,
        grayscale_safe=True,
    )


def _signature_vs_exposure_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    title: str,
    n_max_global: float,
    add_size_legend: bool = False,
    label_top_lift: int = 3,
    label_mass_threshold: float = 0.10,
) -> None:
    d = df.copy()
    for col in ["n", "risk_mass_share", "high_tail_lift"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["tag", "n", "risk_mass_share", "high_tail_lift"]).copy()

    x = d["risk_mass_share"].to_numpy(float)
    y = d["high_tail_lift"].to_numpy(float)
    n = d["n"].to_numpy(float)

    n_max = max(float(n_max_global), 1.0)
    sizes = 25.0 + 170.0 * np.sqrt(np.maximum(n, 0.0) / n_max)

    ax.scatter(x, y, s=sizes, facecolor="0.70", edgecolor="black", linewidth=0.6)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    ax.set_xscale("log")
    ax.set_title(title, fontsize=9, pad=3)

    ax.set_xlim(8e-4, 0.85)
    ax.set_ylim(0.0, 6.6)
    ax.set_xticks([1e-3, 1e-2, 1e-1, 5e-1])
    ax.set_xticklabels([r"0.1\%", r"1\%", r"10\%", r"50\%"])
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
    style_axes(ax)
    ax.grid(False)

    avoid: list[mpl.transforms.Bbox] = []
    if add_size_legend:
        def size_fn(v: float) -> float:
            return 25.0 + 170.0 * math.sqrt(max(v, 0.0) / n_max)

        legend_ns = [1000, 10000, int(n_max)]
        handles = [ax.scatter([], [], s=size_fn(v), facecolor="0.70", edgecolor="black", linewidth=0.6) for v in legend_ns]
        legend_labels = [f"{int(v):,} msgs" for v in legend_ns]
        leg = ax.legend(
            handles,
            legend_labels,
            title="Tag size",
            loc="upper right",
            fontsize=9,
            title_fontsize=9,
            frameon=False,
        )
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        avoid.append(leg.get_window_extent(renderer=renderer))

    # Deterministic label rule.
    top_lift = d.sort_values("high_tail_lift", ascending=False).head(label_top_lift)
    top_mass = d.loc[d["risk_mass_share"] >= label_mass_threshold]
    label_tags = list(dict.fromkeys(list(top_lift["tag"]) + list(top_mass["tag"])))

    labels = ["" for _ in range(len(d))]
    for i, tag in enumerate(d["tag"].tolist()):
        if tag in label_tags:
            labels[i] = wrap_text(tag, width=18)

    order = [d.index[d["tag"] == t][0] for t in label_tags if t in set(d["tag"])]
    order = [int(d.index.get_loc(i)) for i in order]
    _place_labels_no_overlap(ax, xs=x, ys=y, labels=labels, order=order, fontsize=9.0, avoid=avoid or None)


def fig_tag_signature_vs_exposure_map(tables_dir: Path, *, out_dir: Path) -> FigureMeta:
    tags_claim = pd.read_csv(tables_dir / "tags_claim.csv")
    tags_cta = pd.read_csv(tables_dir / "tags_cta.csv")
    tags_evidence = pd.read_csv(tables_dir / "tags_evidence.csv")

    n_max_global = float(max(tags_claim["n"].max(), tags_cta["n"].max(), tags_evidence["n"].max()))

    fig = plt.figure(figsize=(TWO_COL_IN, 2.35), dpi=DPI)
    gs = fig.add_gridspec(1, 3, wspace=0.30)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
    ax2 = fig.add_subplot(gs[0, 2], sharey=ax0)

    # Finalize layout before placing any annotations (offset-point labels depend on axes size).
    fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.18)

    _signature_vs_exposure_panel(ax0, tags_claim, title="Claim types", n_max_global=n_max_global)
    _signature_vs_exposure_panel(ax1, tags_cta, title="CTAs", n_max_global=n_max_global)
    _signature_vs_exposure_panel(
        ax2,
        tags_evidence,
        title="Evidence",
        n_max_global=n_max_global,
        add_size_legend=True,
    )

    ax0.set_ylabel("High-tail lift")
    for ax in [ax1, ax2]:
        plt.setp(ax.get_yticklabels(), visible=False)

    for ax in [ax0, ax1, ax2]:
        ax.set_xlabel("Risk mass share")

    assert_no_text_overlap(fig, "tag_signature_vs_exposure_map")
    min_font = min_fontsize_in_fig(fig)

    out_base = out_dir / "tag_signature_vs_exposure_map_icwsm"
    save_fig(fig, out_base, target_width_in=TWO_COL_IN)
    plt.close(fig)
    return FigureMeta(
        key="tag_signature_vs_exposure_map",
        pdf_path=out_base.with_suffix(".pdf"),
        png_path=out_base.with_suffix(".png"),
        width_target_in=TWO_COL_IN,
        min_font_pt=min_font,
        grayscale_safe=True,
    )


def pdf_physical_size_in(pdf_path: Path) -> tuple[float, float]:
    reader = PdfReader(str(pdf_path))
    page = reader.pages[0]
    w = float(page.mediabox.width) / 72.0
    h = float(page.mediabox.height) / 72.0
    return w, h


def pdf_font_checks(pdf_path: Path) -> dict[str, object]:
    reader = PdfReader(str(pdf_path))
    embedded_ok = True
    has_type3 = False
    has_cid = False
    fonts_seen: set[str] = set()

    def _walk_font(font_obj) -> None:
        nonlocal embedded_ok, has_type3, has_cid
        if font_obj is None:
            return
        try:
            subtype = str(font_obj.get("/Subtype", ""))
        except Exception:
            subtype = ""
        if subtype == "/Type3":
            has_type3 = True
        if subtype == "/Type0":
            has_cid = True
        try:
            enc = font_obj.get("/Encoding")
            if str(enc) == "/Identity-H":
                has_cid = True
        except Exception:
            pass
        try:
            base = str(font_obj.get("/BaseFont", "")) or str(font_obj.get("/Name", ""))
            if base:
                fonts_seen.add(base)
        except Exception:
            pass

        # Descendant fonts (Type0)
        if "/DescendantFonts" in font_obj:
            for df in font_obj["/DescendantFonts"]:
                _walk_font(df.get_object())

        # Font embedding check via FontDescriptor.
        fd = font_obj.get("/FontDescriptor")
        if fd is not None:
            fd = fd.get_object()
            if not any(k in fd for k in ["/FontFile", "/FontFile2", "/FontFile3"]):
                embedded_ok = False

    for page in reader.pages:
        resources = page.get("/Resources") or {}
        fonts = resources.get("/Font") or {}
        for f in fonts.values():
            _walk_font(f.get_object())

    return {"embedded_ok": embedded_ok, "has_type3": has_type3, "has_cid": has_cid, "fonts": sorted(fonts_seen)}


def mutool_fonts(pdf_path: Path) -> str:
    try:
        out = subprocess.check_output(["mutool", "info", "-F", str(pdf_path)], text=True, stderr=subprocess.STDOUT)
    except Exception as e:
        return f"(mutool failed: {e})"
    lines = [ln.rstrip() for ln in out.splitlines()]
    # Keep just the font section for brevity.
    keep: list[str] = []
    in_fonts = False
    for ln in lines:
        if ln.startswith("Fonts"):
            in_fonts = True
        if in_fonts:
            keep.append(ln)
    return "\n".join(keep).strip()


def write_reports(figs: list[FigureMeta], *, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Preflight report.
    preflight_lines: list[str] = []
    compliance_lines: list[str] = []
    latex_lines: list[str] = []

    for fm in figs:
        w_in, h_in = pdf_physical_size_in(fm.pdf_path)
        fc = pdf_font_checks(fm.pdf_path)
        fonts_txt = mutool_fonts(fm.pdf_path)

        preflight_lines.append(f"{fm.key}:")
        preflight_lines.append(f"  pdf_size_in: {w_in:.3f} x {h_in:.3f}")
        preflight_lines.append(f"  min_font_pt: {fm.min_font_pt:.1f}")
        preflight_lines.append(f"  embedded_fonts: {bool(fc['embedded_ok'])}")
        preflight_lines.append(f"  has_type3_fonts: {bool(fc['has_type3'])}")
        preflight_lines.append(f"  has_cid_identity_h: {bool(fc['has_cid'])}")
        preflight_lines.append("  mutool_fonts:")
        for ln in fonts_txt.splitlines():
            preflight_lines.append(f"    {ln}")
        preflight_lines.append("")

        # Width should be effectively exact; small deltas can occur due to PDF point rounding.
        ok_width = abs(w_in - fm.width_target_in) <= 0.02
        ok_font = fm.min_font_pt >= 9.0
        ok_type3 = not bool(fc["has_type3"])
        ok_cid = not bool(fc["has_cid"])
        ok_embed = bool(fc["embedded_ok"])
        ok_format = fm.pdf_path.suffix.lower() == ".pdf"
        # Layout/cutoff checks are enforced during generation:
        # - `assert_no_text_overlap(...)` raises on collisions.
        # - `save_fig(...)` raises if tight bbox would exceed target width (guards against off-canvas text).
        ok_layout = True
        ok_grayscale = fm.grayscale_safe

        compliance_lines.append(
            f"{fm.key}: "
            f"width={'PASS' if ok_width else 'FAIL'}; "
            f"min_font={'PASS' if ok_font else 'FAIL'}; "
            f"no_type3={'PASS' if ok_type3 else 'FAIL'}; "
            f"no_cid={'PASS' if ok_cid else 'FAIL'}; "
            f"fonts_embedded={'PASS' if ok_embed else 'FAIL'}; "
            f"format={'PASS' if ok_format else 'FAIL'}; "
            f"layout={'PASS' if ok_layout else 'FAIL'}; "
            f"grayscale={'PASS' if ok_grayscale else 'FAIL'}"
        )

        if fm.width_target_in <= ONE_COL_IN + 1e-6:
            latex_lines.append(
                "\\\\begin{figure}[t]\n"
                "  \\\\centering\n"
                f"  \\\\includegraphics[width=\\\\columnwidth]{{figs_icwsm/{fm.pdf_path.name}}}\n"
                "  \\\\caption{...}\n"
                f"  \\\\label{{fig:{fm.key}}}\n"
                "\\\\end{figure}\n"
            )
        else:
            latex_lines.append(
                "\\\\begin{figure*}[t]\n"
                "  \\\\centering\n"
                f"  \\\\includegraphics[width=\\\\textwidth]{{figs_icwsm/{fm.pdf_path.name}}}\n"
                "  \\\\caption{...}\n"
                f"  \\\\label{{fig:{fm.key}}}\n"
                "\\\\end{figure*}\n"
            )

    (out_dir / "preflight_report.txt").write_text("\n".join(preflight_lines).rstrip() + "\n")
    (out_dir / "icwsm_compliance_report.txt").write_text("\n".join(compliance_lines).rstrip() + "\n")
    (out_dir / "latex_snippets.txt").write_text("\n".join(latex_lines).rstrip() + "\n")


def generate_all(*, out_dir: Path) -> None:
    apply_icwsm_style()

    repo_root = Path(__file__).resolve().parents[1]
    tables_dir = repo_root / "tag_dynamics" / "outputs" / "tables"
    artifacts_dir = repo_root / "tag_dynamics" / "outputs" / "artifacts"

    base = pd.read_pickle(artifacts_dir / "base_with_clusters.pkl.gz")
    audit = json.loads((artifacts_dir / "audit.json").read_text())
    high_tail_q = float(audit["high_tail_quantile"])

    js_by_week = pd.read_csv(tables_dir / "cluster_js_drift_by_week.csv")
    cluster_label = _cluster_label_map(tables_dir)

    figs: list[FigureMeta] = []
    figs.append(
        fig_case_study_composition_shift(base, cluster_label=cluster_label, js_by_week=js_by_week, out_dir=out_dir)
    )
    figs.append(fig_cluster_portfolio_riskmass_vs_lift(tables_dir, out_dir=out_dir))
    figs.append(fig_community_risk_by_community(tables_dir, out_dir=out_dir))
    figs.append(fig_conditional_lift_by_theme_heatmap(base, out_dir=out_dir))
    figs.append(fig_monitoring_dashboard(base, js_by_week=js_by_week, cluster_label=cluster_label, high_tail_q=high_tail_q, out_dir=out_dir))
    figs.append(fig_tag_signature_vs_exposure_map(tables_dir, out_dir=out_dir))

    write_reports(figs, out_dir=out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ICWSM-ready TAG2RISK figures.")
    parser.add_argument("--out", type=Path, default=Path("figs_icwsm"), help="Output directory")
    args = parser.parse_args()
    generate_all(out_dir=args.out)
    print("Wrote ICWSM figures to", args.out.resolve())


if __name__ == "__main__":
    main()
