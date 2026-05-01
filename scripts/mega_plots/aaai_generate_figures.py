from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ONE_COL_IN = 3.3
TWO_COL_IN = 6.975
DPI = 300


def apply_aaai_rcparams() -> None:
    mpl.rcParams.update(
        {
            # Text: keep >= 9 pt inside figure
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            # Serif to match AAAI Times/Nimbus family.
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
            # Avoid Type 3 fonts in PDFs.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Lines: >= 0.5 pt
            "axes.linewidth": 0.8,
            # Deterministic output
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
        }
    )


def style_axes(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="major", width=0.6, length=3.0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def save_fig(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_base.with_suffix(".png"), dpi=DPI, bbox_inches="tight", pad_inches=0.01)


def load_stats(path: Path) -> dict:
    return json.loads(path.read_text())


def ccdf_from_hist(bin_edges: Iterable[float], bin_counts: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    edges = np.asarray(list(bin_edges), dtype=float)
    counts = np.asarray(list(bin_counts), dtype=float)
    total = counts.sum()
    survival = np.cumsum(counts[::-1])[::-1] / total
    x = edges[:-1]
    return x, survival


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(0.02, 0.96, label, transform=ax.transAxes, ha="left", va="top", fontsize=9)


@dataclass(frozen=True)
class StackedSeg:
    label: str
    pct: float
    color: str


def draw_stacked_barh(
    ax: plt.Axes,
    *,
    y: float,
    height: float,
    segs: list[StackedSeg],
    label_threshold_pct: float,
    small_label_y_offsets: list[float],
    bar_edge_lw: float = 0.6,
    small_label_pad: float = 0.02,
) -> None:
    left = 0.0
    segments_for_labels: list[tuple[float, float, str, float]] = []

    for seg in segs:
        ax.barh(
            y,
            seg.pct,
            left=left,
            height=height,
            color=seg.color,
            edgecolor="black",
            linewidth=bar_edge_lw,
        )
        segments_for_labels.append((left, seg.pct, seg.label, seg.pct))
        left += seg.pct

    y_top = y + height / 2

    # In-segment labels for wide segments.
    for left, width, label, pct in segments_for_labels:
        if width >= label_threshold_pct:
            ax.text(
                left + width / 2,
                y,
                f"{label} ({pct:.1f}%)",
                ha="center",
                va="center",
                fontsize=9,
            )

    # Above-bar labels with short vertical leaders for small segments (no diagonals).
    small = [(left, width, label, pct) for left, width, label, pct in segments_for_labels if width < label_threshold_pct]
    if not small:
        return

    # Stable left-to-right ordering so labels never cross.
    small.sort(key=lambda t: t[0] + t[1] / 2)
    if len(small_label_y_offsets) < len(small):
        raise ValueError("Not enough y-offsets for small labels")

    for (left, width, label, pct), y_off in zip(small, small_label_y_offsets):
        x = left + width / 2
        y_text = y_top + y_off
        ha = "right" if x >= 92 else ("left" if x <= 8 else "center")
        # Draw a guaranteed-vertical leader line (avoid diagonal arrows).
        ax.plot([x, x], [y_top, y_text], color="black", linewidth=0.6, solid_capstyle="butt")
        ax.text(x, y_text, f"{label} ({pct:.1f}%)", ha=ha, va="bottom", fontsize=9)
    # Keep labels from touching the plot frame in extreme cases.
    ax.margins(x=small_label_pad)


def figure_url_presence_split(stats: dict, out_dir: Path) -> None:
    # 1-column: 3.3 x 1.05 in
    fig = plt.figure(figsize=(ONE_COL_IN, 1.05), dpi=DPI)
    # Leave a bit of right margin so small-segment callouts stay inside the canvas.
    ax = fig.add_axes([0.08, 0.20, 0.86, 0.75])

    c = stats["counters"]
    no_url = int(c["total_msgs"]) - int(c["msgs_with_url"])
    unrated = int(c["msgs_with_url_unrated"])
    rated = int(c["msgs_with_rated_url"])
    total = no_url + unrated + rated

    segs = [
        StackedSeg("No URL", no_url / total * 100.0, "0.85"),
        StackedSeg("Unrated", unrated / total * 100.0, "0.60"),
        StackedSeg("Rated", rated / total * 100.0, "0.35"),
    ]

    draw_stacked_barh(
        ax,
        y=0.0,
        height=0.30,
        segs=segs,
        label_threshold_pct=12.0,
        small_label_y_offsets=[0.14, 0.26, 0.38],
    )

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.25, 0.45)
    ax.set_yticks([])
    ax.set_xlabel("Share of messages (%)")
    ax.set_xticks([0, 20, 40, 60, 80, 100])

    style_axes(ax)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    save_fig(fig, out_dir / "url_presence_split")
    plt.close(fig)


def figure_url_count_per_message(stats: dict, out_dir: Path) -> None:
    # 1-column: 3.3 x 1.15 in
    fig = plt.figure(figsize=(ONE_COL_IN, 1.15), dpi=DPI)

    # Leave space on the right for a compact legend (keeps labels collision-free).
    ax_top = fig.add_axes([0.08, 0.60, 0.58, 0.30])
    ax_bottom = fig.add_axes([0.08, 0.20, 0.58, 0.30], sharex=ax_top)
    ax_leg = fig.add_axes([0.70, 0.20, 0.29, 0.70])
    ax_leg.set_axis_off()

    url_count_hist = {int(k): int(v) for k, v in stats["url_count_hist"].items()}
    total_msgs = sum(url_count_hist.values())
    zero = url_count_hist.get(0, 0)
    ge1 = total_msgs - zero

    def draw_bar(
        ax: plt.Axes, segs: list[StackedSeg], *, inside_threshold: float, multiline_below: float | None = None
    ) -> list[tuple[StackedSeg, mpl.patches.Patch]]:
        left = 0.0
        handles: list[tuple[StackedSeg, mpl.patches.Patch]] = []
        for seg in segs:
            bars = ax.barh(
                0.0,
                seg.pct,
                left=left,
                height=0.55,
                color=seg.color,
                edgecolor="black",
                linewidth=0.6,
            )
            handles.append((seg, bars[0]))
            if seg.pct >= inside_threshold:
                label_txt = f"{seg.label} ({seg.pct:.1f}%)"
                if multiline_below is not None and seg.pct < multiline_below:
                    label_txt = f"{seg.label}\n({seg.pct:.1f}%)"
                ax.text(
                    left + seg.pct / 2,
                    0.0,
                    label_txt,
                    ha="center",
                    va="center",
                    fontsize=9,
                )
            left += seg.pct

        ax.set_xlim(0, 100)
        ax.set_ylim(-0.9, 0.9)
        ax.set_yticks([])
        style_axes(ax)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        return handles

    # Row 1: 0 vs ≥1 (unconditional)
    segs_all = [
        StackedSeg("0 URLs", zero / total_msgs * 100.0, "0.85"),
        StackedSeg("≥1 URL", ge1 / total_msgs * 100.0, "0.55"),
    ]
    draw_bar(ax_top, segs_all, inside_threshold=25.0, multiline_below=None)
    ax_top.text(0.0, 0.98, "All messages", transform=ax_top.transAxes, ha="left", va="top", fontsize=9)
    # Callout for the smaller segment (keeps text inside canvas, no diagonals).
    ge1_left = segs_all[0].pct
    ge1_width = segs_all[1].pct
    ge1_x = ge1_left + ge1_width / 2
    ge1_y_top = 0.55 / 2
    y_text = 0.62
    ax_top.plot([ge1_x, ge1_x], [ge1_y_top, y_text], color="black", linewidth=0.6, solid_capstyle="butt")
    ax_top.text(
        ge1_x,
        y_text + 0.02,
        f"≥1 URL\n({segs_all[1].pct:.1f}%)",
        ha="right",
        va="bottom",
        fontsize=9,
    )

    # Row 2: conditional among messages with ≥1 URL
    counts_cond = {
        "1 URL": url_count_hist.get(1, 0),
        "2 URLs": url_count_hist.get(2, 0),
        "3 URLs": url_count_hist.get(3, 0),
        "4 URLs": url_count_hist.get(4, 0),
        "5+ URLs": sum(v for k, v in url_count_hist.items() if k >= 5),
    }
    cond_total = sum(counts_cond.values())
    segs_cond = [
        StackedSeg("1 URL", counts_cond["1 URL"] / cond_total * 100.0, "0.80"),
        StackedSeg("2 URLs", counts_cond["2 URLs"] / cond_total * 100.0, "0.65"),
        StackedSeg("3 URLs", counts_cond["3 URLs"] / cond_total * 100.0, "0.52"),
        StackedSeg("4 URLs", counts_cond["4 URLs"] / cond_total * 100.0, "0.40"),
        StackedSeg("5+ URLs", counts_cond["5+ URLs"] / cond_total * 100.0, "0.28"),
    ]
    handles = draw_bar(ax_bottom, segs_cond, inside_threshold=12.0)
    ax_bottom.text(0.0, 0.98, "Given ≥1 URL", transform=ax_bottom.transAxes, ha="left", va="top", fontsize=9)
    ax_bottom.set_xlabel("Share (%)")
    ax_bottom.set_xticks([0, 20, 40, 60, 80, 100])
    ax_bottom.tick_params(axis="x", which="both", bottom=True, labelbottom=True)
    ax_bottom.spines["bottom"].set_visible(True)
    ax_bottom.spines["top"].set_visible(False)

    # Legend on the right (explicit + collision-free).
    y = 0.82
    dy = 0.16
    for seg, patch in handles:
        ax_leg.add_patch(
            mpl.patches.Rectangle(
                (0.0, y - 0.06),
                0.14,
                0.10,
                facecolor=seg.color,
                edgecolor="black",
                linewidth=0.6,
                transform=ax_leg.transAxes,
                clip_on=False,
            )
        )
        ax_leg.text(0.18, y, f"{seg.label} ({seg.pct:.1f}%)", transform=ax_leg.transAxes, ha="left", va="center", fontsize=9)
        y -= dy

    save_fig(fig, out_dir / "url_count_per_message")
    plt.close(fig)


def figure_url_length_distribution(stats: dict, out_dir: Path) -> None:
    # 1-column: 3.3 x 2.0 in
    fig = plt.figure(figsize=(ONE_COL_IN, 2.0), dpi=DPI)
    ax = fig.add_axes([0.23, 0.25, 0.74, 0.72])

    edges = np.asarray(stats["url_len_bins"], dtype=float)
    counts = np.asarray(stats["url_len_hist"], dtype=float)
    widths = np.diff(edges)
    pct = counts / counts.sum() * 100.0

    ax.bar(edges[:-1], pct, width=widths, align="edge", edgecolor="black", linewidth=0.6, color="0.75")
    ax.set_xlabel("URL length (characters)")
    ax.set_ylabel("Share of URLs (%)")

    # Main view: [0, 500], since tail >500 is negligible (≈0.03%).
    ax.set_xlim(0, 500)
    ax.set_xticks([0, 100, 200, 500])
    ax.set_xticks([20], minor=True)
    ax.tick_params(axis="x", which="minor", width=0.6, length=2.5)

    # Threshold at 20.
    ax.axvline(20, linestyle="--", linewidth=0.8, color="black")
    ax.set_ylim(0, float(pct.max()) * 1.12)
    ax.text(
        20,
        ax.get_ylim()[1] * 0.93,
        "20",
        ha="center",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.2),
    )

    # Inset: tail [500, 1000] with its own y-scale (explicitly labeled as a zoom).
    try:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    except Exception:
        inset_axes = None

    if inset_axes is not None:
        ax_in = inset_axes(ax, width="40%", height="45%", loc="upper right", borderpad=1.0)
        ax_in.bar(edges[:-1], pct, width=widths, align="edge", edgecolor="black", linewidth=0.6, color="0.75")
        ax_in.set_xlim(500, 1000)
        tail_max = float(pct[edges[:-1] >= 500].max()) if np.any(edges[:-1] >= 500) else float(pct.max())
        ax_in.set_ylim(0, max(tail_max * 1.4, 0.001))
        ax_in.set_xticks([500, 1000])
        ax_in.set_yticks([])
        ax_in.tick_params(axis="both", which="major", labelsize=9, pad=1)
        style_axes(ax_in)

    style_axes(ax)
    ax.grid(False)

    save_fig(fig, out_dir / "url_length_distribution")
    plt.close(fig)


def figure_message_length_ccdf_two_panel(stats: dict, out_dir: Path) -> None:
    # 2-column: 6.975 x 2.3 in
    fig = plt.figure(figsize=(TWO_COL_IN, 2.3), dpi=DPI)
    gs = fig.add_gridspec(1, 2, wspace=0.25)

    word_edges = np.asarray(stats["word_bins"], dtype=float)
    word_counts = np.asarray(stats["word_hist"], dtype=float)
    char_edges = np.asarray(stats["char_bins"], dtype=float)
    char_counts = np.asarray(stats["char_hist"], dtype=float)

    n_word = float(word_counts.sum())
    n_char = float(char_counts.sum())
    y_min = 1.0 / max(n_word, n_char)

    def xlim_at_quantile(edges: np.ndarray, counts: np.ndarray, q: float) -> float:
        total = float(counts.sum())
        cdf = np.cumsum(counts) / total
        idx = int(np.searchsorted(cdf, q, side="left"))
        idx = min(max(idx, 0), len(counts) - 1)
        return float(edges[idx + 1])

    ax1 = fig.add_subplot(gs[0, 0])
    xw, yw = ccdf_from_hist(word_edges, word_counts)
    # Extend last bin horizontally (binned CCDF approximation) to avoid an artificial truncation.
    xw_plot = np.concatenate([xw, [word_edges[-1]]])
    yw_plot = np.concatenate([yw, [yw[-1]]])
    ax1.step(xw_plot, yw_plot, where="post", color="black", linewidth=1.0)
    ax1.set_xlabel("Message length (words)")
    ax1.set_ylabel("CCDF (share ≥ x)")
    ax1.set_xlim(0, xlim_at_quantile(word_edges, word_counts, 0.999))
    ax1.set_yscale("log")
    ax1.set_ylim(y_min, 1.0)
    ax1.grid(True, which="major", axis="y", linewidth=0.5, color="0.0", alpha=0.2)
    ax1.grid(False, which="minor", axis="y")
    style_axes(ax1)

    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    xc, yc = ccdf_from_hist(char_edges, char_counts)
    x2_max = xlim_at_quantile(char_edges, char_counts, 0.999)
    # Extend last step horizontally for a clean stop at the axis limit.
    xc_plot = np.concatenate([xc, [x2_max]])
    yc_plot = np.concatenate([yc, [yc[-1]]])
    ax2.step(xc_plot, yc_plot, where="post", color="black", linewidth=1.0)
    ax2.set_xlabel("Message length (characters)")
    ax2.set_xlim(0, x2_max)
    ax2.set_yscale("log")
    ax2.grid(True, which="major", axis="y", linewidth=0.5, color="0.0", alpha=0.2)
    ax2.grid(False, which="minor", axis="y")
    style_axes(ax2)
    plt.setp(ax2.get_yticklabels(), visible=False)

    save_fig(fig, out_dir / "message_length_ccdf_two_panel")
    plt.close(fig)


def figure_mbfc_overall_threepanel(stats: dict, out_dir: Path) -> None:
    # 2-column: 6.975 x 2.8 in
    fig = plt.figure(figsize=(TWO_COL_IN, 2.8), dpi=DPI)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.9, 1.0], wspace=0.25, hspace=0.28)

    def normalize_key(value) -> str:
        if value is None:
            return "Unknown"
        s = str(value).strip()
        if s.lower() in {"nan", "none", ""}:
            return "Unknown"
        return s

    bias_counts_raw = {normalize_key(k): int(v) for k, v in stats["bias_counts"].items()}
    if "Conspiracy-Pseuscience" in bias_counts_raw:
        bias_counts_raw["Conspiracy-Pseudoscience"] = bias_counts_raw.get("Conspiracy-Pseudoscience", 0) + bias_counts_raw.pop(
            "Conspiracy-Pseuscience"
        )

    factual_counts_raw = {normalize_key(k).title(): int(v) for k, v in stats["factual_counts"].items()}
    cred_counts_raw = {normalize_key(k).title(): int(v) for k, v in stats["cred_counts"].items()}

    rated_urls = float(int(stats["counters"]["rated_urls"]))

    def pct_series(counts: dict[str, int]) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        for k, v in counts.items():
            if v <= 0:
                continue
            out.append((k, v / rated_urls * 100.0))
        out.sort(key=lambda t: (-t[1], t[0]))
        return out

    def wrap_label(label: str) -> str:
        # Wrap long labels to 2 lines for readability in 2-column.
        if len(label) <= 14:
            return label
        if "-" in label:
            a, b = label.split("-", 1)
            return f"{a}-\n{b}"
        if " " in label:
            a, b = label.split(" ", 1)
            return f"{a}\n{b}"
        return label

    def hbar_panel(ax: plt.Axes, series: list[tuple[str, float]], panel_label: str, *, x_label: str | None) -> None:
        labels = [wrap_label(k) for k, _ in series]
        pcts = np.asarray([v for _, v in series], dtype=float)
        # Non-uniform vertical spacing: add extra gap before multi-line labels
        # to prevent collisions at 9pt within the fixed figure height.
        y: list[float] = []
        pos = 0.0
        base_step = 1.0
        extra_before_multiline = 0.8
        for lab in labels:
            if "\n" in lab:
                pos += extra_before_multiline
            y.append(pos)
            pos += base_step
        y_arr = np.asarray(y, dtype=float)

        ax.barh(y_arr, pcts, height=0.78, color="0.75", edgecolor="black", linewidth=0.6)
        ax.set_yticks(y_arr)
        ax.set_yticklabels(labels)
        ax.tick_params(axis="y", pad=10)
        ax.invert_yaxis()

        xmax = float(pcts.max()) + 5.0
        ax.set_xlim(0, xmax)
        dx = xmax * 0.015
        for yi, pct in zip(y_arr, pcts):
            label_txt = f"{pct:.1f}%"
            if pct + dx > xmax:
                ax.text(max(pct - dx, 0.5), yi, label_txt, va="center", ha="right", fontsize=9)
            else:
                ax.text(pct + dx, yi, label_txt, va="center", ha="left", fontsize=9)

        if x_label is not None:
            ax.set_xlabel(x_label)

        # Minimize unused vertical padding so tick labels have room.
        ax.margins(y=0.0)
        style_axes(ax)
        add_panel_label(ax, panel_label)

        # Multi-line labels: tighten internal line spacing a bit.
        for t in ax.get_yticklabels():
            if "\n" in t.get_text():
                t.set_linespacing(0.65)

    ax_bias = fig.add_subplot(gs[0, :])
    ax_fact = fig.add_subplot(gs[1, 0])
    ax_cred = fig.add_subplot(gs[1, 1])

    hbar_panel(ax_bias, pct_series(bias_counts_raw), "(a)", x_label=None)
    hbar_panel(ax_fact, pct_series(factual_counts_raw), "(b)", x_label="Share of rated URLs (%)")
    hbar_panel(ax_cred, pct_series(cred_counts_raw), "(c)", x_label="Share of rated URLs (%)")

    # Ensure long y-tick labels never clip.
    fig.subplots_adjust(left=0.34, right=0.99)

    save_fig(fig, out_dir / "mbfc_overall_threepanel")
    plt.close(fig)


def generate_all(stats_path: Path, out_dir: Path) -> None:
    apply_aaai_rcparams()
    stats = load_stats(stats_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    figure_url_presence_split(stats, out_dir)
    figure_url_count_per_message(stats, out_dir)
    figure_url_length_distribution(stats, out_dir)
    figure_message_length_ccdf_two_panel(stats, out_dir)
    figure_mbfc_overall_threepanel(stats, out_dir)


def _find_default_stats_path() -> Path:
    cwd = Path.cwd()
    if (cwd / "mega_dataset_stats.json").exists():
        return cwd / "mega_dataset_stats.json"
    if (cwd.parent / "mega_dataset_stats.json").exists():
        return cwd.parent / "mega_dataset_stats.json"
    raise FileNotFoundError("Could not find mega_dataset_stats.json in cwd or parent")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ICWSM/AAAI-ready figures from mega_dataset_stats.json")
    parser.add_argument("--stats", type=Path, default=_find_default_stats_path(), help="Path to mega_dataset_stats.json")
    parser.add_argument("--out", type=Path, default=Path("mega_plots") / "aaai", help="Output directory")
    args = parser.parse_args()

    generate_all(args.stats, args.out)
    print("Wrote figures to", args.out.resolve())


if __name__ == "__main__":
    main()
