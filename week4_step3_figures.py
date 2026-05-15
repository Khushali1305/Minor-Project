"""
week4_step3_figures.py
======================
Generates camera-ready figures from week4_model_scores.json.

FIGURES
-------
  Figure 1 (RQ1): Scatter — Ordinary Score vs Global Reliability R(f)
                  Models in the high-ordinary/low-reliability quadrant
                  are "hidden failures" — the core RQ1 finding.

  Figure 2 (RQ2): Grouped bar — Invariance / Directional / Shortcut pass
                  rates per model. Shows which family is most diagnostic.

  Figure 3 (RQ2): Heatmap — all metrics per model at a glance.

OUTPUTS
-------
  fig1_rq1_ordinary_vs_reliability.pdf  +  .png
  fig2_rq2_family_bars.pdf              +  .png
  fig3_rq2_heatmap.pdf                  +  .png

STYLE
-----
  EMNLP column-width sizing, Times serif, 300 DPI, ColorBrewer palette.

BUGS FIXED FROM PREVIOUS VERSION
----------------------------------
  1. list[dict] type hints (Python 3.9+) -> removed, compatible with 3.8+
  2. gpt4 display name and category updated to match actual model key "gpt4"

USAGE
-----
  python3 week4_step3_figures.py

  # Different scores file or output directory
  python3 week4_step3_figures.py --scores week4_model_scores.json --output_dir ./figures

DEPENDENCIES
------------
  pip install matplotlib seaborn numpy
"""

import json
import argparse
import logging
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "model_scores_file": "week4_model_scores.json",
    "output_dir"       : ".",
    "dpi"              : 300,
    "fig_width_full"   : 6.9,   # EMNLP full page width in inches
    "fig_width_single" : 3.3,   # EMNLP single column width
    "fig_height"       : 3.0,
}

# Display names for all model keys used in step1
MODEL_DISPLAY_NAMES = {
    "rule_baseline": "Rule\nBaseline",
    "flan_t5_large": "FLAN-T5\nLarge",
    "flan_t5_xl"   : "FLAN-T5\nXL",
    "llama_8b"     : "Llama\n3.1-8B",
    "llama_70b"    : "Llama\n3.1-70B",
    "qwen_32b"     : "Qwen2.5\n32B",
    "gemma_27b"    : "Gemma\n3-27B",
    "gpt4"         : "GPT-4.1",   # key is "gpt4" in outputs
}

# Model categories for Figure 1 colour/marker encoding
MODEL_CATEGORY = {
    "rule_baseline": "rule",
    "flan_t5_large": "seq2seq",
    "flan_t5_xl"   : "seq2seq",
    "llama_8b"     : "llm_small",
    "llama_70b"    : "llm_large",
    "qwen_32b"     : "llm_large",
    "gemma_27b"    : "llm_large",
    "gpt4"         : "proprietary",  # GPT-4.1 is proprietary
}

# ColorBrewer-safe palette (colour-blind safe)
PALETTE = {
    "rule"       : "#d95f02",
    "seq2seq"    : "#7570b3",
    "llm_small"  : "#1b9e77",
    "llm_large"  : "#1b9e77",
    "proprietary": "#e7298a",
}

MARKER = {
    "rule"       : "D",
    "seq2seq"    : "s",
    "llm_small"  : "o",
    "llm_large"  : "^",
    "proprietary": "*",
}

FAMILY_COLORS = {
    "Inv. Rel." : "#4393c3",
    "Dir. Sens.": "#d6604d",
    "Shc. Res." : "#74c476",
}


# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB STYLE
# ─────────────────────────────────────────────────────────────────────────────

def setup_style():
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "font.family"       : "serif",
        "font.serif"        : ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size"         : 9,
        "axes.titlesize"    : 9,
        "axes.labelsize"    : 9,
        "xtick.labelsize"   : 8,
        "ytick.labelsize"   : 8,
        "legend.fontsize"   : 7.5,
        "figure.dpi"        : CONFIG["dpi"],
        "savefig.dpi"       : CONFIG["dpi"],
        "savefig.bbox"      : "tight",
        "savefig.pad_inches": 0.02,
        "axes.spines.top"   : False,
        "axes.spines.right" : False,
        "axes.grid"         : True,
        "grid.alpha"        : 0.3,
        "grid.linewidth"    : 0.5,
    })
    return plt


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — RQ1: Ordinary Score vs Global Reliability scatter
# ─────────────────────────────────────────────────────────────────────────────

def figure1_rq1_scatter(model_scores, plt, output_dir):
    fig, ax = plt.subplots(figsize=(CONFIG["fig_width_full"], CONFIG["fig_height"]))

    valid = [
        s for s in model_scores
        if s.get("ordinary_score") is not None
        and s.get("global_reliability_R") is not None
    ]

    if not valid:
        logging.warning("No data for Figure 1 — generating placeholder.")
        _placeholder(fig, ax, "Figure 1 — run step2 first to generate scores")
        _save(fig, plt, output_dir, "fig1_rq1_ordinary_vs_reliability")
        return

    legend_handles = {}
    for s in valid:
        mid  = s["model_id"]
        cat  = MODEL_CATEGORY.get(mid, "llm_large")
        name = MODEL_DISPLAY_NAMES.get(mid, mid).replace("\n", " ")
        x    = s["ordinary_score"]
        y    = s["global_reliability_R"]

        sc = ax.scatter(
            x, y,
            color=PALETTE[cat], marker=MARKER[cat],
            s=80, zorder=3,
            edgecolors="white", linewidths=0.5,
        )
        if cat not in legend_handles:
            legend_handles[cat] = sc

        ax.annotate(
            name, xy=(x, y), xytext=(4, 4),
            textcoords="offset points",
            fontsize=6.5, color="#333333",
        )

    # Quadrant lines
    ax.axhline(0.5, color="#999999", linewidth=0.8, linestyle="--", zorder=1)
    ax.axvline(0.5, color="#999999", linewidth=0.8, linestyle="--", zorder=1)

    # Quadrant labels
    ax.text(
        0.76, 0.18,
        "High ordinary\nLow reliability\n(hidden failures)",
        transform=ax.transAxes, fontsize=6.5, color="#c0392b",
        ha="center", va="center", style="italic",
        bbox=dict(
            boxstyle="round,pad=0.2", facecolor="#fdecea",
            alpha=0.7, linewidth=0,
        ),
    )
    ax.text(
        0.22, 0.82, "Low ordinary\nHigh reliability",
        transform=ax.transAxes, fontsize=6.5, color="#27ae60",
        ha="center", va="center", style="italic",
    )

    ax.set_xlabel("Ordinary Score (EARS conformance + meaning preservation)")
    ax.set_ylabel("Global Reliability R(f)")
    ax.set_title("Figure 1: Ordinary evaluation vs probe-based reliability (RQ1)", pad=6)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    cat_labels = {
        "rule"       : "Rule-based",
        "seq2seq"    : "Seq2seq (FLAN-T5)",
        "llm_small"  : "LLM (small)",
        "llm_large"  : "LLM (large)",
        "proprietary": "Proprietary (GPT-4.1)",
    }
    cats_present = {MODEL_CATEGORY.get(s["model_id"]) for s in valid}
    handles = [
        plt.scatter([], [], color=PALETTE[c], marker=MARKER[c],
                    s=60, label=cat_labels[c])
        for c in cat_labels if c in cats_present
    ]
    ax.legend(handles=handles, loc="lower right",
              framealpha=0.9, edgecolor="#cccccc")

    _save(fig, plt, output_dir, "fig1_rq1_ordinary_vs_reliability")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — RQ2: Grouped bar chart
# ─────────────────────────────────────────────────────────────────────────────

def figure2_rq2_bars(model_scores, plt, output_dir):
    import numpy as np

    valid = sorted(
        [s for s in model_scores if s.get("global_reliability_R") is not None],
        key=lambda x: x["global_reliability_R"],
        reverse=True,
    )

    if not valid:
        fig, ax = plt.subplots(figsize=(CONFIG["fig_width_full"], CONFIG["fig_height"]))
        _placeholder(fig, ax, "Figure 2 — no data")
        _save(fig, plt, output_dir, "fig2_rq2_family_bars")
        return

    labels   = [MODEL_DISPLAY_NAMES.get(s["model_id"], s["model_id"]) for s in valid]
    inv_vals = [s.get("invariance_reliability")  or 0 for s in valid]
    dir_vals = [s.get("directional_sensitivity") or 0 for s in valid]
    shc_vals = [s.get("shortcut_resistance")     or 0 for s in valid]

    x     = np.arange(len(valid))
    width = 0.26

    fig, ax = plt.subplots(
        figsize=(CONFIG["fig_width_full"], CONFIG["fig_height"] + 0.5)
    )

    b1 = ax.bar(x - width, inv_vals, width, label="Inv. Rel.",
                color=FAMILY_COLORS["Inv. Rel."],  zorder=3)
    b2 = ax.bar(x,          dir_vals, width, label="Dir. Sens.",
                color=FAMILY_COLORS["Dir. Sens."], zorder=3)
    b3 = ax.bar(x + width,  shc_vals, width, label="Shc. Res.",
                color=FAMILY_COLORS["Shc. Res."],  zorder=3)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.02:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom",
                    fontsize=6, color="#333333",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Pass Rate")
    ax.set_ylim(0, 1.12)
    ax.set_title("Figure 2: Probe-family pass rates per model (RQ2)", pad=6)
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="#cccccc", ncol=3)
    ax.axhline(1.0, color="#cccccc", linewidth=0.5, linestyle=":")

    _save(fig, plt, output_dir, "fig2_rq2_family_bars")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — RQ2: Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def figure3_rq2_heatmap(model_scores, plt, output_dir):
    import numpy as np
    import seaborn as sns

    valid = sorted(
        [s for s in model_scores if s.get("global_reliability_R") is not None],
        key=lambda x: x["global_reliability_R"],
        reverse=True,
    )

    if not valid:
        fig, ax = plt.subplots(figsize=(CONFIG["fig_width_full"], CONFIG["fig_height"]))
        _placeholder(fig, ax, "Figure 3 — no data")
        _save(fig, plt, output_dir, "fig3_rq2_heatmap")
        return

    row_labels = [
        MODEL_DISPLAY_NAMES.get(s["model_id"], s["model_id"]).replace("\n", " ")
        for s in valid
    ]
    col_labels = ["Inv. Rel.", "Dir. Sens.", "Shc. Res.", "Global R", "Ord. Score"]
    col_keys   = [
        "invariance_reliability", "directional_sensitivity",
        "shortcut_resistance", "global_reliability_R", "ordinary_score",
    ]

    matrix = np.array([
        [s.get(k) or 0.0 for k in col_keys] for s in valid
    ])

    fig_h = max(CONFIG["fig_height"], 0.45 * len(valid) + 0.8)
    fig, ax = plt.subplots(figsize=(CONFIG["fig_width_full"], fig_h))

    sns.heatmap(
        matrix, ax=ax,
        annot=True, fmt=".3f",
        annot_kws={"size": 7.5},
        cmap="Blues", vmin=0.0, vmax=1.0,
        xticklabels=col_labels,
        yticklabels=row_labels,
        linewidths=0.4, linecolor="#dddddd",
        cbar_kws={"shrink": 0.7, "label": "Pass Rate"},
    )

    # Vertical separator before Ordinary Score
    ax.axvline(x=4, color="#333333", linewidth=1.5)

    ax.set_title(
        "Figure 3: Reliability and ordinary scores across models (RQ1/RQ2)",
        pad=8,
    )
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    _save(fig, plt, output_dir, "fig3_rq2_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _placeholder(fig, ax, message):
    ax.text(
        0.5, 0.5, message,
        ha="center", va="center",
        fontsize=10, color="#888888",
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _save(fig, plt, output_dir, stem):
    for ext in ["pdf", "png"]:
        path = str(Path(output_dir) / f"{stem}.{ext}")
        fig.savefig(path)
        logging.info(f"  Saved: {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores", default="week4_model_scores.json",
        help="Path to model scores JSON (default: week4_model_scores.json)",
    )
    parser.add_argument(
        "--output_dir", default=".",
        help="Output directory for figures (default: current directory)",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()
    CONFIG["model_scores_file"] = args.scores
    CONFIG["output_dir"]        = args.output_dir

    try:
        import matplotlib
        import seaborn
        import numpy
    except ImportError:
        raise ImportError("pip install matplotlib seaborn numpy")

    plt = setup_style()

    with open(CONFIG["model_scores_file"]) as f:
        model_scores = json.load(f)
    logging.info(f"Model scores loaded: {len(model_scores)} models")

    logging.info("Figure 1 (RQ1 scatter) ...")
    figure1_rq1_scatter(model_scores, plt, CONFIG["output_dir"])

    logging.info("Figure 2 (RQ2 bars) ...")
    figure2_rq2_bars(model_scores, plt, CONFIG["output_dir"])

    logging.info("Figure 3 (RQ2 heatmap) ...")
    figure3_rq2_heatmap(model_scores, plt, CONFIG["output_dir"])

    logging.info("All figures saved.")


if __name__ == "__main__":
    main()
