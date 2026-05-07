"""
Visualizations for the transcript intelligence findings.

All charts are written to outputs/ as PNG so the slide deck and
notebook can reference them without re-running the analysis.

Style choices:
  - Single-color base palette with one accent for "negative" signals.
  - No gridlines / chart junk. Clean, leadership-deck-ready.
  - All charts rendered at 1200x800 @ 200 DPI for clean projector display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PALETTE = {
    "primary": "#1E3A8A",     # Aegis-style deep blue
    "accent_pos": "#10B981",  # green
    "accent_neg": "#EF4444",  # red
    "neutral":   "#94A3B8",   # slate
    "support":   "#F59E0B",   # amber
    "external":  "#1E3A8A",   # primary blue
    "internal":  "#6D28D9",   # purple
}


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def chart_sentiment_by_call_type(df: pd.DataFrame, out_dir: Path) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = ["external", "internal", "support"]
    colors = [PALETTE[k] for k in order]

    # Use matplotlib directly to keep color/order alignment unambiguous
    data_groups = [df[df.call_type == ct].sentiment_score.values for ct in order]
    bp = ax.boxplot(
        data_groups, positions=range(len(order)), widths=0.55,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor("black")

    # Strip overlay
    for i, ct in enumerate(order):
        y = df[df.call_type == ct].sentiment_score.values
        x = np.random.normal(i, 0.05, size=len(y))
        ax.scatter(x, y, c="black", alpha=0.35, s=14)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylim(1, 5)
    ax.set_ylabel("Sentiment score (1–5)")
    ax.set_xlabel("")
    ax.set_title("Sentiment is materially lower on support calls")
    ax.axhline(3.0, color=PALETTE["neutral"], lw=0.7, ls="--", alpha=0.6)

    means = df.groupby("call_type")["sentiment_score"].mean()
    for i, ct in enumerate(order):
        ax.text(i, means[ct] + 0.20, f"μ={means[ct]:.2f}", ha="center",
                fontweight="bold", fontsize=10)

    out = out_dir / "sentiment_by_call_type.png"
    plt.savefig(out)
    plt.close()
    return out


def chart_sentiment_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = ["very-negative", "negative", "mixed-negative",
             "mixed-positive", "positive", "very-positive"]
    counts = df["overall_sentiment"].value_counts().reindex(order, fill_value=0)
    bar_colors = [PALETTE["accent_neg"]] * 3 + [PALETTE["accent_pos"]] * 3
    ax.bar(range(len(counts)), counts.values, color=bar_colors, edgecolor="white")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("Number of calls")
    ax.set_title("Polarized corpus: 38% negative-leaning, 61% positive-leaning")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.5, str(v), ha="center", fontweight="bold")
    out = out_dir / "sentiment_distribution.png"
    plt.savefig(out)
    plt.close()
    return out


def chart_account_health(health: pd.DataFrame, out_dir: Path, top_n: int = 12) -> Path:
    _style()
    sub = pd.concat([health.head(top_n // 2), health.tail(top_n // 2)])
    fig, ax = plt.subplots(figsize=(8, 6))
    band_color = {
        "Critical": PALETTE["accent_neg"],
        "At Risk":  "#FB923C",
        "Watch":    "#FACC15",
        "Healthy":  PALETTE["accent_pos"],
    }
    colors = [band_color[b] for b in sub.risk_band]
    ax.barh(sub.account, sub.health_score, color=colors, edgecolor="white")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Account health score (0–100)")
    ax.set_title(f"Account health: {top_n // 2} most at risk vs {top_n // 2} strongest")
    ax.invert_yaxis()
    for i, (acc, score, band) in enumerate(zip(sub.account, sub.health_score, sub.risk_band)):
        ax.text(score + 1.5, i, f"{score:.0f} · {band}", va="center", fontsize=9)
    out = out_dir / "account_health.png"
    plt.savefig(out)
    plt.close()
    return out


def chart_account_journeys(df: pd.DataFrame, out_dir: Path,
                            accounts: list[str]) -> Path:
    """Multi-touchpoint sentiment trajectory for selected accounts."""
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    axes = axes.flatten()
    for ax, acc in zip(axes, accounts):
        sub = df[df.account == acc].sort_values("start_time")
        if len(sub) == 0:
            ax.set_visible(False)
            continue
        type_color = {"support": PALETTE["support"],
                      "external": PALETTE["external"],
                      "internal": PALETTE["internal"]}
        colors = [type_color[t] for t in sub.call_type]
        ax.plot(range(len(sub)), sub.sentiment_score.values,
                color=PALETTE["neutral"], lw=1.2, alpha=0.5, zorder=1)
        ax.scatter(range(len(sub)), sub.sentiment_score.values,
                   c=colors, s=130, edgecolor="white", lw=1.2, zorder=2)
        for i, (st, ct, sc) in enumerate(zip(sub.start_time, sub.call_type, sub.sentiment_score)):
            ax.annotate(
                f"{ct[0].upper()}", (i, sc),
                xytext=(0, 0), textcoords="offset points",
                ha="center", va="center", fontsize=8,
                fontweight="bold", color="white",
            )
        ax.set_ylim(0.8, 5.2)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([d.strftime("%b %d") for d in sub.start_time],
                           rotation=20, ha="right", fontsize=8)
        ax.set_title(acc, fontsize=11)
        ax.axhline(3, color=PALETTE["neutral"], lw=0.5, ls="--", alpha=0.5)
    fig.suptitle(
        "Account journeys reveal the gap between AM optimism and support reality",
        fontweight="bold", y=1.0,
    )
    legend_handles = [
        plt.scatter([], [], c=PALETTE["external"], s=80, label="External (AM)"),
        plt.scatter([], [], c=PALETTE["support"], s=80, label="Support"),
        plt.scatter([], [], c=PALETTE["internal"], s=80, label="Internal"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    out = out_dir / "account_journeys.png"
    plt.savefig(out)
    plt.close()
    return out


def chart_topic_clusters(df_clustered: pd.DataFrame, terms: dict, out_dir: Path) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(9, 5))
    cluster_counts = df_clustered.cluster_id.value_counts().sort_index()
    cluster_labels = [
        f"C{cid}: {' / '.join(terms[cid][:3])}" for cid in cluster_counts.index
    ]
    bars = ax.barh(cluster_labels, cluster_counts.values,
                   color=PALETTE["primary"], edgecolor="white")
    ax.set_xlabel("Number of calls in cluster")
    ax.set_title("8 latent themes consolidate 351 free-form per-call topic labels")
    for i, v in enumerate(cluster_counts.values):
        ax.text(v + 0.3, i, str(v), va="center", fontweight="bold")
    ax.invert_yaxis()
    out = out_dir / "topic_clusters.png"
    plt.savefig(out)
    plt.close()
    return out


def chart_gaps(gaps: pd.DataFrame, out_dir: Path) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(10, 5))
    g = gaps.head(8).sort_values("severity_score")
    ax.barh(g.gap, g.severity_score, color=PALETTE["primary"])
    ax.set_xlabel("Severity score (composite of sentiment, reach, channel mix)")
    ax.set_title("Top product gaps by severity (issues recurring across support + customer calls)")
    for i, (gap, score, n_acc, sup, ext) in enumerate(zip(
        g.gap, g.severity_score, g.n_unique_accounts, g.support_calls, g.external_calls
    )):
        ax.text(score + 1, i, f"{int(n_acc)} accts · {int(sup)} support · {int(ext)} customer",
                va="center", fontsize=9)
    out = out_dir / "product_gaps.png"
    plt.savefig(out)
    plt.close()
    return out


def chart_weekly_volume(df: pd.DataFrame, out_dir: Path) -> Path:
    """Weekly call volume by type — surfaces the Detect outage week."""
    _style()
    df = df.copy()
    # Use plain datetime for the week start to avoid pandas period plotting issues
    df["week"] = (
        df.start_time.dt.tz_convert(None).dt.to_period("W").dt.start_time
    )
    pivot = df.pivot_table(
        index="week", columns="call_type", values="meeting_id",
        aggfunc="count", fill_value=0,
    ).sort_index()
    fig, ax = plt.subplots(figsize=(10, 4.5))

    weeks = pivot.index.tolist()
    x = np.arange(len(weeks))
    bottom = np.zeros(len(weeks))
    for col in pivot.columns:
        vals = pivot[col].values
        ax.bar(x, vals, bottom=bottom, color=PALETTE[col], width=0.85,
               label=col, edgecolor="white")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%b %d") for d in weeks],
                       rotation=30, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Number of calls")
    ax.set_title("Weekly call volume — Mar 9 spike marks the Detect outage week")
    ax.legend(title="Call type", frameon=False, loc="upper left")
    out = out_dir / "weekly_volume.png"
    plt.savefig(out)
    plt.close()
    return out


def chart_themes(df: pd.DataFrame, out_dir: Path) -> Path | None:
    """
    Theme reach × sentiment heatmap. Requires df to have a 'themes' column
    containing lists of theme names. Returns None if the column is absent
    or all theme lists are empty (e.g. no GROQ_API_KEY and no cache).
    """
    if "themes" not in df.columns:
        return None
    from collections import Counter
    theme_counts = Counter(t for ts in df["themes"] for t in ts)
    if not theme_counts:
        return None

    theme_data = []
    for theme, n in theme_counts.most_common():
        mask = df["themes"].apply(lambda ts: theme in ts)
        avg_sent = df.loc[mask, "sentiment_score"].mean()
        theme_data.append({"theme": theme, "calls": n, "avg_sentiment": round(avg_sent, 2)})
    theme_df = pd.DataFrame(theme_data)

    _style()
    fig, ax = plt.subplots(figsize=(9, 5))
    norm = plt.Normalize(2.5, 4.5)
    cmap = plt.cm.RdYlGn
    colors = cmap(norm(theme_df["avg_sentiment"].values[::-1]))
    ax.barh(theme_df["theme"].values[::-1], theme_df["calls"].values[::-1], color=colors)
    ax.set_xlabel("Calls referencing theme")
    ax.set_title("10-theme taxonomy — reach and sentiment (red = pain, green = healthy)")
    for i, (n, s) in enumerate(zip(theme_df["calls"].values[::-1],
                                   theme_df["avg_sentiment"].values[::-1])):
        ax.text(n + 0.5, i, f"{n} · μ={s:.2f}", va="center", fontsize=9)

    import matplotlib as mpl
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Avg sentiment (1–5)", shrink=0.8)

    out = out_dir / "themes.png"
    plt.savefig(out)
    plt.close()
    return out


def render_all_charts(
    df: pd.DataFrame,
    df_clustered: pd.DataFrame,
    cluster_terms: dict,
    health: pd.DataFrame,
    gaps: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "sentiment_by_type": chart_sentiment_by_call_type(df, out_dir),
        "sentiment_dist":    chart_sentiment_distribution(df, out_dir),
        "weekly_volume":     chart_weekly_volume(df, out_dir),
        "topic_clusters":    chart_topic_clusters(df_clustered, cluster_terms, out_dir),
        "account_health":    chart_account_health(health, out_dir),
        "account_journeys":  chart_account_journeys(
            df, out_dir,
            ["Brightpath Commerce", "Coastal Living Co",
             "Summit Trust", "Trailhead Marketplace"]
        ),
        "product_gaps":      chart_gaps(gaps, out_dir),
    }
    themes_path = chart_themes(df_clustered, out_dir)
    if themes_path:
        charts["themes"] = themes_path
    return charts
