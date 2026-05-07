"""
End-to-end pipeline. Run this from the repo root:

    python -m src.run_pipeline

Produces:
    outputs/
        calls.csv                       — call-level features (one row per call)
        action_items.csv                — exploded action items with owners
        account_health.csv              — account-level health scoring
        product_gaps.csv                — ranked product gap candidates
        commitments_by_owner.csv        — owner-level commitment counts
        topic_frequency_raw.csv         — raw per-call topic label counts
        sentiment_validation.csv        — utterance-vs-summary score comparison
        taxonomy_cache.json             — LLM-generated 10-theme taxonomy (cached)
        cluster_terms.json              — top TF-IDF terms per topic cluster
        sentiment_llm_validation.json   — LLM sentiment spot-check (if GROQ_API_KEY set)
        charts/*.png                    — visualizations for slide deck

LLM enrichment (Groq Llama 3.3 70B):
    Set GROQ_API_KEY to run the taxonomy call live. Without it, the pipeline
    falls back to the committed taxonomy_cache.json — every other step still
    produces real output from 100 transcripts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.actions import (
    action_items_table, commitments_by_owner, commitments_by_call_type,
)
from src.gaps import detect_gaps
from src.health import account_health_table
from src.loader import extract_account_name, load_all, to_dataframe
from src.sentiment import call_level_summary, validate_call_sentiment
from src.topics import cluster_topics, topics_aggregated
from src.viz import render_all_charts


def _load_or_build_taxonomy(topic_freq, cache_path: Path) -> dict | None:
    """
    Returns the Groq-built taxonomy if GROQ_API_KEY is set, else loads cache.
    Returns None if neither is available (non-fatal — themes column will be empty).
    """
    if cache_path.exists():
        with open(cache_path) as f:
            taxonomy = json.load(f)
        if not os.getenv("GROQ_API_KEY"):
            print(f"  LLM taxonomy: loaded from cache ({len(taxonomy['themes'])} themes)")
            return taxonomy

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("  LLM taxonomy: GROQ_API_KEY not set and no cache found — skipping theme assignment")
        return None

    print("  LLM taxonomy: building via Groq Llama 4 Scout (live API call)...")
    from src.llm_pipeline import build_taxonomy
    freq_pairs = list(zip(topic_freq.topic, topic_freq["count"]))
    taxonomy = build_taxonomy(freq_pairs, cache_path=cache_path)
    print(f"  LLM taxonomy: {len(taxonomy['themes'])} themes → cached")
    return taxonomy


def _assign_themes(df, taxonomy: dict | None):
    """Map per-call topic labels to the LLM taxonomy themes."""
    if taxonomy is None:
        df["themes"] = [[] for _ in range(len(df))]
        return df

    label_to_theme: dict[str, list[str]] = {}
    for theme in taxonomy["themes"]:
        for lbl in theme.get("absorbs", []):
            label_to_theme.setdefault(lbl.lower().strip(), []).append(theme["name"])

    def map_themes(topics):
        found = []
        for t in topics:
            for theme in label_to_theme.get(t.lower().strip(), []):
                if theme not in found:
                    found.append(theme)
        return found

    df = df.copy()
    df["themes"] = df["topics"].apply(map_themes)
    return df


def main(data_dir: str = "data", out_dir: str = "outputs") -> dict:
    out_dir = Path(out_dir)
    charts_dir = out_dir / "charts"
    out_dir.mkdir(exist_ok=True)
    charts_dir.mkdir(exist_ok=True)

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print("Loading transcripts...")
    transcripts = load_all(data_dir)
    df = to_dataframe(transcripts)
    df["account"] = df["title"].apply(extract_account_name)
    print(f"  {len(transcripts)} transcripts loaded "
          f"({df.start_time.min().date()} → {df.start_time.max().date()})")

    # ── 2. Topic clustering (classical TF-IDF + KMeans) ─────────────────────
    print("Running topic clustering (TF-IDF + KMeans k=8)...")
    df_clustered, cluster_terms = cluster_topics(df, transcripts, n_clusters=8)
    topic_freq = topics_aggregated(transcripts)
    print(f"  {len(topic_freq)} distinct labels across {len(df)} calls")

    # ── 3. LLM taxonomy (Groq) ───────────────────────────────────────────────
    print("Building topic taxonomy...")
    taxonomy = _load_or_build_taxonomy(topic_freq, out_dir / "taxonomy_cache.json")
    df_clustered = _assign_themes(df_clustered, taxonomy)

    # ── 4. Sentiment analysis ────────────────────────────────────────────────
    print("Running sentiment analysis...")
    sentiment_summary = call_level_summary(df)
    sentiment_validation = validate_call_sentiment(transcripts, df)

    # ── 5. Account health ────────────────────────────────────────────────────
    print("Building account health table...")
    health = account_health_table(df_clustered)

    # ── 6. Product gap detection ─────────────────────────────────────────────
    print("Detecting product gaps...")
    gaps = detect_gaps(transcripts, df_clustered)

    # ── 7. Action items ──────────────────────────────────────────────────────
    print("Extracting action items...")
    actions = action_items_table(transcripts, df_clustered)
    owners = commitments_by_owner(actions)
    by_type = commitments_by_call_type(actions)
    by_type.to_csv(out_dir / "actions_by_call_type.csv", index=True)

    # ── 8. Charts ────────────────────────────────────────────────────────────
    print("Rendering charts...")
    chart_paths = render_all_charts(
        df_clustered, df_clustered, cluster_terms, health, gaps, charts_dir
    )
    print(f"  {len(chart_paths)} charts written: {', '.join(chart_paths.keys())}")

    # ── 9. Write outputs ─────────────────────────────────────────────────────
    print("Writing outputs...")
    df_clustered.drop(columns=["topics"]).to_csv(out_dir / "calls.csv", index=False)
    actions.to_csv(out_dir / "action_items.csv", index=False)
    health.to_csv(out_dir / "account_health.csv", index=False)
    gaps.to_csv(out_dir / "product_gaps.csv", index=False)
    topic_freq.to_csv(out_dir / "topic_frequency_raw.csv", index=False)
    owners.to_csv(out_dir / "commitments_by_owner.csv", index=False)
    sentiment_validation.to_csv(out_dir / "sentiment_validation.csv", index=False)
    with open(out_dir / "cluster_terms.json", "w") as f:
        json.dump({str(k): v for k, v in cluster_terms.items()}, f, indent=2)

    # ── 10. Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 56)
    print("KEY FINDINGS")
    print("=" * 56)

    print(f"\nCorpus: {len(df)} calls")
    print(f"  External (AM):   {(df.call_type == 'external').sum()}")
    print(f"  Internal:        {(df.call_type == 'internal').sum()}")
    print(f"  Support:         {(df.call_type == 'support').sum()}")

    print(f"\nSentiment by call type:")
    print(sentiment_summary[["mean", "median", "std"]].to_string())

    gap_delta = (sentiment_summary.loc["external", "mean"]
                 - sentiment_summary.loc["support", "mean"])
    print(f"\n  → AM vs support gap: {gap_delta:.2f} pts  "
          f"({sentiment_summary.loc['external','mean']:.2f} vs "
          f"{sentiment_summary.loc['support','mean']:.2f})")

    print(f"\nAccounts identified: {df.account.notna().sum()} / {len(df)} calls "
          f"({df.account.nunique()} distinct)")
    print(f"Account risk bands:")
    for band in ["Critical", "At Risk", "Watch", "Healthy"]:
        n = (health.risk_band == band).sum()
        print(f"  {band:9s}  {n}")

    if taxonomy:
        print(f"\nLLM taxonomy: {len(taxonomy['themes'])} themes "
              f"(collapsed {len(topic_freq)} raw labels)")

    print(f"\nProduct gaps ranked by severity:")
    for _, row in gaps.head(5).iterrows():
        print(f"  {row.gap:30s}  score={row.severity_score:.1f}  "
              f"({int(row.total_calls)} calls, {int(row.n_unique_accounts)} accts)")

    print(f"\nAction items: {len(actions)} total  "
          f"({len(actions[actions.owner != 'Unassigned'])} with named owner)")
    print(f"Top 5 by workload:")
    for _, row in owners.head(5).iterrows():
        print(f"  {row.owner:20s}  {int(row.n_commitments)} commitments across "
              f"{int(row.n_calls)} calls")

    print(f"\nCharts written to: {charts_dir}/")
    print("=" * 56)

    return {
        "n_calls": len(df),
        "n_accounts": int(df.account.notna().sum()),
        "n_health_critical": int((health.risk_band == "Critical").sum()),
        "n_actions": len(actions),
        "am_support_gap": round(gap_delta, 2),
    }


if __name__ == "__main__":
    main()
