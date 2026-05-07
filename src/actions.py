"""
Action item & commitment tracking across the corpus.

Every summary.json includes a list of actionItems in the form:
    "Jordan Whitfield: Clean up the competitive deck and share by EOD"
    "Tom Bradley: Review the updated deck before next week's meeting"

Across 100 calls there are hundreds of these. Aggregating them by owner
gives you an organization-wide commitment register: who owes what.

Use cases this enables:
  - Manager dashboard: how many open commitments per person
  - Topic dashboard: which themes generate the most follow-up work
  - Stale-commitment alerts: items mentioned in old calls and never
    referenced again

This is a stub-grade implementation. The owner-extraction is a regex
because the format is consistent in this dataset; in production you'd
want NER or an LLM extraction pass for messier free-form summaries.
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd


# Pattern: "Name Name: action description"
# Some action items don't have an owner prefix — those become "Unassigned".
OWNER_PATTERN = re.compile(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*:\s*(.+)$")


def parse_action_item(item: str) -> tuple[str, str]:
    """Split 'Name: action' into (owner, action). Returns (Unassigned, item) on failure."""
    m = OWNER_PATTERN.match(item.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "Unassigned", item.strip()


def action_items_table(transcripts: list, df: pd.DataFrame) -> pd.DataFrame:
    """
    Long-format DataFrame: one row per action item across the corpus.
    Joined back to the call's call_type and date for filtering.
    """
    df_idx = df.set_index("meeting_id")
    rows = []
    for t in transcripts:
        meta = df_idx.loc[t.meeting_id]
        for item in t.action_items:
            owner, action = parse_action_item(item)
            rows.append({
                "meeting_id": t.meeting_id,
                "call_type": meta.call_type,
                "start_time": meta.start_time,
                "title": t.title,
                "owner": owner,
                "action": action,
            })
    return pd.DataFrame(rows)


def commitments_by_owner(actions: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Top N most committed-to people across the corpus."""
    counts = (
        actions[actions.owner != "Unassigned"]
        .groupby("owner")
        .agg(n_commitments=("action", "count"),
             n_calls=("meeting_id", "nunique"),
             last_commitment=("start_time", "max"))
        .sort_values("n_commitments", ascending=False)
        .head(top_n)
        .reset_index()
    )
    return counts


def commitments_by_call_type(actions: pd.DataFrame) -> pd.DataFrame:
    """How many action items per call type? Surfaces where work is generated."""
    return (
        actions.groupby("call_type")
        .agg(total_actions=("action", "count"),
             n_calls=("meeting_id", "nunique"))
        .assign(actions_per_call=lambda d: (d.total_actions / d.n_calls).round(2))
    )


def keyword_themes_in_actions(actions: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Cheap keyword-frequency to see what kinds of work the org is committing to.
    NOT a full topic model — just enough to populate a slide.
    """
    text = " ".join(actions.action.tolist()).lower()
    words = re.findall(r"\b[a-z]{4,}\b", text)
    stop = {
        "with", "from", "this", "that", "have", "will", "into", "next", "week",
        "tomorrow", "today", "before", "after", "share", "send", "review",
        "schedule", "follow", "update", "discuss", "team", "meeting", "call",
        "provide", "ensure", "their", "they", "them", "should", "would", "could",
    }
    words = [w for w in words if w not in stop]
    counts = Counter(words).most_common(top_n)
    return pd.DataFrame(counts, columns=["term", "frequency"])
