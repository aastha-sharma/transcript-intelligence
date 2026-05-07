"""
Account-level health and churn-risk scoring.

The single most useful aggregation for a Customer Success / Sales /
Product team: roll every touchpoint (support, external, escalation)
for an account up into one health view.

Why this matters:
  - Account managers see the polite version of reality on quarterly
    reviews. Support sees the daily friction. Neither has the other's
    context. This module joins them.
  - A 'green' account that's been opening support tickets every two
    weeks is not actually green. Conversely, an account that survived
    a P0 outage with sentiment intact is far stickier than one that
    has been quietly drifting.

Scoring inputs (all derivable from existing fields):
  - Sentiment trend across calls (improving / flat / declining)
  - Volume and severity of support contacts
  - Presence of escalation/urgent calls
  - Presence of churn-risk signals in pre-computed topics
  - Recency: most recent touchpoint dominates the score
"""

from __future__ import annotations

import pandas as pd
import numpy as np


CHURN_TOPIC_TERMS = {
    "churn risk", "renewal", "competitive", "pricing pressure",
    "platform concerns", "service reliability", "account recovery",
}

ESCALATION_KEYWORDS = ("urgent", "escalation", "incident", "outage", "complete loss")


def _has_escalation(title: str) -> bool:
    t = title.lower()
    return any(kw in t for kw in ESCALATION_KEYWORDS)


def _has_churn_signal(topics: list[str]) -> bool:
    return any(any(c in t.lower() for c in CHURN_TOPIC_TERMS) for t in topics)


def account_health_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per account with health-scoring features.

    Input df must have columns:
      account, call_type, sentiment_score, start_time, title, topics

    Output columns (per account):
      n_touchpoints, n_support, n_external, n_escalations,
      avg_sentiment, latest_sentiment, sentiment_trend,
      churn_signals, last_seen, health_score, risk_band
    """
    accounts = df.dropna(subset=["account"]).copy()
    rows = []
    for acc, sub in accounts.groupby("account"):
        sub = sub.sort_values("start_time")
        n = len(sub)
        n_support = (sub.call_type == "support").sum()
        n_external = (sub.call_type == "external").sum()
        n_escalations = sub.title.apply(_has_escalation).sum()
        churn_signals = sub.topics.apply(_has_churn_signal).sum()

        # Trend: linear slope of sentiment over time, normalized
        if n >= 2:
            x = np.arange(n)
            y = sub.sentiment_score.values
            slope = np.polyfit(x, y, 1)[0]
        else:
            slope = 0.0

        avg = sub.sentiment_score.mean()
        latest = sub.sentiment_score.iloc[-1]
        last_seen = sub.start_time.max()

        # Composite health score on roughly [0, 100] for readability.
        # Weighting: recency-weighted sentiment is dominant, then trend,
        # penalty for escalations and high support volume.
        recency_weighted_sent = 0.6 * latest + 0.4 * avg
        score = (
            (recency_weighted_sent - 1) / 4 * 100   # base 0-100 from sentiment
            + 10 * np.tanh(slope)                    # trend bonus/penalty
            - 8 * n_escalations                      # each escalation hurts
            - 3 * max(0, n_support - 1)              # repeated support pain
            - 5 * churn_signals                      # explicit churn topics
        )
        score = float(np.clip(score, 0, 100))

        if score >= 70:
            band = "Healthy"
        elif score >= 50:
            band = "Watch"
        elif score >= 30:
            band = "At Risk"
        else:
            band = "Critical"

        rows.append({
            "account": acc,
            "n_touchpoints": n,
            "n_support": int(n_support),
            "n_external": int(n_external),
            "n_escalations": int(n_escalations),
            "avg_sentiment": round(avg, 2),
            "latest_sentiment": round(latest, 2),
            "sentiment_trend": round(slope, 3),
            "churn_signals": int(churn_signals),
            "last_seen": last_seen,
            "health_score": round(score, 1),
            "risk_band": band,
        })

    out = pd.DataFrame(rows).sort_values("health_score").reset_index(drop=True)
    return out


def watchlist(health: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """The N accounts most worth a customer success conversation today."""
    return health.head(top_n).copy()


def expansion_candidates(health: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """The N healthiest accounts — most likely to expand or refer."""
    return health.tail(top_n).iloc[::-1].copy()
