"""
Recurring product-gap detection.

The hypothesis: if a topic appears in multiple support tickets AND in
multiple external customer calls, it is a real product gap — not a
one-off issue and not just internal team noise.

Output: a ranked list of product gaps with the supporting evidence
(which calls, which accounts, which sentiment).

This is the kind of finding a Product Manager wants on their desk
weekly. Today they piece this together by scrolling through Slack
threads and Salesforce cases manually.
"""

from __future__ import annotations

import pandas as pd


# Domain-specific term clusters we treat as candidate "gap" topics.
# In production this list would be auto-mined from clustering or LLM
# topic extraction; here we hand-pick a handful that the corpus exhibits
# clearly. This is honest about scope: it's a starter taxonomy.
GAP_TERMS = {
    "logvault_connector":   ["logvault", "log vault"],
    "detect_alerting":      ["detect alert", "alert delay", "alert not firing", "alerts not firing", "false positive"],
    "detect_reliability":   ["detect outage", "detect pipeline", "detect data gap", "detect latency", "dashboard down"],
    "comply_v2":            ["comply v2", "compliance reporting", "comply v two"],
    "sso_mfa":              ["sso", "saml", "mfa", "okta", "single sign", "ldap sync"],
    "billing_overages":     ["overage", "billing", "invoice", "billing dispute", "billing inquiry"],
    "backup_performance":   ["backup", "slow backup", "backup window"],
    "scim_provisioning":    ["scim", "provisioning"],
    "competitive_pressure": ["sentinelshield", "vaultedge", "cybernova", "fortiguard", "competitive"],
}


def detect_gaps(transcripts: list, df: pd.DataFrame) -> pd.DataFrame:
    """
    For each candidate gap, count how many support / external / internal
    calls reference it, and the average sentiment of those calls.
    """
    df_idx = df.set_index("meeting_id")
    rows = []

    for gap_name, terms in GAP_TERMS.items():
        terms_lc = [t.lower() for t in terms]
        hits = []  # list of (meeting_id, call_type, sentiment_score, account)
        for t in transcripts:
            text = (t.title + " " + t.full_text + " " + t.summary).lower()
            if any(term in text for term in terms_lc):
                meta = df_idx.loc[t.meeting_id]
                hits.append({
                    "meeting_id": t.meeting_id,
                    "call_type": meta.call_type,
                    "sentiment": float(meta.sentiment_score),
                    "account": meta.get("account") if "account" in meta else None,
                    "title": t.title,
                })
        if not hits:
            continue
        hits_df = pd.DataFrame(hits)
        rows.append({
            "gap": gap_name,
            "total_calls": len(hits_df),
            "support_calls": (hits_df.call_type == "support").sum(),
            "external_calls": (hits_df.call_type == "external").sum(),
            "internal_calls": (hits_df.call_type == "internal").sum(),
            "n_unique_accounts": hits_df.account.nunique() if "account" in hits_df else 0,
            "avg_sentiment": round(hits_df.sentiment.mean(), 2),
            "min_sentiment": round(hits_df.sentiment.min(), 2),
        })

    out = pd.DataFrame(rows)
    if len(out):
        # A real gap shows up across BOTH support and external calls
        out["cross_channel"] = (out.support_calls > 0) & (out.external_calls > 0)
        # Severity: prioritize gaps with low sentiment and broad customer reach
        out["severity_score"] = (
            (5 - out.avg_sentiment) * 20         # lower sentiment -> more severe
            + out.n_unique_accounts * 5           # broader reach -> more severe
            + out.cross_channel.astype(int) * 15  # bonus if cross-channel
        ).round(1)
        out = out.sort_values("severity_score", ascending=False).reset_index(drop=True)
    return out


def gap_evidence(transcripts: list, df: pd.DataFrame, gap_name: str) -> pd.DataFrame:
    """List every call that references a given gap, with metadata."""
    if gap_name not in GAP_TERMS:
        raise ValueError(f"Unknown gap: {gap_name}")
    terms_lc = [t.lower() for t in GAP_TERMS[gap_name]]
    df_idx = df.set_index("meeting_id")
    rows = []
    for t in transcripts:
        text = (t.title + " " + t.full_text + " " + t.summary).lower()
        if any(term in text for term in terms_lc):
            meta = df_idx.loc[t.meeting_id]
            rows.append({
                "date": meta.start_time,
                "call_type": meta.call_type,
                "sentiment": meta.sentiment_score,
                "title": t.title,
            })
    return pd.DataFrame(rows).sort_values("date") if rows else pd.DataFrame()
