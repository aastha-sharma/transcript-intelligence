"""
Transcript loader.

Each transcript bundle has 6 JSON files. This module loads them into
clean Python objects and a pandas DataFrame for downstream analysis.

Key design choices:
- We treat the dataset as a *call-level* unit. Each call has a
  pre-computed summary (overallSentiment, topics, actionItems) which we
  use as features rather than re-deriving from scratch.
- We compute call_type (support / external / internal) ourselves by
  inspecting the title and participant emails, because the dataset does
  not ship with this label.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# The org domain that everything in this dataset belongs to.
INTERNAL_DOMAIN = "aegiscloud.com"


@dataclass
class Transcript:
    """One call. Lazily loads transcript utterances on demand."""

    meeting_id: str
    title: str
    start_time: str
    end_time: str
    duration_min: float
    organizer: str
    participants: list[str]
    summary: str
    overall_sentiment: str
    sentiment_score: float
    topics: list[str]
    action_items: list[str]
    key_moments: list[dict] = field(default_factory=list)
    _utterances: list[dict] | None = None
    _bundle_path: Path | None = None

    @property
    def utterances(self) -> list[dict]:
        """Load transcript utterances on demand to keep memory low."""
        if self._utterances is None and self._bundle_path is not None:
            with open(self._bundle_path / "transcript.json") as f:
                self._utterances = json.load(f).get("data", [])
        return self._utterances or []

    @property
    def full_text(self) -> str:
        """Concatenated transcript text for embedding / search."""
        return " ".join(u["sentence"] for u in self.utterances)

    @property
    def participant_count(self) -> int:
        return len(self.participants)


def _classify_call_type(title: str, participants: list[str]) -> str:
    """
    Classify each call as support / external / internal.

    Logic (rule-based, transparent, easy to defend in interview):
      - "Support Case #..." in title -> support
      - Any participant from a non-aegiscloud.com domain -> external
      - Otherwise -> internal

    Edge cases observed in the data:
      - Titles like "Aegis / Brightpath Commerce - ..." are external
        (account meetings) and they typically have a customer email
        attached, so the domain check catches them.
      - "ESCALATION:" titles can be either support or external
        depending on participants — handled by the domain check.
    """
    t = title.lower()

    if re.match(r"^\s*support case\b", t) or "support case #" in t:
        return "support"

    has_external = any(
        p and INTERNAL_DOMAIN not in p.lower() for p in participants
    )
    if has_external:
        return "external"

    return "internal"


def load_bundle(bundle_dir: Path) -> Transcript:
    """Load a single transcript bundle from disk."""
    with open(bundle_dir / "meeting-info.json") as f:
        meeting = json.load(f)
    with open(bundle_dir / "summary.json") as f:
        summary = json.load(f)

    participants = meeting.get("allEmails") or meeting.get("invitees") or []

    return Transcript(
        meeting_id=meeting["meetingId"],
        title=meeting.get("title", ""),
        start_time=meeting.get("startTime", ""),
        end_time=meeting.get("endTime", ""),
        duration_min=float(meeting.get("duration", 0)),
        organizer=meeting.get("organizerEmail", ""),
        participants=participants,
        summary=summary.get("summary", ""),
        overall_sentiment=summary.get("overallSentiment", "neutral"),
        sentiment_score=float(summary.get("sentimentScore", 3.0)),
        topics=summary.get("topics", []),
        action_items=summary.get("actionItems", []),
        key_moments=summary.get("keyMoments", []),
        _bundle_path=bundle_dir,
    )


def load_all(data_dir: str | Path) -> list[Transcript]:
    """Load every bundle under data_dir."""
    data_dir = Path(data_dir)
    bundles = sorted(p for p in data_dir.iterdir() if p.is_dir())
    return [load_bundle(b) for b in bundles]


def to_dataframe(transcripts: list[Transcript]) -> pd.DataFrame:
    """Flatten the corpus into a call-level DataFrame for analysis."""
    rows = []
    for t in transcripts:
        rows.append({
            "meeting_id": t.meeting_id,
            "title": t.title,
            "call_type": _classify_call_type(t.title, t.participants),
            "start_time": pd.to_datetime(t.start_time),
            "duration_min": t.duration_min,
            "participant_count": t.participant_count,
            "organizer": t.organizer,
            "overall_sentiment": t.overall_sentiment,
            "sentiment_score": t.sentiment_score,
            "topics": t.topics,
            "n_topics": len(t.topics),
            "n_action_items": len(t.action_items),
            "summary": t.summary,
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("start_time").reset_index(drop=True)
    return df


# Known accounts in this dataset, derived from titles.
# Used as a known-set to disambiguate support-ticket account extraction
# (where the account name is followed by a free-form issue description).
KNOWN_ACCOUNTS = [
    "Atlas Precision", "Axiom Labs", "Blackridge Investments",
    "Bridgeport Health", "Brightpath Commerce", "Clearwater Medical",
    "Coastal Living Co", "Cobalt Software", "Crestline Wealth",
    "Crestline Wealth Group", "Forge Industries", "Frostbyte AI",
    "Harborview Banking", "Helix Data", "Ironclad Financial",
    "Ironworks Corp", "Keystone Health", "Maplewood Goods",
    "Meridian Capital", "Nimbus Platform", "Northstar Pharma",
    "Nova Retail Group", "Pineridge Systems", "Pinnacle Insurance",
    "Quantum Edge", "Redwood Clinical", "Ridgeline Logistics",
    "Silverline Brands", "Steelpoint Manufacturing", "Stratos Cloud",
    "Summit Trust", "Trailhead Marketplace", "Vanta Health Systems",
]


def extract_account_name(title: str) -> str | None:
    """
    Extract the customer account name from a meeting title.

    Examples:
      "Aegis / Brightpath Commerce - PCI Review" -> "Brightpath Commerce"
      "Support Case #6977 - Brightpath Commerce Slow Backup" -> "Brightpath Commerce"
      "URGENT: Cobalt Software - Aegis Detect Dashboard Down" -> "Cobalt Software"
      "ESCALATION: Northstar Pharma - Detect Outage Impact" -> "Northstar Pharma"
      "Weekly Engineering Standup" -> None (internal call)
    """
    # Pattern 1: "Aegis / <ACCOUNT> - ..."
    m = re.match(r"^\s*aegis\s*/\s*([^-]+?)\s*-", title, re.IGNORECASE)
    if m:
        return _normalize_account(m.group(1).strip())

    # Pattern 2: "URGENT:" or "ESCALATION:" or "Aegis /" prefixes
    m = re.match(
        r"^\s*(?:urgent|escalation):\s*([^-]+?)\s*-", title, re.IGNORECASE
    )
    if m:
        return _normalize_account(m.group(1).strip())

    # Pattern 3: "Support Case #NNNN - <ACCOUNT> <issue description>"
    # Match against the known account list (longest match wins to handle
    # "Crestline Wealth" vs "Crestline Wealth Group" correctly).
    m = re.match(r"^\s*support case #?\d+\s*-\s*(.+)$", title, re.IGNORECASE)
    if m:
        rest = m.group(1)
        for acc in sorted(KNOWN_ACCOUNTS, key=len, reverse=True):
            if rest.lower().startswith(acc.lower()):
                return _normalize_account(acc)

    return None


def _normalize_account(name: str) -> str:
    """Collapse trivial naming variants ('Crestline Wealth Group' -> 'Crestline Wealth')."""
    name = name.strip()
    if name == "Crestline Wealth Group":
        return "Crestline Wealth"
    return name
