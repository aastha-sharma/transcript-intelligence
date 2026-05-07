"""
LLM-based augmentation pipeline using Groq (Llama 3.3 70B).

Three LLM jobs — each chosen because classical methods can't do them cheaply:
  1. Topic taxonomy consolidation: collapses 351 noisy per-call labels into
     a clean org-wide taxonomy in a single prompt call.
  2. Structured action item extraction: adds due-date hints, priority, and
     dependency fields that the regex parser in actions.py can't recover.
  3. Sentiment validation: independent LLM grading lets us spot drift
     between the pre-computed scores and a fresh read.

All three use Groq's free tier (Llama 3.3 70B) for cost and speed.
Output is cached so re-runs are instantaneous.

Usage:
    export GROQ_API_KEY="gsk_..."
    python -m src.llm_pipeline           # runs all three jobs
    python -m src.llm_pipeline taxonomy  # taxonomy only
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    from groq import Groq
except ImportError:
    Groq = None

# Llama 4 Scout: best for structured / JSON tasks (function calling support, large context)
MODEL_STRUCTURED = "meta-llama/llama-4-scout-17b-16e-instruct"
# Llama 3.3 70B: fast, cheap, reliable for simpler text-generation tasks
MODEL_TEXT = "llama-3.3-70b-versatile"

TAXONOMY_PROMPT = """You are designing a topic taxonomy for a B2B cybersecurity SaaS company's call transcripts.

Below are free-form topic labels with their corpus frequency, applied across ~100 calls. Many are duplicates or near-duplicates.

Consolidate them into 8-12 clean themes. Each theme:
- Has a short noun-phrase name (e.g. "Compliance & Audit")
- A one-sentence description
- An `absorbs` list with the original labels it covers

Return JSON only, no prose, in this shape:
{{"themes": [{{"name": "...", "description": "...", "absorbs": ["...", "..."]}}]}}

Frequency list:
{frequency_list}
"""

SENTIMENT_VALIDATION_PROMPT = """You are auditing call-level sentiment scores for a B2B SaaS company.

For the transcript summary below, estimate an overall sentiment score from 1.0 (very negative) to 5.0 (very positive). Consider:
- Customer/speaker frustration vs satisfaction
- Whether outcomes were resolved or remain open
- Tone of language, not just literal positive/negative words

Return JSON only, no prose:
{{"score": <float 1.0-5.0>, "reasoning": "<one short sentence>"}}

Summary:
\"\"\"{summary}\"\"\"

Pre-computed score for calibration: {labeled_score}
"""

ACTION_ITEM_PROMPT = """Extract structured action items from this call's action item list.

For each item return:
  - owner: person's full name, or "Unassigned"
  - action: imperative-form description (clean up the text)
  - due_hint: any explicit deadline ("EOD tomorrow", "by Friday"), else null
  - priority: "high" | "medium" | "low" based on language and urgency
  - depends_on: other person/team named as a dependency, else null

Return JSON only:
{{"items": [{{"owner": "...", "action": "...", "due_hint": null, "priority": "medium", "depends_on": null}}]}}

Action items:
{action_items}
"""

THEME_ASSIGNMENT_PROMPT = """You have the following topic taxonomy for a B2B SaaS company:

{taxonomy_json}

Given this call's topic labels, assign it to one or more themes from the taxonomy above.
Return only theme names that clearly apply (0-3 themes per call is normal).

Return JSON only:
{{"themes": ["Theme Name 1", "Theme Name 2"]}}

Call topics: {topics}
"""


def _client() -> Any:
    if Groq is None:
        raise RuntimeError("groq package not installed. Run: pip install groq")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set. Export your key before calling LLM functions.")
    return Groq(api_key=api_key)


def _chat(client: Any, prompt: str, model: str = MODEL_TEXT) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def build_taxonomy(
    topic_frequency: list[tuple[str, int]],
    cache_path: str | Path | None = None,
) -> dict:
    """
    Consolidate noisy per-call topic labels into a clean org-wide taxonomy.
    Uses Llama 4 Scout for better semantic grouping. Result is cached.
    """
    if cache_path:
        cache_path = Path(cache_path)
        if cache_path.exists():
            print(f"  taxonomy: using cache ({cache_path})")
            return json.loads(cache_path.read_text())

    client = _client()
    freq_text = "\n".join(f"  {t} ({c})" for t, c in topic_frequency)
    result = json.loads(_chat(client, TAXONOMY_PROMPT.format(frequency_list=freq_text),
                              model=MODEL_STRUCTURED))

    if cache_path:
        cache_path.write_text(json.dumps(result, indent=2))
        print(f"  taxonomy: {len(result['themes'])} themes generated and cached")

    return result


def validate_sentiment(summary: str, labeled_score: float) -> dict:
    """
    LLM-graded sentiment for a single call.
    Returns {"score": float, "reasoning": str}.
    """
    client = _client()
    prompt = SENTIMENT_VALIDATION_PROMPT.format(
        summary=summary[:4000],
        labeled_score=labeled_score,
    )
    return json.loads(_chat(client, prompt))


def extract_structured_actions(raw_action_items: list[str]) -> dict:
    """
    Parse raw "Owner: action" strings into structured records with
    due_hint, priority, and depends_on fields.
    Uses Llama 4 Scout (function-calling support → reliable JSON).
    Returns {"items": [...]}.
    """
    client = _client()
    items_text = "\n".join(f"  - {x}" for x in raw_action_items)
    return json.loads(_chat(client, ACTION_ITEM_PROMPT.format(action_items=items_text),
                            model=MODEL_STRUCTURED))


def assign_themes_to_call(topics: list[str], taxonomy: dict) -> list[str]:
    """
    Use the LLM to map a single call's raw topics to taxonomy theme names.
    Uses Llama 4 Scout for reliable structured JSON output.
    Returns a list of theme names that apply.
    """
    client = _client()
    taxonomy_summary = "\n".join(
        f'- "{t["name"]}": {t["description"]}' for t in taxonomy["themes"]
    )
    prompt = THEME_ASSIGNMENT_PROMPT.format(
        taxonomy_json=taxonomy_summary,
        topics=", ".join(topics),
    )
    result = json.loads(_chat(client, prompt, model=MODEL_STRUCTURED))
    return result.get("themes", [])


def run_batch_sentiment_validation(
    transcripts: list,
    df,
    sample_size: int = 20,
    out_path: str | Path | None = None,
) -> list[dict]:
    """
    Validate sentiment scores on a random sample of calls.
    Writes results to out_path as JSON if provided.
    Returns list of {meeting_id, labeled_score, llm_score, delta, reasoning}.
    """
    import random
    sample = random.sample(transcripts, min(sample_size, len(transcripts)))
    df_idx = df.set_index("meeting_id")
    results = []
    for t in sample:
        meta = df_idx.loc[t.meeting_id]
        labeled = float(meta.sentiment_score)
        try:
            validated = validate_sentiment(t.summary, labeled)
            results.append({
                "meeting_id": t.meeting_id,
                "title": t.title,
                "labeled_score": labeled,
                "llm_score": validated["score"],
                "delta": round(abs(labeled - validated["score"]), 2),
                "reasoning": validated["reasoning"],
            })
        except Exception as e:
            results.append({"meeting_id": t.meeting_id, "error": str(e)})

    if out_path:
        Path(out_path).write_text(json.dumps(results, indent=2))

    avg_delta = sum(r.get("delta", 0) for r in results) / len(results)
    print(f"  sentiment validation: avg |delta| = {avg_delta:.2f} over {len(results)} calls")
    return results


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add repo root to path when running as script
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.loader import load_all, to_dataframe, extract_account_name
    from src.topics import topics_aggregated

    data_dir = Path("data")
    out_dir = Path("outputs")

    print("Loading corpus...")
    transcripts = load_all(data_dir)
    df = to_dataframe(transcripts)
    df["account"] = df["title"].apply(extract_account_name)

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode in ("all", "taxonomy"):
        print("Building LLM taxonomy...")
        topic_freq = topics_aggregated(transcripts)
        taxonomy = build_taxonomy(
            list(zip(topic_freq.topic, topic_freq["count"])),
            cache_path=out_dir / "taxonomy_cache.json",
        )
        for t in taxonomy["themes"]:
            print(f"  · {t['name']:30s} ({len(t['absorbs'])} labels absorbed)")

    if mode in ("all", "sentiment"):
        print("Validating sentiment scores (sample of 20)...")
        results = run_batch_sentiment_validation(
            transcripts, df,
            sample_size=20,
            out_path=out_dir / "sentiment_llm_validation.json",
        )

    if mode in ("all", "actions"):
        print("Extracting structured actions (first 5 calls as demo)...")
        for t in transcripts[:5]:
            if t.action_items:
                result = extract_structured_actions(t.action_items)
                print(f"\n  {t.title[:60]}")
                for item in result.get("items", []):
                    print(f"    [{item.get('priority','?'):6s}] {item.get('owner','?')}: {item.get('action','?')[:70]}")
