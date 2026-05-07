# Transcript Intelligence

Hybrid analysis of 100 B2B SaaS call transcripts. Take-home interview
assignment, May 2026.

## The TL;DR

Each transcript ships with a pre-computed `summary.json` containing topics,
sentiment (1–5), and action items. **The per-call ML is already done.**
The work that adds value is the **cross-call layer** — rolling per-call
signal up into decisions someone can act on this week.

This repo delivers that layer:

| Layer | Method | What it produces |
|---|---|---|
| **Classical** | pandas + regex + scikit-learn | Call-type classification, account extraction, cross-channel sentiment gap, account health scoring, action-item rollups |
| **LLM** (Groq Llama 3.3 70B) | One-shot prompt | Consolidates 351 free-form per-call topic labels into 10 themes the org can actually use |

**Why hybrid, not pure-LLM:** the unglamorous work (pandas joins, regex
title parsing, scikit-learn rollups) is faster, cheaper, and more
reliable than asking an LLM to do it. The LLM gets the one job it's
genuinely better at — semantic consolidation across the whole corpus
in a single shot.

## Headline findings

1. **Support runs 0.77 points lower than AM calls** (2.94 vs 3.71 on a
   1–5 scale). AMs see customers at their best; support sees the
   unfiltered version. The two views never get joined today.

2. **Account health watchlist** identifies 4 Critical and 11 At Risk
   accounts. Examples: Northstar Pharma (post-outage scarring),
   Summit Trust (sentiment trending −0.7/call), Brightpath (last
   touchpoint was a competitive evaluation — they're shopping).

3. **Account journeys** reveal AM-vs-support contradictions. Coastal
   Living: AM 4.9/4.8 vs support 2.2/2.2. Trailhead: support 1.4
   (worst in corpus) → renewal 4.5.

4. **The Mar 9 Detect-outage week** had 13 calls (60% above mean) with
   3 customer-facing escalations. Weekly call volume is a leading
   indicator of incidents; today nobody monitors it.

5. **397 action items extracted** with 94% having a named owner.
   Maria Santos has 31 commitments across 13 calls — workload signal
   nobody sees today.

## What's in the repo

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── transcript_intelligence.ipynb   ← lead artifact, runs top-to-bottom
├── src/                                ← reusable package (backup)
│   ├── loader.py     · load JSON bundles, classify call type, extract account
│   ├── topics.py     · TF-IDF + KMeans clustering (deterministic alternative to LLM taxonomy)
│   ├── sentiment.py  · per-call sentiment validation against utterance signal
│   ├── health.py     · account-level health & risk scoring
│   ├── gaps.py       · cross-channel product gap detection
│   ├── actions.py    · owner-attributed action item extraction
│   ├── viz.py        · chart rendering
│   ├── llm_pipeline.py · OpenAI SDK reference (alternative to Groq)
│   └── run_pipeline.py · one-command end-to-end run of the package
├── data/                               ← 100 transcript bundles
└── outputs/
    ├── account_health.csv              ← health scores, risk bands, trends
    ├── action_items.csv                ← 397 owner-attributed commitments
    ├── calls.csv                       ← call-level features
    ├── product_gaps.csv                ← ranked cross-channel gaps
    ├── taxonomy_cache.json             ← 10-theme taxonomy (cached LLM output)
    └── charts/                         ← all visualizations
```

## How to run

### Option A — the notebook (recommended)

```bash
pip install -r requirements.txt
jupyter notebook notebooks/transcript_intelligence.ipynb
```

Run cells top to bottom. Takes ~30 seconds with no API key, ~60 seconds
with the LLM cell live.

### Option B — Colab

Upload the entire repo folder to Drive (or `git clone`), then open the
notebook. Uncomment the `pip install` line in cell 0. The repo path
auto-detects.

### Option C — the package, one command

```bash
pip install -r requirements.txt
python -m src.run_pipeline
```

Produces all CSVs and charts in `outputs/` without any LLM call.

## Optional — the LLM cell

The notebook's taxonomy cell uses Groq's free tier (Llama 3.3 70B).

1. Sign up at https://console.groq.com (free, 30 seconds, no credit card)
2. Copy your API key
3. Set it before launching:
   ```bash
   export GROQ_API_KEY="gsk_..."
   ```
4. Run the cell. Output is cached to `outputs/taxonomy_cache.json`
   so subsequent runs don't re-call the API.

If `GROQ_API_KEY` is unset, the cell falls back to the cached taxonomy
that's committed to the repo — every other cell still produces real
output.

## Five recommended decisions (the punchline)

1. **Ship the CS watchlist** to Customer Success this week. Critical
   accounts get a touchpoint within 7 days.
2. **Standardize on the 10-theme taxonomy.** Retire the 351 free-form
   per-call labels.
3. **Wire weekly call volume as an outage leading indicator.** Auto-page
   when (this week's support volume / trailing 4-week mean) > 1.5.
4. **Triage the 4 Critical accounts** this week. Each has a specific
   story — the touchpoints write themselves from the journey charts.
5. **Add a weekly accountability digest** owned by Eng Ops. Stale
   commitments and workload imbalance are invisible today.

## What I would NOT do

- Re-run sentiment with another LLM. Provided scores correlate 0.94
  with utterance-level signal — replacing them costs tokens for no gain.
- Build a real-time dashboard before validating impact on CS retention.
- Move to embedding-based clustering on a 100-doc corpus. TF-IDF +
  one-shot LLM taxonomy is more interpretable and just as accurate.
  Embeddings are right at 10× the scale, not now.
