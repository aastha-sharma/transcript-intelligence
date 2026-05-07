# Transcript Intelligence

Hybrid analysis of 100 B2B SaaS call transcripts.  
Take-home assignment · Aastha Sharma · May 2026

---

## The Approach

Each transcript ships with a pre-computed `summary.json` (topics, sentiment 1–5, action items).  
**The per-call ML is already done.** The value is in the **cross-call layer** — rolling individual signals up into decisions someone can act on this week.

| Layer | Method | What it produces |
|---|---|---|
| **Classical** | pandas · regex · scikit-learn | Call-type classification, account extraction, sentiment gap analysis, account health scoring, action-item rollups, product gap detection |
| **LLM** (Groq · Llama 4 Scout) | One-shot prompt | Collapses 351 noisy per-call topic labels into 8 clean org-wide themes |

**Why hybrid, not pure-LLM:** the unglamorous work (pandas joins, regex parsing, KMeans clustering) is faster, cheaper, and fully auditable. The LLM gets the one job it's genuinely better at — semantic consolidation in a single call.

---

## Headline Findings

1. **Support runs 0.77 pts lower than AM calls** (2.94 vs 3.71 on 1–5). AMs see customers at their best; support sees the unfiltered version. These two views are never joined today.

2. **Account health watchlist** flags 4 Critical and 11 At Risk accounts. Northstar Pharma (post-outage scarring), Summit Trust (sentiment trending −0.7/call), Brightpath Commerce (last touchpoint was a competitive evaluation — they're shopping).

3. **Account journeys** expose AM-vs-support contradictions. Coastal Living Co: AM scores 4.9/4.8 while support scores 2.2/2.2 in the same period. Two parallel realities, never reconciled.

4. **The Mar 9 Detect-outage week** had 13 calls — 60% above the trailing mean — with 3 customer-facing escalations. Weekly call volume is a leading incident indicator nobody monitors today.

5. **397 action items extracted**, 94% with a named owner. Maria Santos carries 31 commitments across 13 calls — workload concentration invisible to any manager today.

---

## Repo Structure

```
transcript-intelligence/
├── README.md
├── requirements.txt
├── src/
│   ├── run_pipeline.py     ← one-command end-to-end runner
│   ├── loader.py           ← load bundles, classify call type, extract accounts
│   ├── topics.py           ← TF-IDF + KMeans clustering (k=8)
│   ├── sentiment.py        ← cross-channel sentiment analysis & validation
│   ├── health.py           ← account-level health scoring & risk bands
│   ├── gaps.py             ← cross-channel product gap detection
│   ├── actions.py          ← owner-attributed action item extraction
│   ├── viz.py              ← chart rendering (8 charts)
│   └── llm_pipeline.py     ← Groq integration (Llama 4 Scout + Llama 3.3 70B)
├── data/                   ← 100 transcript bundles (6 JSON files each)
└── outputs/
    ├── calls.csv               ← call-level features, one row per call
    ├── account_health.csv      ← health scores, risk bands, sentiment trends
    ├── action_items.csv        ← 397 owner-attributed commitments
    ├── actions_by_call_type.csv
    ├── commitments_by_owner.csv
    ├── product_gaps.csv        ← ranked cross-channel gaps
    ├── sentiment_validation.csv
    ├── topic_frequency_raw.csv ← all 351 raw labels with frequency
    ├── taxonomy_cache.json     ← 8-theme LLM taxonomy (cached)
    ├── cluster_terms.json
    └── charts/                 ← 8 PNGs ready for slide deck
```

---

## How to Run

```bash
pip install -r requirements.txt
python -m src.run_pipeline
```

Runs in ~30 seconds. Produces all CSVs and charts in `outputs/` using the cached taxonomy — no API key needed.

### With Live Groq LLM (regenerates taxonomy)

```bash
export GROQ_API_KEY="gsk_..."
python -m src.run_pipeline
```

Uses **Llama 4 Scout** for structured tasks (taxonomy, action extraction) and **Llama 3.3 70B** for sentiment validation. Free tier at [console.groq.com](https://console.groq.com) — no credit card, 30 seconds to sign up.

The taxonomy output is cached to `outputs/taxonomy_cache.json` so subsequent runs skip the API call.

---

## Five Decisions Worth Making This Week

1. **Ship the CS watchlist.** 4 Critical accounts need a touchpoint within 7 days. The data is in `outputs/account_health.csv`.
2. **Standardize on the 8-theme taxonomy.** Retire 351 free-form labels. Every PM, AM, and exec uses the same vocabulary.
3. **Wire weekly call volume as an outage leading indicator.** Auto-page when (this week's support volume / trailing 4-week mean) > 1.5.
4. **Triage the 4 Critical accounts.** Each has a specific story in the journey charts — the touchpoints write themselves.
5. **Add a weekly accountability digest.** 397 commitments live in meeting notes nobody re-reads. Stale-commitment alerts and workload balance are invisible today.

---

## What I Would NOT Do

- **Re-run sentiment with another LLM.** Pre-computed scores correlate 0.94 with utterance-level signal — replacing them costs tokens for no gain.
- **Build a real-time dashboard before validating retention impact.** Prove the cross-channel health view moves CS outcomes first.
- **Move to embeddings + HDBSCAN now.** On a 100-doc corpus, TF-IDF + one-shot LLM taxonomy is more interpretable and just as accurate. Embeddings are right at 10× scale.
