"""
Topic discovery on the transcript corpus.

We compare two approaches:
  1. Pre-computed topics from summary.json (the "free" labels)
  2. Bottom-up TF-IDF + KMeans clustering on the raw transcript text

Why both: the per-call topic lists in summary.json are good but
inconsistent — different calls use different vocabulary for the same
underlying issue (e.g. "siem integration" vs "siem connector" vs
"logvault connector"). Clustering on the raw text reveals the latent
themes that span the corpus.

Why TF-IDF + KMeans (not embeddings + HDBSCAN):
  - TF-IDF over a 100-doc corpus is interpretable: we can read off the
    top terms per cluster and immediately see what each cluster is about.
  - For a sample this small, embedding models add latency and a heavy
    dependency without adding much signal. KMeans with k=8 produces
    clean, readable clusters on this corpus.
  - In production at scale we would absolutely move to embeddings +
    HDBSCAN for better noise handling and to surface long-tail topics.
    Documented as a recommendation in the slide deck.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


# Domain-aware stopwords. These words appear in every transcript and
# don't help differentiate topics. Tuned by inspecting the top terms
# from an initial run.
DOMAIN_STOPWORDS = {
    "yeah", "okay", "ok", "right", "like", "just", "kind", "really", "actually",
    "think", "know", "going", "got", "thing", "things", "want", "need", "say",
    "said", "let", "talk", "talking", "good", "look", "looking", "way", "way",
    "team", "guys", "guy", "stuff", "make", "sure", "make sure",
    "aegis", "aegiscloud",  # company name appears everywhere
    "uh", "um", "hmm", "mm", "yep", "huh", "gonna", "wanna", "doesn",
    "didn", "isn", "wasn", "won", "couldn", "wouldn", "ve", "ll", "don",
}


def cluster_topics(
    df: pd.DataFrame,
    transcripts: list,  # list[Transcript] but avoiding circular import
    n_clusters: int = 8,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[int, list[str]]]:
    """
    Cluster transcripts by content using TF-IDF + KMeans.

    Returns:
      - df with new columns: cluster_id, cluster_label
      - dict mapping cluster_id -> top terms (for label inspection)
    """
    docs = [t.full_text for t in transcripts]
    meeting_ids = [t.meeting_id for t in transcripts]

    # English stopwords + our domain-specific ones
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stop = list(ENGLISH_STOP_WORDS.union(DOMAIN_STOPWORDS))

    vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        min_df=3,             # term must appear in >= 3 calls
        max_df=0.7,           # drop terms in > 70% of calls (too generic)
        stop_words=stop,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(docs)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_ids = km.fit_predict(X)

    # Top terms per cluster (interpretable summary)
    terms = np.array(vectorizer.get_feature_names_out())
    centers = km.cluster_centers_
    top_terms_by_cluster = {}
    for i in range(n_clusters):
        top_idx = centers[i].argsort()[::-1][:12]
        top_terms_by_cluster[i] = terms[top_idx].tolist()

    cluster_map = dict(zip(meeting_ids, cluster_ids))
    df = df.copy()
    df["cluster_id"] = df["meeting_id"].map(cluster_map)

    # Auto-label clusters from their top terms.
    # In production this would be a small LLM call — here we use a
    # simple deterministic label based on the top 3 terms.
    cluster_labels = {
        i: _humanize_label(top_terms_by_cluster[i])
        for i in range(n_clusters)
    }
    df["cluster_label"] = df["cluster_id"].map(cluster_labels)

    return df, top_terms_by_cluster


def _humanize_label(top_terms: list[str]) -> str:
    """Take the top TF-IDF terms and produce a short human-readable label."""
    return " / ".join(t.title() for t in top_terms[:3])


def topics_aggregated(transcripts: list) -> pd.DataFrame:
    """
    Roll up the pre-computed per-call topics into a corpus-level frequency table.
    Useful for showing how much overlap/divergence exists in the raw labels.
    """
    counter = Counter()
    for t in transcripts:
        for topic in t.topics:
            counter[topic.strip().lower()] += 1
    return pd.DataFrame(
        counter.most_common(), columns=["topic", "count"]
    )
