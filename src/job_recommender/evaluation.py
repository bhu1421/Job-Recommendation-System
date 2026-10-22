"""Evaluation helpers for manually reviewed recommendation rankings."""

from __future__ import annotations

import hashlib

import pandas as pd


REQUIRED_COLUMNS = {"query_id", "query", "rank", "relevant"}


def make_review_sheet(results: pd.DataFrame, query: str, method: str = "tfidf") -> pd.DataFrame:
    """Create a CSV-ready sheet for human relevance judgments."""
    query_id = hashlib.sha1(query.encode("utf-8")).hexdigest()[:10]
    review = results.copy()
    review.insert(0, "query_id", query_id)
    review.insert(1, "query", query)
    review.insert(2, "method", method)
    review.insert(3, "rank", range(1, len(review) + 1))
    review["relevant"] = ""  # Fill with 1 or 0 after reviewing each result.
    columns = ["query_id", "query", "method", "rank", "Job Id", "Job Title", "Company", "matched_terms", "relevant"]
    return review[[column for column in columns if column in review.columns]]


def evaluate_judgments(judgments: pd.DataFrame, k: int = 5) -> tuple[pd.DataFrame, dict]:
    """Calculate per-query and mean Precision@k from reviewed ranked results."""
    missing = REQUIRED_COLUMNS - set(judgments.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if k < 1:
        raise ValueError("k must be at least 1")

    reviewed = judgments.copy()
    reviewed["rank"] = pd.to_numeric(reviewed["rank"], errors="raise")
    reviewed["relevant"] = pd.to_numeric(reviewed["relevant"], errors="raise")
    if not reviewed["relevant"].isin([0, 1]).all():
        raise ValueError("relevant must contain only 1 (relevant) or 0 (not relevant)")

    top_results = reviewed.loc[reviewed["rank"] <= k]
    group_columns = ["query_id", "query"]
    if "method" in top_results.columns:
        group_columns.append("method")
    scores = (
        top_results.groupby(group_columns, as_index=False)["relevant"]
        .sum()
        .rename(columns={"relevant": "relevant_in_top_k"})
    )
    scores["precision_at_k"] = scores["relevant_in_top_k"] / k
    summary = {
        "queries_reviewed": int(scores["query_id"].nunique()),
        "k": k,
        "mean_precision_at_k": (
            round(float(scores["precision_at_k"].mean()), 4) if not scores.empty else 0.0
        ),
    }
    if "method" in scores.columns:
        summary["precision_at_k_by_method"] = (
            scores.groupby("method")["precision_at_k"].mean().round(4).to_dict()
        )
    return scores, summary
