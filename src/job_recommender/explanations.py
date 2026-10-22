"""Small, human-readable explanations for lexical job matches."""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def matching_terms(
    query: str,
    job_text: str,
    vectorizer: TfidfVectorizer,
    limit: int = 5,
) -> str:
    """Return the highest-weight terms present in both query and job text."""
    analyzer = vectorizer.build_analyzer()
    query_terms = set(analyzer(query))
    job_terms = set(analyzer(job_text))
    shared = query_terms & job_terms
    if not shared:
        return "No exact keyword overlap; ranked by overall text similarity"

    weights = vectorizer.transform([query]).toarray()[0]
    term_weights = dict(zip(vectorizer.get_feature_names_out(), weights))
    best_terms = sorted(shared, key=lambda term: term_weights.get(term, 0), reverse=True)
    return ", ".join(best_terms[:limit])


def add_explanations(
    results: pd.DataFrame,
    query: str,
    vectorizer: TfidfVectorizer,
) -> pd.DataFrame:
    """Append an explanation column without changing the ranking."""
    output = results.copy()
    output["matched_terms"] = output.apply(
        lambda row: matching_terms(
            query,
            " ".join(
                str(row.get(column, ""))
                for column in ("Job Title", "Role", "skills", "Job Description", "Responsibilities")
            ),
            vectorizer,
        ),
        axis=1,
    )
    return output
