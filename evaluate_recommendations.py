"""Summarize manually reviewed recommendation results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.job_recommender.evaluation import evaluate_judgments


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate mean Precision@k from a reviewed CSV file."
    )
    parser.add_argument("judgments", type=Path, help="CSV downloaded from the app and reviewed")
    parser.add_argument("--k", type=int, default=5, help="Rank cutoff (default: 5)")
    args = parser.parse_args()

    scores, summary = evaluate_judgments(pd.read_csv(args.judgments), k=args.k)
    print(scores.to_string(index=False))
    print(
        f"\nQueries reviewed: {summary['queries_reviewed']}"
        f"\nMean Precision@{summary['k']}: {summary['mean_precision_at_k']:.3f}"
    )
    for method, score in summary.get("precision_at_k_by_method", {}).items():
        print(f"{method} mean Precision@{summary['k']}: {score:.3f}")


if __name__ == "__main__":
    main()
