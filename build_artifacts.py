import argparse
from pathlib import Path

from src.job_recommender import JobRecommendationSystem, analyze_dataset
from src.job_recommender.semantic import DEFAULT_SEMANTIC_MODEL


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "job_descriptions.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build analysis and recommender artifacts.")
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Also build sentence embeddings (requires requirements-semantic.txt).",
    )
    args = parser.parse_args()

    analyze_dataset(DATA_PATH, output_dir=OUTPUT_DIR)
    recommender = JobRecommendationSystem.from_csv(
        DATA_PATH,
        max_rows_per_title=400,
        semantic_model_name=DEFAULT_SEMANTIC_MODEL if args.semantic else None,
    )
    recommender.save(ARTIFACT_DIR / "job_recommender.joblib")
    print(f"Artifacts saved to {ARTIFACT_DIR}")
    print(f"Analysis outputs saved to {OUTPUT_DIR}")
    if args.semantic:
        print(f"Semantic embeddings built with {DEFAULT_SEMANTIC_MODEL}")


if __name__ == "__main__":
    main()
