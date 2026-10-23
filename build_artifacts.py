from pathlib import Path

from src.job_recommender import JobRecommendationSystem, analyze_dataset
from src.job_recommender.semantic import DEFAULT_SEMANTIC_MODEL


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "job_descriptions.csv"
MODEL_PATH = PROJECT_ROOT / "job_recommender.joblib"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main() -> None:
    analyze_dataset(DATA_PATH, output_dir=OUTPUT_DIR)
    recommender = JobRecommendationSystem.from_csv(
        DATA_PATH,
        max_rows_per_title=400,
        semantic_model_name=DEFAULT_SEMANTIC_MODEL,
    )
    recommender.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Analysis outputs saved to {OUTPUT_DIR}")
    print(f"Semantic embeddings built with {DEFAULT_SEMANTIC_MODEL}")


if __name__ == "__main__":
    main()
