from pathlib import Path

from src.job_recommender import JobRecommendationSystem, analyze_dataset


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "job_descriptions.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main() -> None:
    analyze_dataset(DATA_PATH, output_dir=OUTPUT_DIR)
    recommender = JobRecommendationSystem.from_csv(DATA_PATH, max_rows_per_title=400)
    recommender.save(ARTIFACT_DIR / "job_recommender.joblib")
    print(f"Artifacts saved to {ARTIFACT_DIR}")
    print(f"Analysis outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
