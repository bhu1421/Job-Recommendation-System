"""Streamlit interface for job discovery and recommendation review."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.job_recommender import (
    JobRecommendationSystem,
    add_explanations,
    analyze_dataset,
    make_review_sheet,
)
from src.job_recommender.semantic import DEFAULT_SEMANTIC_MODEL


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "job_descriptions.csv"
MODEL_PATH = PROJECT_ROOT / "job_recommender.joblib"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ANALYSIS_VERSION = 2


@st.cache_resource(show_spinner=False, max_entries=2)
def load_recommender(signature: tuple[int, int]) -> JobRecommendationSystem:
    """Reuse a model only when it was built from the current dataset."""
    if MODEL_PATH.exists():
        saved = JobRecommendationSystem.load(MODEL_PATH)
        if (
            getattr(saved, "artifact_version", None) == 3
            and getattr(saved, "source_signature", None) == signature
            and saved.semantic_available
        ):
            return saved

    recommender = JobRecommendationSystem.from_csv(
        DATA_PATH,
        max_rows_per_title=400,
        semantic_model_name=DEFAULT_SEMANTIC_MODEL,
    )
    recommender.save(MODEL_PATH)
    return recommender


@st.cache_data(show_spinner=False, max_entries=2)
def load_analysis(signature: tuple[int, int], version: int) -> dict:
    """Cache dataset statistics using the source file signature as the key."""
    return analyze_dataset(DATA_PATH, output_dir=OUTPUT_DIR)


def show_results(
    results, query: str, method: str, recommender: JobRecommendationSystem, key: str
) -> None:
    """Show recommendations, lexical clues, and a human-review download."""
    explained = add_explanations(results, query, recommender.vectorizer)
    visible_columns = [
        "Job Title", "Role", "Company", "Country", "Work Type", "Experience",
        "similarity_score", "matched_terms", "skills", "Job Description",
    ]
    display = explained[visible_columns].rename(
        columns={"similarity_score": "match_score", "matched_terms": "why_it_matched"}
    )
    st.dataframe(display, hide_index=True)
    st.download_button(
        "Download for relevance review",
        data=make_review_sheet(explained, query, method).to_csv(index=False),
        file_name=f"recommendation_review_{key}_{method}.csv",
        mime="text/csv",
        key=f"download_{key}",
    )


st.set_page_config(page_title="Job recommendation system", layout="wide")
st.title("Job recommendation system")
st.caption(
    "Explore similar job listings with keyword matching, semantic embeddings, or both."
)

if not DATA_PATH.exists():
    st.error(
        "Dataset not found. Place `job_descriptions.csv` next to `app.py`, then restart the app."
    )
    st.stop()

data_stat = DATA_PATH.stat()
data_signature = (data_stat.st_size, data_stat.st_mtime_ns)

with st.spinner("Loading dataset analysis and recommender. The first load can take a while."):
    analysis = load_analysis(data_signature, ANALYSIS_VERSION)
    recommender = load_recommender(data_signature)

metric_columns = st.columns(4)
metric_columns[0].metric("Total jobs", f"{analysis['row_count']:,}")
metric_columns[1].metric("Unique job titles", analysis["unique_job_titles"])
metric_columns[2].metric("Average description length", analysis["avg_job_description_chars"])
metric_columns[3].metric("Average skills length", analysis["avg_skills_chars"])

with st.expander("Dataset highlights"):
    chart_columns = st.columns(2)
    title_chart = OUTPUT_DIR / "top_job_titles.png"
    work_chart = OUTPUT_DIR / "top_work_types.png"
    if title_chart.exists():
        chart_columns[0].image(
            str(title_chart), caption="Most common job titles", alt="Bar chart of common job titles"
        )
    if work_chart.exists():
        chart_columns[1].image(
            str(work_chart),
            caption="Work type distribution",
            alt="Horizontal bar chart with listing counts and dataset percentages",
        )
    work_counts = list(analysis["top_work_types"].values())
    if work_counts:
        lowest, highest = min(work_counts), max(work_counts)
        st.caption(
            "Work types are nearly balanced: the displayed categories range from "
            f"{lowest:,} ({lowest / analysis['row_count']:.1%}) to "
            f"{highest:,} ({highest / analysis['row_count']:.1%}) of all listings."
        )

methods = {
    "Keyword (TF-IDF)": "tfidf",
    "Semantic embeddings": "semantic",
    "Hybrid (rank fusion)": "hybrid",
}

method_label = st.selectbox("Recommendation method", list(methods))
method = methods[method_label]
st.caption(
    "Hybrid uses reciprocal-rank fusion to combine TF-IDF and semantic rankings. "
    "Scores are for ordering results and are not probabilities."
    if method == "hybrid"
    else "Scores rank listings within this method; they are not probabilities."
)

st.subheader("Find jobs similar to a title")
countries = ["Any"] + sorted(recommender.jobs["Country"].dropna().unique().tolist())
work_types = ["Any"] + sorted(recommender.jobs["Work Type"].dropna().unique().tolist())
experience_ranges = ["Any"] + sorted(
    recommender.jobs["Experience"].dropna().unique().tolist()
)
with st.form("title_search"):
    selected_title = st.selectbox("Job title", recommender.available_job_titles())
    filter_columns = st.columns(3)
    selected_country = filter_columns[0].selectbox("Country", countries)
    selected_work_type = filter_columns[1].selectbox("Work type", work_types)
    selected_experience = filter_columns[2].selectbox(
        "Experience range", experience_ranges
    )
    top_k = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)
    title_submit = st.form_submit_button("Get recommendations", type="primary")

if title_submit:
    try:
        results = recommender.recommend_by_job_title(
            selected_title,
            top_k=top_k,
            country=selected_country,
            work_type=selected_work_type,
            experience=selected_experience,
            method=method,
        )
        if results.empty:
            st.info("No jobs match those filters. Try another country or work type.")
        else:
            show_results(results, selected_title, method, recommender, "title")
    except (ValueError, RuntimeError) as error:
        st.error(str(error))

st.subheader("Find jobs from a skill profile")
with st.form("profile_search"):
    profile_text = st.text_area(
        "Skills, interests, or target role",
        value="Python SQL machine learning data visualization communication",
        height=120,
    )
    profile_filter_columns = st.columns(3)
    profile_country = profile_filter_columns[0].selectbox(
        "Country preference", countries
    )
    profile_work_type = profile_filter_columns[1].selectbox(
        "Work type preference", work_types
    )
    profile_experience = profile_filter_columns[2].selectbox(
        "Experience range", experience_ranges
    )
    profile_submit = st.form_submit_button("Match my profile")

if profile_submit:
    if not profile_text.strip():
        st.warning("Enter at least one skill, interest, or target role.")
    else:
        try:
            results = recommender.recommend_for_profile(
                profile_text,
                top_k=top_k,
                method=method,
                country=profile_country,
                work_type=profile_work_type,
                experience=profile_experience,
            )
            if results.empty:
                st.info("No jobs match those filters. Try broader preferences.")
            else:
                show_results(results, profile_text, method, recommender, "profile")
        except (ValueError, RuntimeError) as error:
            st.error(str(error))
