from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.job_recommender import JobRecommendationSystem, analyze_dataset


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "job_descriptions.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "job_recommender.joblib"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


@st.cache_resource(show_spinner=False)
def load_recommender() -> JobRecommendationSystem:
    if MODEL_PATH.exists():
        return JobRecommendationSystem.load(MODEL_PATH)

    recommender = JobRecommendationSystem.from_csv(DATA_PATH, max_rows_per_title=400)
    recommender.save(MODEL_PATH)
    return recommender


@st.cache_data(show_spinner=False)
def load_analysis() -> dict:
    return analyze_dataset(DATA_PATH, output_dir=OUTPUT_DIR)


st.set_page_config(page_title="Job Recommendation System", layout="wide")

st.title("Job Recommendation System")
st.caption(
    "Content-based filtering using TF-IDF and cosine similarity on job title, skills, and description."
)

analysis = load_analysis()
recommender = load_recommender()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Jobs", f"{analysis['row_count']:,}")
col2.metric("Unique Job Titles", analysis["unique_job_titles"])
col3.metric("Avg Description Length", f"{analysis['avg_job_description_chars']}")
col4.metric("Avg Skills Length", f"{analysis['avg_skills_chars']}")

st.subheader("Dataset Highlights")
left, right = st.columns(2)
with left:
    title_chart = OUTPUT_DIR / "top_job_titles.png"
    if title_chart.exists():
        st.image(str(title_chart), caption="Most common job titles")
    else:
        st.info("Install matplotlib to generate the job-title chart.")
with right:
    work_type_chart = OUTPUT_DIR / "top_work_types.png"
    if work_type_chart.exists():
        st.image(str(work_type_chart), caption="Work type distribution")
    else:
        st.info("Install matplotlib to generate the work-type chart.")

st.subheader("Recommend Similar Jobs by Title")

countries = ["Any"] + sorted(recommender.jobs["Country"].dropna().unique().tolist())
work_types = ["Any"] + sorted(recommender.jobs["Work Type"].dropna().unique().tolist())

selected_title = st.selectbox("Select a job title", recommender.available_job_titles())
selected_country = st.selectbox("Optional country filter", countries)
selected_work_type = st.selectbox("Optional work type filter", work_types)
top_k = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)

if st.button("Get Recommendations", type="primary"):
    title_results = recommender.recommend_by_job_title(
        selected_title,
        top_k=top_k,
        country=selected_country,
        work_type=selected_work_type,
    )
    st.dataframe(
        title_results.assign(
            similarity_score=title_results["similarity_score"].round(4)
        ),
        use_container_width=True,
    )

st.subheader("Recommend Jobs from a Skill Profile")
profile_text = st.text_area(
    "Enter skills, interests, or a target role",
    value="Python SQL machine learning data visualization communication",
    height=120,
)

if st.button("Match My Profile"):
    profile_results = recommender.recommend_for_profile(profile_text, top_k=top_k)
    st.dataframe(
        profile_results.assign(
            similarity_score=profile_results["similarity_score"].round(4)
        ),
        use_container_width=True,
    )
