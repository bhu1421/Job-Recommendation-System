from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_COLUMNS = [
    "Job Id",
    "Job Title",
    "Role",
    "Qualifications",
    "Experience",
    "Work Type",
    "Country",
    "Company",
    "Job Description",
    "skills",
    "Responsibilities",
]


def _clean_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"[\r\n\t]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _combine_features(frame: pd.DataFrame) -> pd.Series:
    fields = [
        _clean_text(frame["Job Title"]),
        _clean_text(frame["Role"]),
        _clean_text(frame["skills"]),
        _clean_text(frame["Job Description"]),
        _clean_text(frame["Responsibilities"]),
        _clean_text(frame["Qualifications"]),
        _clean_text(frame["Experience"]),
        _clean_text(frame["Work Type"]),
    ]
    combined = fields[0]
    for field in fields[1:]:
        combined = combined + " " + field
    return combined.str.lower()


def build_balanced_sample(
    csv_path: str | Path,
    max_rows_per_title: int = 400,
    chunksize: int = 100_000,
    random_state: int = 42,
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    selected_chunks: list[pd.DataFrame] = []
    title_counts: dict[str, int] = {}

    for chunk_index, chunk in enumerate(
        pd.read_csv(csv_path, usecols=MODEL_COLUMNS, chunksize=chunksize)
    ):
        chunk = chunk.dropna(subset=["Job Title", "Job Description", "skills"]).copy()
        for job_title, group in chunk.groupby("Job Title", sort=False):
            current_count = title_counts.get(job_title, 0)
            remaining = max_rows_per_title - current_count
            if remaining <= 0:
                continue
            if len(group) > remaining:
                seed = random_state + chunk_index + current_count
                group = group.sample(n=remaining, random_state=seed)
            selected_chunks.append(group)
            title_counts[job_title] = current_count + len(group)

    if not selected_chunks:
        raise ValueError("No rows were selected from the dataset.")

    sample = pd.concat(selected_chunks, ignore_index=True)
    sample = sample.drop_duplicates(subset=["Job Id"]).reset_index(drop=True)
    sample["combined_text"] = _combine_features(sample)
    sample["job_title_key"] = sample["Job Title"].str.lower().str.strip()
    return sample


@dataclass
class JobRecommendationSystem:
    jobs: pd.DataFrame
    vectorizer: TfidfVectorizer
    feature_matrix: object

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        max_rows_per_title: int = 400,
        min_df: int = 2,
        max_features: int = 8_000,
    ) -> "JobRecommendationSystem":
        jobs = build_balanced_sample(csv_path, max_rows_per_title=max_rows_per_title)
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=min_df,
            max_features=max_features,
        )
        feature_matrix = vectorizer.fit_transform(jobs["combined_text"])
        return cls(jobs=jobs, vectorizer=vectorizer, feature_matrix=feature_matrix)

    def save(self, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)

    @classmethod
    def load(cls, model_path: str | Path) -> "JobRecommendationSystem":
        return joblib.load(model_path)

    def available_job_titles(self) -> list[str]:
        return sorted(self.jobs["Job Title"].dropna().unique().tolist())

    def recommend_by_job_title(
        self,
        job_title: str,
        top_k: int = 5,
        country: str | None = None,
        work_type: str | None = None,
    ) -> pd.DataFrame:
        title_key = job_title.lower().strip()
        seed_mask = self.jobs["job_title_key"] == title_key
        seed_mask_array = seed_mask.to_numpy()
        if not seed_mask.any():
            similar_titles = get_close_matches(
                job_title,
                self.available_job_titles(),
                n=3,
                cutoff=0.5,
            )
            suggestion = (
                f" Try one of these: {', '.join(similar_titles)}."
                if similar_titles
                else ""
            )
            raise ValueError(
                f"Job title '{job_title}' was not found in the sampled dataset.{suggestion}"
            )

        anchor_vector = np.asarray(self.feature_matrix[seed_mask_array].mean(axis=0))
        scores = cosine_similarity(anchor_vector, self.feature_matrix).ravel()

        recommendations = self.jobs.copy()
        recommendations["similarity_score"] = scores
        recommendations = recommendations.loc[~seed_mask].copy()

        if country and country != "Any":
            recommendations = recommendations.loc[recommendations["Country"] == country]
        if work_type and work_type != "Any":
            recommendations = recommendations.loc[recommendations["Work Type"] == work_type]

        recommendations = recommendations.sort_values(
            by="similarity_score", ascending=False
        ).head(top_k)

        return recommendations[
            [
                "Job Title",
                "Role",
                "Company",
                "Country",
                "Work Type",
                "Qualifications",
                "Experience",
                "similarity_score",
                "skills",
                "Job Description",
            ]
        ].reset_index(drop=True)

    def recommend_for_profile(self, profile_text: str, top_k: int = 5) -> pd.DataFrame:
        cleaned_profile = " ".join(profile_text.split()).strip().lower()
        if not cleaned_profile:
            raise ValueError("Profile text cannot be empty.")

        profile_vector = self.vectorizer.transform([cleaned_profile])
        scores = cosine_similarity(profile_vector, self.feature_matrix).ravel()

        recommendations = self.jobs.copy()
        recommendations["similarity_score"] = scores
        recommendations = recommendations.sort_values(
            by="similarity_score", ascending=False
        ).head(top_k)

        return recommendations[
            [
                "Job Title",
                "Role",
                "Company",
                "Country",
                "Work Type",
                "Qualifications",
                "Experience",
                "similarity_score",
                "skills",
                "Job Description",
            ]
        ].reset_index(drop=True)
