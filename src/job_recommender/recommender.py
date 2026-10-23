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
    # Reservoir sampling keeps a uniform sample across the whole CSV instead
    # of silently favoring rows that happen to occur near the beginning.
    reservoirs: dict[str, list[tuple]] = {}
    title_counts: dict[str, int] = {}
    rng = np.random.default_rng(random_state)
    columns = MODEL_COLUMNS

    for chunk in pd.read_csv(csv_path, usecols=columns, chunksize=chunksize):
        chunk = chunk.dropna(subset=["Job Title", "Job Description", "skills"]).copy()
        # pandas preserves CSV column order for usecols; align rows with the
        # MODEL_COLUMNS order before converting each row to a tuple.
        chunk = chunk[columns]
        for job_title, group in chunk.groupby("Job Title", sort=False):
            reservoir = reservoirs.setdefault(job_title, [])
            seen = title_counts.get(job_title, 0)
            for row in group.itertuples(index=False, name=None):
                seen += 1
                if len(reservoir) < max_rows_per_title:
                    reservoir.append(row)
                else:
                    replacement = int(rng.integers(0, seen))
                    if replacement < max_rows_per_title:
                        reservoir[replacement] = row
            title_counts[job_title] = seen

    if not reservoirs:
        raise ValueError("No rows were selected from the dataset.")

    selected_rows = [row for reservoir in reservoirs.values() for row in reservoir]
    sample = pd.DataFrame.from_records(selected_rows, columns=columns)
    sample = sample.drop_duplicates(subset=["Job Id"]).reset_index(drop=True)
    sample["combined_text"] = _combine_features(sample)
    sample["job_title_key"] = sample["Job Title"].str.lower().str.strip()
    return sample


@dataclass
class JobRecommendationSystem:
    jobs: pd.DataFrame
    vectorizer: TfidfVectorizer
    feature_matrix: object
    source_signature: tuple[int, int] | None = None
    semantic_embeddings: np.ndarray | None = None
    semantic_model_name: str | None = None
    artifact_version: int = 3

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        max_rows_per_title: int = 400,
        min_df: int = 2,
        max_features: int = 8_000,
        semantic_model_name: str | None = None,
    ) -> "JobRecommendationSystem":
        jobs = build_balanced_sample(csv_path, max_rows_per_title=max_rows_per_title)
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=min_df,
            max_features=max_features,
        )
        feature_matrix = vectorizer.fit_transform(jobs["combined_text"])
        stat = Path(csv_path).stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        semantic_embeddings = None
        if semantic_model_name:
            from .semantic import encode_texts

            semantic_embeddings = encode_texts(
                jobs["combined_text"].tolist(),
                semantic_model_name,
                show_progress_bar=True,
            )

        return cls(
            jobs=jobs,
            vectorizer=vectorizer,
            feature_matrix=feature_matrix,
            source_signature=signature,
            semantic_embeddings=semantic_embeddings,
            semantic_model_name=semantic_model_name,
        )

    def save(self, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)

    @classmethod
    def load(cls, model_path: str | Path) -> "JobRecommendationSystem":
        return joblib.load(model_path)

    def available_job_titles(self) -> list[str]:
        return sorted(self.jobs["Job Title"].dropna().unique().tolist())

    @property
    def semantic_available(self) -> bool:
        return self.semantic_embeddings is not None and self.semantic_model_name is not None

    def _score(self, text: str, method: str) -> np.ndarray:
        """Score all sampled listings with TF-IDF, semantic, or rank fusion."""
        tfidf_query = self.vectorizer.transform([text])
        tfidf_scores = cosine_similarity(tfidf_query, self.feature_matrix).ravel()
        if method == "tfidf":
            return tfidf_scores

        if not self.semantic_available:
            raise ValueError(
                "Semantic embeddings are not in this model. Rebuild it with python build_artifacts.py."
            )

        from .semantic import encode_texts

        semantic_query = encode_texts([text], self.semantic_model_name)[0]
        semantic_scores = self.semantic_embeddings @ semantic_query
        return self._fuse_scores(tfidf_scores, semantic_scores, method)

    @staticmethod
    def _fuse_scores(
        tfidf_scores: np.ndarray, semantic_scores: np.ndarray, method: str
    ) -> np.ndarray:
        if method == "semantic":
            return semantic_scores
        if method != "hybrid":
            raise ValueError("method must be 'tfidf', 'semantic', or 'hybrid'")

        # Reciprocal-rank fusion avoids adding scores from different scales.
        tfidf_order = np.argsort(-tfidf_scores)
        semantic_order = np.argsort(-semantic_scores)
        tfidf_rank = np.empty_like(tfidf_order)
        semantic_rank = np.empty_like(semantic_order)
        tfidf_rank[tfidf_order] = np.arange(1, len(tfidf_order) + 1)
        semantic_rank[semantic_order] = np.arange(1, len(semantic_order) + 1)
        return 1 / (60 + tfidf_rank) + 1 / (60 + semantic_rank)

    def _result_frame(
        self,
        scores: np.ndarray,
        top_k: int,
        exclude_mask: pd.Series | None = None,
        country: str | None = None,
        work_type: str | None = None,
        experience: str | None = None,
    ) -> pd.DataFrame:
        results = self.jobs.copy()
        results["similarity_score"] = scores
        if exclude_mask is not None:
            results = results.loc[~exclude_mask].copy()
        if country and country != "Any":
            results = results.loc[results["Country"] == country]
        if work_type and work_type != "Any":
            results = results.loc[results["Work Type"] == work_type]
        if experience and experience != "Any":
            results = results.loc[results["Experience"] == experience]

        columns = [
            "Job Id", "Job Title", "Role", "Company", "Country", "Work Type",
            "Qualifications", "Experience", "similarity_score", "skills",
            "Job Description", "Responsibilities", "combined_text",
        ]
        return results.sort_values("similarity_score", ascending=False).head(top_k)[
            columns
        ].reset_index(drop=True)

    def recommend_by_job_title(
        self,
        job_title: str,
        top_k: int = 5,
        country: str | None = None,
        work_type: str | None = None,
        experience: str | None = None,
        method: str = "tfidf",
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

        tfidf_anchor = self.feature_matrix[seed_mask_array].mean(axis=0)
        tfidf_scores = cosine_similarity(
            np.asarray(tfidf_anchor), self.feature_matrix
        ).ravel()
        if method == "tfidf":
            scores = tfidf_scores
        else:
            if not self.semantic_available:
                raise ValueError(
                    "Semantic embeddings are not in this model. Rebuild it with python build_artifacts.py."
                )
            semantic_anchor = self.semantic_embeddings[seed_mask_array].mean(axis=0)
            semantic_anchor /= np.linalg.norm(semantic_anchor)
            semantic_scores = self.semantic_embeddings @ semantic_anchor
            scores = self._fuse_scores(tfidf_scores, semantic_scores, method)
        return self._result_frame(
            scores, top_k, seed_mask, country, work_type, experience
        )

    def recommend_for_profile(
        self,
        profile_text: str,
        top_k: int = 5,
        method: str = "tfidf",
        country: str | None = None,
        work_type: str | None = None,
        experience: str | None = None,
    ) -> pd.DataFrame:
        cleaned_profile = " ".join(profile_text.split()).strip().lower()
        if not cleaned_profile:
            raise ValueError("Profile text cannot be empty.")

        return self._result_frame(
            self._score(cleaned_profile, method),
            top_k,
            country=country,
            work_type=work_type,
            experience=experience,
        )
