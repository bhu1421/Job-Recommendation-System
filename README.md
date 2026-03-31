# Job Recommendation System

This mini project implements a content-based job recommendation system using the dataset in `job_descriptions.csv` and the project flow described in the uploaded brief. The recommender combines `Job Title`, `Role`, `skills`, `Job Description`, `Responsibilities`, `Qualifications`, `Experience`, and `Work Type`, converts the text into TF-IDF vectors, and finds similar jobs with cosine similarity.

## Dataset note

The original dataset file is not committed to GitHub because it is too large for GitHub's file-size limits. To run the project, place `job_descriptions.csv` in the project root locally before building artifacts or starting the app.

## Why this approach fits the brief

- The dataset does not include user ratings or click history, so collaborative filtering is not a good fit.
- The brief explicitly asks for content-based filtering based on job title, skills, and description.
- TF-IDF + cosine similarity works well for cold-start recommendations when we only have job content.

## Dataset analysis used in the project

The repository dataset is much larger than the brief's example size.

- Total rows: 1,615,940
- Unique job titles: 147
- Average job description length: 185.37 characters
- Average skills length: 139.01 characters
- Core columns used for recommendation: `Job Title`, `Role`, `skills`, `Job Description`, `Responsibilities`
- Useful metadata filters: `Qualifications`, `Experience`, `Work Type`, `Country`

The analysis script also generates:

- `outputs/dataset_summary.md`
- `outputs/top_job_titles.png`
- `outputs/top_work_types.png`

## How the recommender is made practical

The full CSV is very large, so the project uses a balanced sample builder that scans the whole dataset in chunks and keeps up to `400` rows per job title. This preserves title diversity while keeping the TF-IDF model small enough to run comfortably on a laptop.

## Project files

- `src/job_recommender/analysis.py`: chunked EDA and chart generation
- `src/job_recommender/recommender.py`: balanced sampling, TF-IDF vectorization, cosine similarity recommendations
- `build_artifacts.py`: builds analysis outputs and saves the trained recommender
- `app.py`: Streamlit demo app for title-based and profile-based recommendations

## Run the project

Install dependencies:

```bash
pip install -r requirements.txt
```

Build artifacts:

```bash
python build_artifacts.py
```

Run the app:

```bash
streamlit run app.py
```

## Example project explanation for viva or report

### Problem statement

Job seekers face information overload on job portals. Keyword search often returns many irrelevant listings, so a smarter recommendation approach is needed.

### Chosen technique

We use content-based filtering. Each job is represented by its title, skills, and description-related text. TF-IDF converts the text into vectors, and cosine similarity measures how similar two jobs are.

### Input and output

- Input: a selected job title or a user skill profile
- Processing: text cleaning, feature combination, TF-IDF vectorization, cosine similarity
- Output: top similar job recommendations

### Limitations

- No personalization from past user behavior
- Depends on text quality in the dataset
- Recommendations are similarity-based rather than preference-learned

### Future enhancements

- Add resume parsing
- Learn from user clicks or saves
- Use sentence-transformer embeddings for richer semantics
