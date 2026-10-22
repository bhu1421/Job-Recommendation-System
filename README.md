# Job Recommendation System

A Streamlit application that recommends job listings from job descriptions and skill profiles. The baseline uses TF-IDF and cosine similarity; optional semantic embeddings and hybrid ranking are also supported.

## Features

- Find similar listings from a selected job title.
- Match a free-text profile of skills, interests, or target roles.
- Filter title-based recommendations by country and work type.
- Compare keyword, semantic, and hybrid ranking when semantic embeddings are installed.
- See shared keywords as clues for why a listing ranked highly.
- Download ranked results for human relevance review.
- Explore dataset counts and charts.

## Dataset

Place `job_descriptions.csv` in the project root, next to `app.py`. The dataset is not included in this repository because it is large. Confirm that you have permission to use and redistribute your copy.

The current dataset contains **1,615,940 rows** and **147 unique job titles**. Work type counts are nearly balanced:

| Work type | Listings | Share |
|---|---:|---:|
| Part-Time | 324,044 | 20.05% |
| Temporary | 323,439 | 20.02% |
| Contract | 323,131 | 20.00% |
| Intern | 323,090 | 19.99% |
| Full-Time | 322,236 | 19.94% |

The current charts show counts directly on the bars. Run the analysis step below to regenerate charts and the dataset summary from your local CSV.

## How recommendations work

1. The CSV is read in chunks. Reservoir sampling keeps up to 400 listings per job title, giving rows across the file a chance to be selected.
2. Text fields are combined: job title, role, skills, description, responsibilities, qualifications, experience, and work type.
3. The selected job text is converted to TF-IDF vectors. Similarity is ranked with cosine similarity.
4. If enabled, a Sentence Transformers model also converts the job text and query into dense embeddings for semantic ranking.
5. Hybrid mode combines the keyword and semantic result ranks with reciprocal-rank fusion.

The sample cap helps control model size, but recommendations only come from sampled listings. Similarity scores are ranking signals, not probabilities or a measure of hiring suitability.

## Setup on Windows

Use Python 3.11 or newer. From PowerShell, open the project folder and create a virtual environment:

```powershell
cd "C:\Users\Bhuvan\Desktop\jobrecommendation\Job-Recommendation-System"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell says script execution is disabled, use the virtual environment's Python without activating it:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Build and run

Build the baseline recommender and dataset analysis before starting the app:

```powershell
python build_artifacts.py
python -m streamlit run app.py
```

The first build scans the large CSV and can take several minutes. Stop Streamlit with **Ctrl+C** in its terminal. The app can also build a baseline model automatically if a current model artifact is missing.

### Enable semantic and hybrid ranking

Semantic mode is optional. Its package and pretrained model files add installation, download, build-time, and storage costs. With the virtual environment active, run:

```powershell
python -m pip install -r requirements-semantic.txt
python build_artifacts.py --semantic
```

Then restart Streamlit so it loads the new artifact. The recommendation method menu will include **Semantic embeddings** and **Hybrid (rank fusion)**. To rebuild a baseline-only artifact, run `python build_artifacts.py` without `--semantic`.

If Windows Application Control blocks package installation, the app can still run in TF-IDF mode; ask your administrator to allow the installation before enabling semantic ranking.

## Evaluate recommendation relevance

The dataset has no user ratings or relevance labels, so recommendation quality needs human review.

1. Run several title and skill-profile searches in the app.
2. Download the review CSV for each search.
3. Mark `relevant` as `1` when the job fits the query and `0` when it does not.
4. Combine the reviewed rows into one CSV file.
5. Calculate Precision@5:

```powershell
python evaluate_recommendations.py reviewed_results.csv --k 5
```

Precision@5 is the number of relevant listings in the first five results divided by five. To compare methods, review results for the same queries with the same criteria. Record your query set, relevance criteria, dataset version, and actual scores below. Do not report results until the review is complete.

**Evaluation status:** Pending human relevance review.

## Project structure

See [ARCHITECTURE.md](ARCHITECTURE.md) for the data flow and a short presentation explanation.

| File | Responsibility |
|---|---|
| `app.py` | Streamlit interface and cached model loading |
| `src/job_recommender/analysis.py` | Chunked dataset statistics, report, and charts |
| `src/job_recommender/recommender.py` | Reservoir sampling, TF-IDF, ranking, and model artifacts |
| `src/job_recommender/semantic.py` | Optional sentence embedding model |
| `src/job_recommender/explanations.py` | Shared keyword clues for results |
| `src/job_recommender/evaluation.py` | Review CSV creation and Precision@k calculations |
| `build_artifacts.py` | Build dataset analysis and recommender artifacts |
| `evaluate_recommendations.py` | Calculate metrics from reviewed CSV results |

## Limitations

- No recommendations are personalized from clicks, applications, or saved jobs.
- TF-IDF favors shared wording; embeddings can help with related wording but still need evaluation.
- The model only searches the sampled listings.
- Similarity is not a probability or proof that a job is suitable.
- Human-reviewed relevance results have not been collected yet.
