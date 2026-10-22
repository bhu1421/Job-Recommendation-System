# Job Recommendation System

A content-based job recommender built with Python and Streamlit. It recommends listings from a selected job title or a free-text skills profile. TF-IDF keyword matching is available by default; optional sentence embeddings enable semantic and hybrid ranking.

## Features

- Find listings similar to a selected job title or skills profile.
- Filter by country, work type, and the dataset's experience label using dropdowns.
- Choose how many results to show; recommendations appear together in a sortable table.
- Compare keyword (TF-IDF), semantic, and hybrid ranking when semantic support is installed.
- View shared keyword clues and download recommendation results.
- Explore dataset metrics and charts, including count labels on chart bars.

Recommendation scores rank listings within the selected method. They are not probabilities or a guarantee that a job is suitable.

## Dataset

Download `job_descriptions.csv` from the [Google Drive dataset folder](https://drive.google.com/drive/folders/1EdrQWOTa18f2dmPmS1JfoJdZjvOgNyx5) and place it in the project root, next to `app.py`. The large dataset is not stored in this repository. Confirm that you have permission to use and redistribute your copy.

The dataset currently analyzed by this project contains **1,615,940 listings** and **147 unique job titles**. Its work types are close to evenly represented:

| Work type | Listings | Share |
|---|---:|---:|
| Part-Time | 324,044 | 20.05% |
| Temporary | 323,439 | 20.02% |
| Contract | 323,131 | 20.00% |
| Intern | 323,090 | 19.99% |
| Full-Time | 322,236 | 19.94% |

These counts describe the current dataset copy. Running the analysis with your local CSV regenerates the charts and calculates its summary.

## Run locally on Windows

Use Python 3.11 or newer. Open PowerShell in the project folder and create the virtual environment:

```powershell
cd "C:\Users\Bhuvan\Desktop\jobrecommendation\Job-Recommendation-System"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the virtual environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Build the dataset analysis and recommendation model, then start Streamlit:

```powershell
python build_artifacts.py
python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually <http://localhost:8501>. The initial build reads the large CSV in chunks and may take several minutes. The app also creates a baseline model automatically when a current artifact is missing. Stop it with **Ctrl+C** in the terminal.

## Run with Docker

Install and start Docker Desktop. Put `job_descriptions.csv` in the project root, then run:

```powershell
docker compose up --build
```

Open <http://localhost:8501>. The compose file mounts the dataset read-only and stores generated artifacts and charts in the local `artifacts` and `outputs` folders. Stop the running container with **Ctrl+C**, then remove it with:

```powershell
docker compose down
```

## Enable semantic and hybrid ranking

Semantic support is optional. It installs Sentence Transformers and downloads a pretrained embedding model. With the virtual environment active, run:

```powershell
python -m pip install -r requirements-semantic.txt
python build_artifacts.py --semantic
python -m streamlit run app.py
```

The app's recommendation method menu will then include **Semantic embeddings** and **Hybrid (rank fusion)**. To rebuild a baseline-only model, run `python build_artifacts.py` without `--semantic`.

For a Docker image with semantic dependencies, run:

```powershell
$env:INSTALL_SEMANTIC = "true"
docker compose build
docker compose run --rm recommender python build_artifacts.py --semantic
docker compose up
```

The dataset, model artifacts, and charts are mounted from the project folder. If Windows Application Control blocks installation, TF-IDF mode remains available; semantic mode requires an administrator to allow the dependencies to be installed.

## How recommendations work

1. The CSV is read in chunks. Reservoir sampling selects up to 400 listings per job title across the file.
2. The model combines job title, role, skills, description, responsibilities, qualifications, experience, and work type into text for each listing.
3. TF-IDF represents words and two-word phrases as weighted features. Cosine similarity ranks listings by their text similarity to the selected title or profile.
4. Semantic mode embeds listings and the query, then ranks them by embedding similarity.
5. Hybrid mode combines the keyword and semantic rankings with reciprocal-rank fusion.

The sample cap controls model size, so recommendations come only from the sampled listings. Semantic and hybrid methods are not available until semantic embeddings have been built.

## Project structure

See [ARCHITECTURE.md](ARCHITECTURE.md) for the data flow and a short explanation suitable for a project presentation.

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface, dataset highlights, filters, and result tables |
| `src/job_recommender/analysis.py` | Chunked dataset statistics and charts |
| `src/job_recommender/recommender.py` | Sampling, TF-IDF ranking, and model artifacts |
| `src/job_recommender/semantic.py` | Optional sentence embedding support |
| `src/job_recommender/explanations.py` | Shared keyword clues for recommendations |
| `build_artifacts.py` | Build the dataset analysis and recommendation model |
| `Dockerfile`, `compose.yaml` | Build and run the app with Docker |

## Limitations

- Recommendations are not personalized from clicks, applications, or saved jobs.
- TF-IDF favors shared wording; embeddings may match related wording when the dataset contains useful context.
- Results are limited to sampled listings and the dataset's available fields.
- Similarity scores are ranking signals, not probabilities or measures of hiring suitability.
