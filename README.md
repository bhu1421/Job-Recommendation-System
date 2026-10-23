# Job Recommendation System

A content-based job recommender built with Python and Streamlit. It recommends listings from a selected job title or a free-text skills profile, with keyword (TF-IDF), semantic embedding, and hybrid ranking methods available.

## Live demo

[Open the deployed app](https://job-recommendation-system-1-jljt.onrender.com/)

## Features

- Find listings similar to a selected job title or skills profile.
- Filter by country, work type, and the dataset's experience label using dropdowns.
- Choose how many results to show; recommendations appear together in a sortable table.
- Compare keyword (TF-IDF), semantic, and hybrid ranking.
- View shared keyword clues and download recommendation results.
- Explore dataset metrics and charts, including count labels on chart bars.

Recommendation scores rank listings within the selected method. They are not probabilities or a guarantee that a job is suitable.

## Dataset

The app obtains `job_descriptions.csv` from the public [Kaggle job description dataset](https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset?select=job_descriptions.csv) on first run. The CSV is cached locally in `.dataset_cache/` (or in the directory set by `JOB_DATA_DIR`) and is not stored in this repository or Docker image. You can also place your own `job_descriptions.csv` in the project root; that local file takes precedence. If Kaggle requires authentication, set `KAGGLE_API_TOKEN` as a deployment secret. Check the dataset's license and terms before redistribution.

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

Open the local URL printed in the terminal, usually <http://localhost:8501>. If the CSV is absent, the app downloads it from Kaggle. The initial build reads the large CSV in chunks and creates semantic embeddings, so it may take several minutes and downloads the pretrained embedding model. The app also creates a model with all three ranking methods automatically when a compatible model is missing. Stop it with **Ctrl+C** in the terminal.

## Run with Docker

Install and start Docker Desktop, then run this command from the project folder:

```powershell
docker compose up --build
```


This builds the image and starts the app in a container named `job-recommendation-system`. The model is included in the image; the dataset downloads from Kaggle on first startup and is kept in the persistent `job_dataset_cache` Docker volume. Charts are stored in the local `outputs` folder. Open <http://localhost:8501>. Confirm the container is running with `docker ps`. Stop it with **Ctrl+C**, then remove the container with:

```powershell
docker compose down
```

The three recommendation methods are always available. The first semantic model build downloads `sentence-transformers/all-MiniLM-L6-v2`; Docker installs CPU-only PyTorch and Sentence Transformers by default. The app loads the checked-in model without rebuilding it just because the downloaded dataset has a different file timestamp. If Kaggle requires authentication, provide `KAGGLE_API_TOKEN` to the container through an environment variable or an untracked `.env` file; never commit the token.

## How recommendations work

1. The CSV is read in chunks. Reservoir sampling selects up to 400 listings per job title across the file.
2. The model combines job title, role, skills, description, responsibilities, qualifications, experience, and work type into text for each listing.
3. TF-IDF represents words and two-word phrases as weighted features. Cosine similarity ranks listings by their text similarity to the selected title or profile.
4. Semantic mode embeds listings and the query, then ranks them by embedding similarity.
5. Hybrid mode combines the keyword and semantic rankings with reciprocal-rank fusion.

The sample cap controls model size, so recommendations come only from the sampled listings. The model stores embeddings for those sampled listings so semantic and hybrid methods are ready alongside TF-IDF.

## Project structure

See [ARCHITECTURE.md](ARCHITECTURE.md) for the data flow and a short explanation suitable for a project presentation.

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface, dataset highlights, filters, and result tables |
| `src/job_recommender/analysis.py` | Chunked dataset statistics and charts |
| `src/job_recommender/recommender.py` | Sampling, TF-IDF ranking, and model artifacts |
| `src/job_recommender/semantic.py` | Sentence embedding support |
| `src/job_recommender/explanations.py` | Shared keyword clues for recommendations |
| `build_artifacts.py` | Build the dataset analysis and recommendation model |
| `Dockerfile`, `compose.yaml` | Build and run the app with Docker |

## Limitations

- Recommendations are not personalized from clicks, applications, or saved jobs.
- TF-IDF favors shared wording; embeddings may match related wording when the dataset contains useful context.
- Results are limited to sampled listings and the dataset's available fields.
- Similarity scores are ranking signals, not probabilities or measures of hiring suitability.
