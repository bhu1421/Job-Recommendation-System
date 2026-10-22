# Job Recommendation System

This project recommends job listings from a 1.6M-row dataset using a sampled, content-based retrieval pipeline. It supports keyword matching with TF-IDF and optional semantic matching with Sentence Transformers. The Streamlit app supports a selected job title or a typed skill profile and shows matching keywords to make results easier to inspect.

## Dataset note

The original dataset file is not committed to GitHub because it is too large for GitHub's file-size limits. To run the project, place `job_descriptions.csv` in the project root locally before building artifacts or starting the app.

## Recommendation methods

- **TF-IDF** is the default baseline. It favors shared keywords and phrases, which can be useful for exact skills.
- **Semantic** uses dense sentence embeddings to match related text with different wording.
- **Hybrid** combines the two ranked lists with reciprocal-rank fusion. This avoids directly adding scores from methods with different scales.
- There is no user interaction history in the dataset, so collaborative filtering is not used.

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

The full CSV is very large, so the project scans it in chunks and uses reservoir sampling to keep up to `400` rows per job title. Selected rows are then used to fit TF-IDF. Semantic embeddings are optional and are computed for the same sample. The per-title cap limits memory and build time, but means results only come from the sample.

The app checks whether its saved model matches the current dataset file (size and modification time) and rebuilds it when needed. Changing the dataset can trigger a full scan and model build on the next app run. Semantic query embeddings are computed at search time; the first model load may download the pretrained encoder.

## Project files

See [ARCHITECTURE.md](ARCHITECTURE.md) for the data flow, module responsibilities, and a short presentation explanation.

- `src/job_recommender/analysis.py`: chunked dataset analysis and charts
- `src/job_recommender/recommender.py`: sampling, model artifacts, and ranked recommendations
- `src/job_recommender/semantic.py`: optional sentence embedding encoder
- `src/job_recommender/explanations.py`: lexical clues for each recommendation
- `src/job_recommender/evaluation.py`: review-sheet creation and Precision@k calculations
- `build_artifacts.py`: builds analysis outputs and model artifacts
- `evaluate_recommendations.py`: summarizes reviewed recommendations
- `app.py`: Streamlit user interface

## Run the project

Use Python 3.11 or newer. Install baseline dependencies from the project root:

```bash
pip install -r requirements.txt
```

Build the baseline model and analysis outputs:

```bash
python build_artifacts.py
```

Run the app:

```bash
streamlit run app.py
```

### Enable semantic and hybrid methods

Sentence Transformers and its model files add installation, download, build-time, and storage costs. Install the optional dependency and build an artifact containing embeddings:

```bash
pip install -r requirements-semantic.txt
python build_artifacts.py --semantic
streamlit run app.py
```

The app offers semantic and hybrid methods when the saved artifact includes embeddings. To return to the baseline-only model, run `python build_artifacts.py` without `--semantic`.

## Evaluate recommendation relevance

The app can download a review CSV for each search. Run several searches with the same recommendation method, download each review sheet, and combine the rows into one CSV. Fill `relevant` with `1` when a recommendation is relevant to the query and `0` otherwise. Use the same criteria for all methods and queries. When comparing methods, review each method's results independently.

Calculate Precision@5:

```bash
python evaluate_recommendations.py reviewed_results.csv --k 5
```

The script reports per-query scores and mean Precision@5. The score is the number of relevant jobs in the first five results divided by five. Do not report an evaluation result until the review column has been completed. Record the review date, number and type of queries, relevance criteria, dataset version, and recommendation method in `REPORT.md`.

## Example project explanation for viva or report

### Problem statement

Job seekers face information overload on job portals. Keyword search often returns many irrelevant listings, so a smarter recommendation approach is needed.

### Chosen technique

The baseline uses TF-IDF and cosine similarity. Optional semantic matching encodes text into dense vectors; hybrid ranking combines ranked results with reciprocal-rank fusion.

### Input and output

- Input: a selected job title or a user skill profile
- Processing: text cleaning, balanced sampling, vectorization or embeddings, similarity ranking
- Output: top similar job recommendations

### Limitations

- No personalization from past user behavior
- Depends on text quality in the dataset
- Recommendations are similarity-based rather than preference-learned

### Evaluation status

Recommendation quality must be measured with human judgments. Use the review CSV workflow above and add the observed results here; do not treat similarity scores as accuracy or probability.
