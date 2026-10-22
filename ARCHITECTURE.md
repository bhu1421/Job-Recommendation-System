# Project architecture

## Data flow

```text
job_descriptions.csv
        |
        v
chunked reader -> per-title reservoir sample -> combined job text
                                           |                    |
                                           v                    v
                                       TF-IDF             optional embeddings
                                           \                    /
                                            ranked listings
                                                  |
                                                  v
                                  Streamlit results + match clues
                                                  |
                                                  v
                              downloaded human-review CSV -> Precision@k
```

## Code map

- `analysis.py` reads the large CSV in chunks and creates dataset statistics and charts.
- `recommender.py` samples jobs, builds the TF-IDF matrix, stores artifacts, and ranks results.
- `semantic.py` optionally turns job text and user queries into dense vectors. It loads Sentence Transformers only when semantic matching is requested.
- `explanations.py` finds shared words and phrases to provide lexical clues for a result. For semantic mode, these are clues, not a full explanation of the embedding model.
- `evaluation.py` creates review sheets and calculates Precision@k from human relevance labels.
- `app.py` connects the recommender to the Streamlit controls, applies country/work-type/experience preferences, and displays results.
- `build_artifacts.py` rebuilds the model and dataset charts.
- `evaluate_recommendations.py` summarizes a reviewed CSV from the app.

## Explain the three ranking methods

- **TF-IDF:** represent words and two-word phrases as weighted features; return listings with the most similar vectors. This is useful when exact skills matter.
- **Semantic:** represent each listing and query as a dense embedding; rank by cosine similarity. Similar wording can match even when exact words differ.
- **Hybrid:** rank independently with TF-IDF and semantic similarity, then combine the two ranks with reciprocal-rank fusion. This avoids treating their raw scores as if they shared a scale.

Scores are for sorting within a method; they are not probabilities or a guarantee that a job is suitable. The human-review workflow is how this project checks relevance.

## Short presentation explanation

> “The dataset contains job text but no user ratings, so I use content-based retrieval. I sample listings across the full CSV to keep the model manageable. The default TF-IDF method rewards shared skill terms; an optional embedding method can match related wording, and hybrid mode combines their rankings. The app shows the results and keyword clues. I evaluate recommendation relevance by having people label ranked results and calculating Precision@5.”

Only claim evaluation results after completing the human review and recording the method, queries, criteria, and dataset version in the evaluation table in `README.md`.
