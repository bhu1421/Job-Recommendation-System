# Mini Project Report: Job Recommendation System

## 1. Problem Understanding

In online job portals, users face a very large number of job listings and keyword search often returns many irrelevant results. A recommendation system helps reduce information overload by suggesting jobs that are similar to a job role or user skill profile.

This project builds a **content-based job recommendation system** using the provided `job_descriptions.csv` dataset. The system does not depend on user ratings or click history, which makes it suitable for the cold-start condition described in the project brief. It provides a TF-IDF baseline and optional semantic and hybrid ranking modes.

## 2. Project Objectives

- Build a recommendation system that suggests similar jobs.
- Use text similarity techniques for recommendation.
- Avoid dependence on user ratings or user interaction history.
- Provide a simple interface that can be demonstrated in a mini-project presentation.

## 3. Dataset Exploration and Understanding

### 3.1 Dataset Details

- Source mentioned in brief: Kaggle
- Dataset file used: `job_descriptions.csv`
- Total rows: `1,615,940`
- Unique job titles: `147`

### 3.2 Features Used

The recommendation model uses:

- `Job Title`
- `Role`
- `skills`
- `Job Description`
- `Responsibilities`
- `Qualifications`
- `Experience`
- `Work Type`

Metadata used for filtering or presentation:

- `Country`
- `Company`

### 3.3 Observations

- The dataset is very large and contains many repeated job categories.
- Job descriptions are short but informative enough for text-based matching.
- Skills and job descriptions contain strong domain signals such as analytics, networking, software development, and marketing terminology.
- Work types are distributed almost evenly across `Full-Time`, `Part-Time`, `Contract`, `Intern`, and `Temporary`.

### 3.4 Data Issues Identified

- The full dataset is too large to vectorize directly on a normal laptop.
- Several titles appear many thousands of times, which can over-dominate the model.
- Some semantically similar roles use different titles, so exact title match input is not always available.

## 4. Approach Selection Justification

### Chosen Method: Content-Based Filtering

This method was selected because:

- The dataset does not provide user-item interaction history.
- Text columns such as title, skills, and description are rich enough for similarity-based matching.
- Content-based filtering handles cold-start better than collaborative filtering in this scenario.

### Why Collaborative Filtering Was Not Used

- It requires user ratings, clicks, views, or application history.
- Such interaction data is not present in the dataset.
- It would not perform well for new jobs or new users in this project setting.

## 5. System Design and Logic

### 5.1 Input

The system supports two inputs:

- A selected job title
- A custom user skill/profile text

### 5.2 Processing Steps

1. Read the CSV in chunks.
2. Create a balanced sample by keeping up to `400` rows per job title.
3. Merge text fields into one feature string.
4. Apply TF-IDF vectorization and, when enabled, sentence embedding encoding.
5. Rank jobs with TF-IDF cosine similarity, semantic cosine similarity, or reciprocal-rank fusion of both rankings.
6. Return the top recommended jobs and show lexical overlap terms as clues.

### 5.3 Output

- Top `5` similar job recommendations by default
- Similarity score for each recommendation
- Company, country, work type, qualification, and experience details

## 6. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Sentence Transformers (optional semantic mode)

## 7. Illustrative checks (not a measured evaluation)

The examples below describe plausible outputs to inspect during a demo. They are not measured relevance results and should be replaced or confirmed by the human-review evaluation before presenting them as findings.

### Test Case 1

- Input: `Data Analyst`
- Output examples: `Business Analyst`
- Expected behavior to review: recommendations should share analytics or business-intelligence skills and responsibilities.

### Test Case 2

- Input: `Software Engineer`
- Output examples: `Web Developer`
- Expected behavior to review: recommendations should share software-development skills or responsibilities.

### Test Case 3

- Input profile: `python sql dashboards statistics business intelligence`
- Output examples: `Data Analyst`
- Expected behavior to review: analytics-oriented listings should rank well when their skills match the profile.

## 8. Evaluation plan

The dataset has no relevance labels or user feedback, so recommendation quality has not yet been measured. The app provides a review CSV workflow: run a set of searches, download the ranked results, mark each result as relevant (`1`) or not relevant (`0`), combine the rows, and run `python evaluate_recommendations.py reviewed_results.csv --k 5`.

The script reports per-query Precision@5 and its mean. Precision@5 is the count of relevant jobs in the first five results divided by five. To compare methods, review results from the same queries independently for each method and report the mean for each method. Fill in actual review details and observed values below after completing the review; do not infer quality from similarity scores.

- Review date: pending
- Number and type of queries: pending
- Relevance criteria: pending
- Dataset version: pending
- TF-IDF mean Precision@5: pending
- Semantic mean Precision@5: pending (after enabling semantic embeddings)
- Hybrid mean Precision@5: pending (after enabling semantic embeddings)

### Performance considerations

The app scans the full CSV for analysis and builds its model from the balanced sample. Semantic encoding is optional because it adds a pretrained model download, extra build time, and embedding storage. The sample is capped to 400 listings per title, so recommendations do not cover every row.

No measured performance claim is made until the human relevance review is completed.

## 9. Limitations

- No personalization from user history
- Exact job title input may fail if that title does not exist in the sampled data
- Recommendation quality depends on text quality in the dataset
- TF-IDF is lexical and can miss related terms expressed differently; semantic embeddings can help but still require relevance evaluation
- The training sample is limited to 400 listings per title
- Human relevance review is required before making claims about recommendation quality

## 10. Conclusion

The project demonstrates a modular content-based job retrieval system with a TF-IDF baseline and optional semantic and hybrid modes. Its practical value should be assessed with human-reviewed relevance judgments; the implementation's similarity scores alone do not establish recommendation quality.
