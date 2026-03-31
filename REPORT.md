# Mini Project Report: Job Recommendation System

## 1. Problem Understanding

In online job portals, users face a very large number of job listings and keyword search often returns many irrelevant results. A recommendation system helps reduce information overload by suggesting jobs that are similar to a job role or user skill profile.

This project builds a **content-based job recommendation system** using the provided `job_descriptions.csv` dataset. The system does not depend on user ratings or click history, which makes it suitable for the cold-start condition described in the project brief.

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
4. Apply TF-IDF vectorization.
5. Compute cosine similarity between the input representation and all job vectors.
6. Return the top recommended jobs.

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

## 7. Results and Observations

### Test Case 1

- Input: `Data Analyst`
- Output examples: `Business Analyst`
- Observation: Recommendations stay close to analytics and business intelligence roles.

### Test Case 2

- Input: `Software Engineer`
- Output examples: `Web Developer`
- Observation: Recommendations move toward related software development roles based on shared technical text.

### Test Case 3

- Input profile: `python sql dashboards statistics business intelligence`
- Output examples: `Data Analyst`
- Observation: Skill-based profile matching correctly surfaces analytics-oriented roles.

## 8. Evaluation and Performance Thinking

This system is evaluated using logical relevance rather than rating prediction accuracy because the dataset has no explicit user feedback. The recommendations are meaningful when the text fields clearly describe the job domain, required skills, and responsibilities.

The system performs well for:

- common job titles
- roles with strong skill keywords
- quick content-based exploration

## 9. Limitations

- No personalization from user history
- Exact job title input may fail if that title does not exist in the sampled data
- Recommendation quality depends on text quality in the dataset
- TF-IDF captures lexical similarity better than deep semantic similarity

## 10. Conclusion

The project successfully demonstrates a job recommendation system based on content-based filtering. By using TF-IDF and cosine similarity on job title, skills, and description-related features, the system can recommend relevant jobs without needing ratings or user interaction history. This makes it a strong and practical mini-project for demonstrating recommendation system concepts.
