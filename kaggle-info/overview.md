# Overview

## Intro to NLP Project — University of Waterloo

This competition is part of the University of Waterloo course **CS 498: Introduction to Natural Language Processing**.

**Objective:**
Develop NLP models that solve the task defined in the provided dataset. Your goal is to build and submit models that maximize performance according to the evaluation metric on the competition leaderboard.

Participants will train, validate, and submit model predictions on the provided NLP data. Submissions will be scored based on the evaluation metric specified in the Data and Evaluation section.

This competition emphasizes practical understanding of NLP techniques, reproducible model development, and clear documentation of methods.

**Who should enter:**
Undergraduate and graduate students enrolled in CS 498.

**What you will do:**

- Explore and preprocess the provided text data
- Build and train NLP models
- Submit predictions to Kaggle (leaderboard scoring)
- Submit your code on LEARN for reproducibility and verification
- Document methodology and results for grading

**Important (grading + code submission):**
Kaggle is used for leaderboard scoring. Course staff will use the **private leaderboard** for grading-related evaluation.

You must also submit your code separately on **LEARN** (as described in the course project repository) for reproducibility and verification.

Results will be used for course evaluation and learning.

Good luck.

## Description

### Competition Overview

This competition is the course project for **CS 498: Introduction to Natural Language Processing** at the University of Waterloo.

Participants are required to design, implement, and evaluate NLP models for the task defined by the provided dataset. The focus is on applying core NLP concepts covered in the course, including text preprocessing, representation learning, modeling, and evaluation.

The competition leaderboard reflects model performance on a held-out evaluation set and is used as one component of course assessment.

### Problem Setting

You are given a dataset consisting of natural language inputs and corresponding labels or targets. The exact task (e.g., classification, regression, sequence labeling) is defined in the **Data** section.

Your objective is to learn a function
[ f(x) \rightarrow y ] that generalizes well to unseen data, as measured by the competition’s evaluation metric.

### Evaluation

Submissions are evaluated automatically using a fixed metric described on the **Evaluation** page. Higher scores indicate better performance.

Only up to **two final submissions** will be considered for grading. The public leaderboard is provided for feedback during development.

### Allowed Methods and Resources

- Any NLP model architecture covered in the course is allowed.
- External data and pretrained models are permitted **only if** they are publicly available, free, and accessible to all students.
- All external resources must be clearly documented in your final report.

Failure to disclose external data or models constitutes an academic integrity violation.

### Reproducibility and Reporting

Your final submission must be reproducible. You are expected to:

- Clearly describe preprocessing steps
- Specify model architecture and training procedure
- Report hyperparameters and evaluation details

Course grading will consider both leaderboard performance and methodological clarity.

### Academic Integrity

This is an individual or team-based academic assignment. Collaboration beyond approved team structures, code sharing, or plagiarism is prohibited and subject to University of Waterloo academic integrity policies.

## Evaluation

Submissions are evaluated using a fixed metric on a held-out test set. The evaluation metric is defined on this page and is applied uniformly to all submissions.

Leaderboard scores reflect model performance on the evaluation data. Higher scores indicate better performance. During the competition, the public leaderboard provides feedback only; final grading may rely on a separate evaluation split.

Only the **two selected final submissions** per team will be considered for official evaluation.

### Submission Format

For each example in the test set, you must provide a prediction for the target variable.

Your submission must be a CSV file with:

- A header row
- One row per test example
- Predictions in the correct order and format

The file must follow this structure exactly:

```
ID,TARGET
2,0.137
5,0.842
6,0.421
...
```

- `ID`: the unique identifier for each test example
- `TARGET`: your model’s prediction (format depends on the task, e.g., probability, class label, or numeric value)

Submissions that do not match the required format will be rejected.