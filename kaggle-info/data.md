# Dataset Description

This page describes the files provided for the competition and the structure of each file. Review this section carefully before building your model or preparing a submission.

## Files

- `train.csv`
The training dataset. Each row contains an input sentence and the corresponding ground-truth next character.

- `test.csv`
The test dataset. Each row contains an input sentence without the ground-truth next character. You must generate predictions for this file and submit them for evaluation.

- `sample_submission.csv`
A sample submission file illustrating the required submission format. Use this as a template when creating your own submission.

- `metaData.csv`
Supplemental information about the dataset.

## Columns

### train.csv

- `id`
A unique identifier for each example.

- `context`
A natural language sentence serving as the input.

- `prediction`
The **single ground-truth next character** that immediately follows the `context`.

### test.csv

- `id`
A unique identifier for each example.

- `context`
A natural language sentence serving as the input.

### Submission file

- `id` A unique identifier matching the test input.

- `prediction` A **3-character string** containing the model’s top 3 next-character predictions.

All predicted guesses must be **single characters**, and `prediction` must contain exactly **3 characters**.

## What You Are Predicting

For each input sentence (`context`), the dataset provides exactly **one true next character**. Your task is to predict the **top three most likely next characters** in your submission, ordered from most to least likely.

A prediction is considered correct if the true next character appears in any of the three submitted predictions.