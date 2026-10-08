# SHL Hiring Assessment 2026 — Spoken Grammar Scoring Engine

## Overview

This project was developed for the SHL Hiring Assessment 2026 Kaggle competition. The objective was to predict a continuous grammar score from 0 to 5 for spoken English audio recordings.

The solution combines acoustic/prosodic features, Whisper speech representations, and transcript-based text features.

## Methodology

### 1. Acoustic / Prosodic Features

Audio recordings were processed using Librosa to extract:

- Duration
- RMS energy
- Zero-crossing rate
- Spectral centroid
- Spectral bandwidth
- Spectral rolloff
- MFCC statistics
- Pitch statistics
- Voicing ratio
- Low-energy ratio

These features were modeled using CatBoost regression.

### 2. Whisper Speech Representation

Whisper was used to process the complete audio recordings.

Two types of information were used:

- Speech transcripts
- Whisper encoder embeddings

The encoder representations were mean-pooled into fixed-length vectors and modeled using Ridge regression.

### 3. Transcript Features

Whisper transcripts were converted into word-level TF-IDF features using unigrams and bigrams. Ridge regression was used for prediction.

## Evaluation

A fixed 5-fold K-Fold cross-validation strategy was used.

Out-of-fold predictions were generated for model comparison and ensemble selection.

Evaluation metrics:

- Pearson correlation — higher is better
- RMSE — lower is better

## Final Ensemble

The final ensemble combined the complementary model predictions using:

- 50% Acoustic / CatBoost
- 45% Whisper encoder / Ridge
- 5% Word TF-IDF / Ridge

The OOF evaluation of this ensemble achieved:

| Metric | Score |
|---|---:|
| Pearson | 0.8478 |
| RMSE | 0.6627 |

## Output

Final predictions were clipped to the valid 0–5 score range and written to `submission.csv`.

## Repository Contents

- `final-submission.ipynb` — complete competition notebook
- `submission.csv` — competition submission output
- `requirements.txt` — Python dependencies

The competition dataset and audio recordings are not included in this repository.

## Reproducibility

The competition dataset is provided through the Kaggle competition environment. The notebook contains the complete preprocessing, feature extraction, modeling, evaluation, ensemble, and submission pipeline.
