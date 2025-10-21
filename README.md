# Event RecSys MVP

> MVP of a hybrid event recommendation system comparing Content-Based, Collaborative Filtering (WMF), and Social approaches using Kaggle's Event Recommendation Challenge dataset. Evaluated with MAP@200.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF.svg)](https://www.kaggle.com/c/event-recommendation-engine-challenge)

## 🎯 Overview

Implementation of three recommendation approaches for event discovery, evaluated independently and combined:

- **Content-Based Filtering**: Embeddings (categorical, numerical, textual) + cosine similarity
- **Collaborative Filtering**: Weighted Matrix Factorization (WMF) with WALS
- **Social Recommendation**: Friend-based collaborative filtering

## 📊 Evaluation Metrics

Models are evaluated using:
- **Recall@K**: Coverage of user's interested events in top-K recommendations
- **Hit Rate@K**: Percentage of users with at least one correct recommendation
- **Contamination@K**: Percentage of recommendations that user marked as "not interested"

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/event-recsys-mvp
cd event-recsys-mvp

# Create conda environment
conda env create -f environment.yml
conda activate event-recsys-mvp
```

### Dataset Setup

1. Download from [Kaggle Event Recommendation Challenge](https://www.kaggle.com/c/event-recommendation-engine-challenge)
2. Place files in `data/raw/`:
   - `train.csv`
   - `events.csv`, `users.csv`
   - `user_friends.csv`, `event_attendees.csv`

### Run Experiments

```bash
jupyter notebook notebooks/experiments.ipynb
```

**Evaluation Approach:** This implementation uses a **temporal train/validation split** on `train.csv`:
- Each user's interactions are sorted by timestamp
- 50% oldest interactions → training set
- 50% newest interactions → validation set (with labels)
- **Note:** This simulates predicting future events for known users, which differs from the original competition objective (cold start for new users)

## 📁 Project Structure

```
event-recsys-mvp/
├── data/
│   ├── raw/                    # Kaggle dataset
│   └── processed/              # Preprocessed data (generated)
├── models/
│   ├── base.py                 # Base recommender interface
│   ├── content_based.py        # Content-based filtering
│   ├── collaborative.py        # Collaborative filtering (WMF)
│   ├── social.py               # Social-based recommendations
│   └── hybrid.py               # Hybrid ensemble model
├── utils/
│   ├── metrics.py              # Evaluation metrics (Recall, Hit Rate, Contamination)
│   ├── preprocessing.py        # Data preprocessing utilities
│   ├── temporal_split.py       # Temporal train/val split
│   └── geo_filter.py           # Geographic filtering utilities
├── notebooks/
│   └── experiments.ipynb       # Main experiments notebook
├── environment.yml
└── README.md
```

## 📦 Dataset & Mapping

### Kaggle Dataset Structure

**Files:**
- `train.csv`: user-event interactions (user, event, invited, timestamp, interested, not_interested)
- `events.csv`: 110 columns (event_id, user_id, start_time, lat, lng, city, state, country, c_1...c_100, c_other)
- `users.csv`: demographics (user_id, locale, birthyear, gender, joinedAt, location, timezone)
- `user_friends.csv`: social graph (user, friends)
- `event_attendees.csv`: attendance status (event, yes, maybe, invited, no)

**Statistics:** ~38K users, ~3M events, ~15K labeled interactions

### Mapping to Event Recommendation System

#### Interaction Types

| System Concept | Kaggle Data | Usage |
|----------------|-------------|-------|
| **Swipe Right** | `interested = 1` | Positive signal |
| **Swipe Left** | `not_interested = 1` | Negative signal |
| **Ticket Purchase** | `yes` in event_attendees | Strongest positive signal |

#### Event Features (Available)

| Feature | Source | Transformation |
|---------|--------|----------------|
| **Tags** | c_1...c_100 | K-means clustering (k~30) → event categories |
| **Horário** | start_time | Extract hour (0-23) |
| **Dia da Semana** | start_time | Extract weekday (0-6) |
| **Localização** | lat, lng, city | GPS coordinates + city name |

**Note:** Dataset does not include explicit event titles, descriptions, styles, or price information. Tags are synthetically created via clustering of word frequency features.

#### WMF Weights (Collaborative Filtering)

Matrix W (confidence weights):
- Ticket purchase (yes): **100.0**
- Swipe right (interested): **10.0**
- Swipe left (not_interested): **1.0**
- Not seen (no record): **0.1**

Matrix R (observed preference):
- R = 1 for positive signals (interested=1 or purchase)
- R = 0 for negative signals (not_interested=1) or not seen

**Note:** "Not seen" means no interaction was recorded. The low weight (0.1) reflects low confidence in the absence of observation, not that it's implicitly negative.

#### User Embedding Weights (Content-Based)

Weighted average of positive interactions only:
- Ticket purchase: **3.0**
- Swipe right: **1.0**

Temporal decay: `w = base_weight × exp(-0.01 × days_since_interaction)`

**Note:** Only positive signals (interested=1 or purchase) are used to build user embeddings. Negative signals (not_interested=1) are excluded.

## 🔬 Methodology Summary

**All models** use geographic pre-filtering: candidate events limited to top-K nearest based on user's median location (Haversine distance).

### Content-Based
- Event embeddings: categorical (K-means clusters) and numerical (hour, weekday) features
- User embeddings: weighted average of positive interactions with temporal decay
- Similarity: cosine distance

### Collaborative Filtering
- Algorithm: Weighted Matrix Factorization (WMF)
- Solver: Alternating Least Squares (WALS)
- Hyperparameters: k=20 latent factors, λ=0.01 regularization

### Social
- Friend graph analysis
- Events ranked by friend engagement (interested, attending)

### Hybrid
- Weighted ensemble: `score = 0.3×CB + 0.3×CF + 0.4×Social`
- Min-max normalization per component

## 📈 Evaluation

Models are evaluated on a temporal train/validation split where:
- Training uses 50% oldest user interactions
- Validation uses 50% newest user interactions
- Only users with at least one label (interested or not_interested) in validation are evaluated

**Metrics:**
- **Recall@K**: What percentage of the user's interested events were found in top-K?
- **Hit Rate@K**: What percentage of users received at least one good recommendation?
- **Contamination@K**: What percentage of recommendations were events users marked as not interested?

## 🛠️ Tech Stack

- Python 3.8+
- numpy, pandas, scipy, scikit-learn
- implicit (ALS implementation)
- sentence-transformers (text embeddings)

## 📚 References

Allan Carroll, Avalon, Ben Hamner, events4you, glhf, and Greg Melton. Event Recommendation Engine Challenge. https://kaggle.com/competitions/event-recommendation-engine-challenge, 2013. Kaggle.

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

*MVP implementation for educational and research purposes.*