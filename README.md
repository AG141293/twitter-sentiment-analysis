# 🐦 Twitter Sentiment Analysis — NLP with NLTK, SpaCy & Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org)
[![NLTK](https://img.shields.io/badge/NLTK-3.8-green)](https://www.nltk.org/)
[![SpaCy](https://img.shields.io/badge/SpaCy-3.x-09a3d5)](https://spacy.io/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/173qY6wT-momWIkfiTdqLk2sE86bR4s2Z?usp=sharing)

> Classify tweets as **Positive 😊 | Negative 😠 | Neutral 😐** using a full NLP pipeline built with NLTK, SpaCy, Scikit-learn, and PyTorch.

---

## 📌 Project Overview

This project builds a complete **end-to-end Twitter Sentiment Analysis** system trained on the **Sentiment140** dataset (1.6M tweets). It demonstrates four progressively powerful approaches:

| Model | Library | Accuracy (approx.) |
|-------|---------|-------------------|
| VADER Rule-Based | NLTK | ~65–70% |
| TF-IDF + Naive Bayes | NLTK + Scikit-learn | ~75–79% |
| TF-IDF + Logistic Regression | NLTK + Scikit-learn | ~78–82% |
| **BiLSTM (Deep Learning)** | **PyTorch** | **~83–86%** |

---

## 🗂️ Project Structure

```
twitter-sentiment-analysis/
│
├── Twitter_Sentiment_Analysis.ipynb   ← Main Colab notebook (all code here)
├── README.md                          ← This file
├── requirements.txt                   ← Dependencies
└── outputs/                           ← Saved figures (auto-generated)
    ├── eda_distribution.png
    ├── wordclouds.png
    ├── cm_lr.png
    ├── cm_bilstm.png
    ├── bilstm_training_curves.png
    └── model_comparison.png
```

---

## 🚀 Getting Started

### Option 1 — Run in Google Colab (Recommended)

Click the **Open in Colab** badge above, then:
1. Go to `Runtime → Change runtime type → T4 GPU`
2. Run all cells top to bottom (`Runtime → Run all`)
3. No dataset download needed — the notebook fetches Sentiment140 automatically!

### Option 2 — Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Launch Jupyter
jupyter notebook Twitter_Sentiment_Analysis.ipynb
```

---

## 🧰 Libraries Used

| Library | Purpose |
|---------|---------|
| **NLTK** | TweetTokenizer, stopwords, WordNet lemmatizer, VADER sentiment |
| **SpaCy** | Fast lemmatization using `en_core_web_sm` model |
| **Scikit-learn** | TF-IDF vectorizer, Logistic Regression, Naive Bayes, metrics |
| **PyTorch** | BiLSTM deep learning model |
| **Pandas / NumPy** | Data manipulation |
| **Matplotlib / Seaborn** | Visualizations and confusion matrices |
| **WordCloud** | Word cloud generation for EDA |

---

## 🧹 Text Preprocessing Pipeline

```
Raw Tweet
   │
   ▼
Stage 1 (Regex) ── remove URLs, @mentions, expand contractions, lowercase
   │
   ▼
Stage 2 (NLTK)  ── TweetTokenizer → remove stopwords (keep negations!)
   │
   ▼
Stage 3 (SpaCy) ── Lemmatize tokens using en_core_web_sm
   │
   ▼
Clean Text → ready for modelling
```

---

## 📊 Key Results

- **Best model:** BiLSTM with ~84% accuracy and ~0.83 Macro-F1
- **Key insight:** Keeping negation words ("not", "no") in the vocabulary during stopword removal improves accuracy by ~3%
- **VADER works well** as a zero-shot baseline but struggles with sarcasm and neutral tweets

---

## 📁 Dataset

**Sentiment140** by Go et al. (Stanford, 2009)  
- 1.6 million tweets, automatically labelled by emoticons  
- Downloaded automatically in the notebook (no manual upload needed)  
- URL: https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip

A **neutral class** is derived using VADER's compound score (|score| < 0.05) to create a balanced 3-class dataset of 60,000 tweets (20k per class).

---

## 🤝 Acknowledgements

- **Codec Technologies** — internship project
- Sentiment140 dataset by Go, Bhayani & Huang (Stanford University)
- VADER: Hutto & Gilbert, ICWSM 2014

---

## 👤 Author

**[Ankita Ghosh]**  
[LinkedIn Profile](www.linkedin.com/in/ank1412) | [GitHub](https://github.com/AG141293)
