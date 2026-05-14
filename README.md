# Amazon Reviews NLP Intelligence App

Streamlit app for Amazon review sentiment analysis, weak fake-review detection, and weak issue classification using NLTK preprocessing, TF-IDF features, and classic machine-learning models.

Dataset source: [Amazon Reviews for Sentiment Analysis on Kaggle](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews). The source files are in fastText format, where `__label__1` is negative sentiment from 1-2 star reviews and `__label__2` is positive sentiment from 4-5 star reviews.

## What It Includes

- Sentiment model comparison: Logistic Regression, Multinomial Naive Bayes, Random Forest, and Linear SVM.
- Real-time review predictions with sentiment, issue category, and fake-review risk.
- Interactive dashboards for sentiment mix, issue trends, fake-risk distribution, model metrics, confusion matrices, and top predictive terms.
- Modular project layout under `src/`.
- Safe sampling controls for the large Kaggle files.

## Important Label Note

The Kaggle dataset has sentiment labels only. It does not include verified fake-review labels or issue-category labels. This app handles those two tasks with transparent weak supervision:

- Issue classification starts from a keyword taxonomy for delivery, quality, packaging, returns, price, usability, authenticity, and related themes, then can train a TF-IDF classifier from those weak labels when enough examples exist.
- Fake-review detection uses behavioral text signals such as repetition, excessive punctuation, marketing language, URLs, extreme generic praise, and very short reviews. If enabled by the data, a weak classifier is trained from those heuristic labels.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Add The Kaggle Data

Option 1: Download through Kaggle manually and place these files in `data/raw/`:

- `train.ft.txt.bz2`
- `test.ft.txt.bz2`

Option 2: Use the helper script after configuring Kaggle API credentials:

```powershell
python scripts/download_data.py --target data/raw
```

The app also accepts a local file path or upload for `.bz2`, `.txt`, `.zip`, and `.csv` files.

## Run

```powershell
streamlit run app.py
```

Open the local URL Streamlit prints, usually `http://localhost:8501`.

## Project Structure

```text
app.py
src/
  config.py
  data_loader.py
  evaluation.py
  fake_detection.py
  modeling.py
  preprocessing.py
  weak_labeling.py
scripts/
  download_data.py
data/
  raw/
```
