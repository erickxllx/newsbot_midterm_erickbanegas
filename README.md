# BBC News NLP Classifier & Sentiment Analyzer

## Project Overview
This project implements a complete system for news classification and sentiment analysis using the BBC News dataset.  
The goal was to develop an end-to-end pipeline that integrates Natural Language Processing (NLP) with machine learning models to analyze news articles, identify their main category, and extract relevant information such as named entities, grammatical relations, and emotional tone.

The project combines text representation techniques based on **TF-IDF** with classical classification models (Logistic Regression, Linear SVM, and Naive Bayes), along with linguistic analysis using **spaCy** and **NLTK** to produce interpretable metrics and insights.

---

## Repository Structure

```
project-root/
│
├── data/
│   └── newsbot_dataset.csv              ← Clean dataset generated in Notebook #1
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb           ← Data cleaning and preparation
│   ├── 02_training_model.ipynb          ← Model training and evaluation
│   └── 03_sentiment_ner_pos.ipynb       ← Semantic, syntactic, and sentiment analysis
│
├── outputs/
│   ├── models/
│   │   └── newsbot_best_LinearSVC.pkl   ← Trained model with best performance
│   │
│   ├── reports/
│   │   ├── metrics.txt                  ← Classification metrics and accuracy
│   │   └── test_predictions.csv         ← Test set predictions
│   │
│   └── analysis/
│       ├── sentiment_by_doc.csv         ← Sentiment score for each document
│       ├── sentiment_distribution.csv   ← Sentiment distribution per category
│       ├── pos_rates_by_category.csv    ← POS tag frequency by category
│       ├── dep_pairs_by_category.csv    ← Dependency relations
│       ├── entities_raw.csv             ← Extracted named entities
│       ├── entities_by_category.csv     ← Entities grouped by category
│       └── summary.txt                  ← Summary of main results
│
└── README.md
```

---

## Notebooks Overview

### 1. 01_data_cleaning.ipynb
In this notebook I loaded the raw Kaggle dataset (`BBCNews.csv`) and performed text cleaning, label normalization, and removal of missing or duplicate records.  
Finally, I saved the processed dataset as `newsbot_dataset.csv` in the `data/` folder.

### 2. 02_training_model.ipynb
In this step I transformed the cleaned text using **TF-IDF** and trained three models:
- Logistic Regression  
- Linear SVM  
- Multinomial Naive Bayes  

After evaluating performance, I selected the best model (LinearSVC) and saved it in `.pkl` format along with its evaluation metrics and test predictions.

### 3. 03_sentiment_ner_pos.ipynb
In the final notebook I implemented the complete linguistic analysis using **spaCy** and **NLTK**.  
This stage included:
- Sentiment analysis (positive, neutral, negative) using Vader.  
- Part-of-speech tagging and syntactic dependency analysis.  
- Named Entity Recognition (NER) for entities like people, organizations, locations, dates, and amounts.  
All results were exported to the `outputs/analysis/` folder.

---

## Example Results

### Model Performance
| Model | Accuracy |
|--------|-----------|
| Logistic Regression | 0.948 |
| Linear SVC | 0.960 |
| MultinomialNB | 0.873 |

### Sentiment Distribution
| Sentiment | Count |
|------------|-------|
| Positive | 1463 |
| Negative | 134 |
| Neutral  | 12 |

### Example Prediction
```python
sample = "Apple announced new chips focused on AI acceleration and enterprise security."
→ Predicted category: tech
```

---

## Requirements

To run this project in Google Colab or locally, install the following dependencies:

```bash
pip install -U scikit-learn==1.4.2 nltk==3.9.1 spacy==3.8.0 joblib==1.4.2
python -m spacy download en_core_web_sm
```

---

## How to Run

1. Open the project in Google Colab.  
2. Run each notebook in the following order:
   ```
   01_data_cleaning.ipynb
   02_training_model.ipynb
   03_sentiment_ner_pos.ipynb
   ```
3. After running the final notebook, the results are saved in the `outputs/analysis/` folder.  
4. (Optional) To compress all results into a single zip file:
   ```python
   import shutil
   shutil.make_archive("newsbot_outputs", "zip", "outputs")
   ```

---

## License
This project was developed for academic purposes as part of the *Artificial Intelligence and Natural Language Processing* coursework.  
Author: **Erick Banegas**
