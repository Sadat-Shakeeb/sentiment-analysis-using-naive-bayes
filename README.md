# sentiment-analysis-using-naive-bayes
### Sentiment Analysis of IMDB Reviews using Pandas and Naive Bayes Algorithms

This project performs sentiment analysis on IMDB movie reviews using data cleaning, preprocessing, exploratory data analysis (EDA), vectorization, and Naive Bayes classification techniques. The dataset contains two columns: movie reviews and their associated sentiment labels (positive or negative).

---

## Table of Contents
- [Introduction](#introduction)
- [Data Gathering](#data-gathering)
- [Data Cleaning](#data-cleaning)
- [Preprocessing](#preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Vectorization](#vectorization)
- [Modeling & Results](#modeling--results)
- [Libraries Used](#libraries-used)

---

## Introduction
The goal of this project is to classify IMDB movie reviews as positive or negative. This sentiment analysis task applies various NLP operations such as text cleaning, preprocessing, EDA, and vectorization. The cleaned and transformed text is later used to train Naive Bayes models for sentiment prediction.

---

## Data Gathering
The dataset was obtained from Kaggle and contains:
- **Reviews**: Textual IMDB movie reviews.
- **Sentiment**: Label specifying positive or negative sentiment.

---

## Data Cleaning
The following steps were applied to clean the text data:
- Lowercasing all reviews.
- Removing leading and trailing spaces.
- Removing URLs using Regular Expressions.
- Removing HTML tags using Regular Expressions.
- Correcting spelling using the TextBlob library.
- Removing punctuation marks.
- Removing special characters.

---

## Preprocessing
To prepare the reviews for NLP tasks, the following were performed:
- **Tokenization** using `nltk.tokenize.word_tokenize`.
- **Stop word removal** using `nltk.corpus.stopwords`.
- **Stemming** to reduce words to their root forms.

---

## Exploratory Data Analysis (EDA)
The dataset was analyzed using:
- Distribution of text length and word count.
- Most frequent unigrams, bigrams, and trigrams.
- Word cloud visualizations for high-frequency terms.

---

## Vectorization
Text data was converted into numerical format using:

### Bag of Words (BoW)
- Converts documents into a matrix of token frequency counts.
- Suitable for classical ML algorithms like Naive Bayes.

---

## Modeling & Results
Three Naive Bayes variants from scikit-learn were used:

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

## Model Accuracy Results
| Model         | Accuracy |
| ------------- | -------- |
| GaussianNB    | 0.7185   |
| MultinomialNB | 0.8295   |
| BernoulliNB   | 0.8350   |

## Interpretation
- GaussianNB is suited for continuous numeric data; therefore, its lower accuracy on text features is expected.

- MultinomialNB and BernoulliNB work effectively with discrete word-count features, leading to higher accuracy.

- BernoulliNB achieved the best overall performance for this sentiment classification task.

## Libraries Used
pandas
numpy
nltk
textblob
matplotlib
seaborn
scikit-learn
