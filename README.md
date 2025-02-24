# IMDB Sentiment Analysis using NLP and Logistic Regression

Welcome to the Sentiment Analysis project. This repository contains a machine learning solution that employs Natural Language Processing (NLP) techniques and Logistic Regression to classify movie reviews from the IMDB dataset as either positive or negative. This project demonstrates how text data can be transformed and analyzed to predict sentiment.

## Project Overview

The objective of this project is to build a predictive model that determines the sentiment (positive or negative) of a movie review based solely on its textual content. The model is trained on the IMDB movie reviews dataset, which comprises 50,000 reviews labeled as positive or negative.

## Dataset

The IMDB movie reviews dataset used for training and testing is not hosted on GitHub due to its size. You can download it from [this link](https://drive.google.com/file/d/1-Evj1WeUZLBtyTEMm7s9JF6h9Bct9vO_/view?usp=sharing).

## Key Components

- **Data Preprocessing**: Text cleaning (removing punctuation and stopwords) and tokenization.
- **TF-IDF Vectorization**: Converts cleaned text into numerical vectors suitable for machine learning.
- **Model**: Implements Logistic Regression for sentiment classification.
- **Evaluation**: Uses metrics such as accuracy, precision, recall, and a confusion matrix to assess model performance.
- **Interactive UI**: Provides an interface for users to input reviews and receive immediate sentiment predictions.

## Technologies Used

- **Python**
- **Pandas**: For data manipulation.
- **Numpy**: For numerical operations.
- **NLTK**: For natural language processing.
- **Scikit-learn**: For TF-IDF vectorization and model building.
- **Matplotlib & Seaborn**: For data visualization.
- **ipywidgets**: For creating the interactive UI.

## Results

- **Accuracy**: The model achieved 88% accuracy in classifying reviews.
- **Precision & Recall**: The performance is balanced across both positive and negative reviews, indicating strong generalization.

