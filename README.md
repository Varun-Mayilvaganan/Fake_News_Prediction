---

# Fake News Prediction

## Overview

The Fake News Prediction project classifies news articles as either real or fake using machine learning. The project utilizes Natural Language Processing (NLP) techniques for text preprocessing and a Logistic Regression model for classification.

## Features

- **Text Preprocessing**: The text is cleaned and preprocessed by removing stopwords and applying stemming to reduce words to their root forms.
- **TF-IDF Vectorization**: Converts the text data into numerical format using Term Frequency-Inverse Document Frequency (TF-IDF).
- **Machine Learning Model**: A Logistic Regression model is used to classify the news articles as real or fake.
- **Evaluation**: The model is evaluated using accuracy scores on both training and test datasets.

## Dataset

The dataset consists of news articles labeled as either "real" (0) or "fake" (1). The dataset is loaded from Google Drive in CSV format.

## Execution Steps

1. **Mount Google Drive**:
   - Mount your Google Drive in the notebook to access the dataset.
   - Example:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```

2. **Load the Dataset**:
   - Load the dataset from the mounted Google Drive.
   - Example:
     ```python
     news_data = pd.read_csv('/content/drive/MyDrive/train.csv')
     ```

3. **Data Preprocessing**:
   - Handle missing values by replacing them with empty strings.
   - Merge the `author` and `title` columns into a single `content` column.
   - Apply text preprocessing including lowercasing, removing non-alphabetical characters, removing stopwords, and stemming.

4. **Vectorization**:
   - Convert the preprocessed text data into numerical format using TF-IDF vectorization.

5. **Model Training**:
   - Split the data into training and testing sets.
   - Train a Logistic Regression model using the training data.

6. **Evaluation**:
   - Evaluate the model on both training and test data by calculating accuracy scores.

7. **Prediction**:
   - Use the trained model to predict whether a new piece of news is real or fake.
   - Example:
     ```python
     x_new = X_test[1]
     prediction = model.predict(x_new)

     if prediction[0] == 0:
         print("Real News")
     else:
         print("Fake News")
     ```

## Conclusion

This notebook provides a comprehensive workflow for building and evaluating a machine learning model to classify fake news. It covers the entire process from data loading and preprocessing to model training, evaluation, and prediction, all within a single Jupyter notebook.

---
