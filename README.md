# Fake News Detection Using Machine Learning

## Introduction
Fake news detection is a critical task in today's digital world, where misinformation can spread rapidly through social media and other online platforms. This project utilizes various machine learning algorithms to classify news articles as either "fake" or "real."

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Modeling](#modeling)
6. [Results](#results)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Contributing](#contributing)

## Project Overview
This project implements and compares the performance of several machine learning algorithms to detect fake news. The algorithms used include:
- Naive Bayes
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

## Dataset
The datasets used in this project are:
- `Fake.csv`: Contains fake news articles.
- `True.csv`: Contains real news articles.

Both datasets are preprocessed to remove any unwanted columns and text data is cleaned for better analysis.

## Data Preprocessing
Data preprocessing steps include:
- Combining the fake and real datasets with labels.
- Shuffling the data.
- Converting text to lowercase.
- Removing punctuation.
- Removing stopwords.

## Exploratory Data Analysis
Basic data exploration includes:
- Counting the number of articles per subject.
- Visualizing the distribution of articles using bar plots.
- Generating word clouds for fake and real news articles.
- Identifying the most frequent words in both fake and real news articles.

## Modeling
Several machine learning models are trained and evaluated:
- **Naive Bayes**: Achieved an accuracy of 95.31%
- **Logistic Regression**: Achieved an accuracy of 98.84%
- **Decision Tree**: Achieved an accuracy of 99.65%
- **Random Forest**: Achieved an accuracy of 98.94%
- **Support Vector Machine (SVM)**: Achieved an accuracy of 99.48%

Confusion matrices are plotted to visualize the performance of each model.

## Results
The Decision Tree model achieved the highest accuracy of 99.65%. The performance of each model is summarized in a bar plot for easy comparison.

## Installation
To run this project locally, you need to have Python and the following libraries installed:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- nltk
- wordcloud

You can install the required libraries using the following command:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/kishore-klassy/Fake-news-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Fake-news-prediction
   ```
3. Run the Jupyter Notebook or Python script to preprocess the data, train the models, and evaluate the results.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.
