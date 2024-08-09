# Fake News Classification Using Machine Learning and Deep Learning Models

# Project Overview

This project aims to build and evaluate models for classifying news articles as either "True" or "Fake" using various Machine Learning (ML) and Deep Learning (DL) algorithms. The dataset used for this project is retrieved from Kaggle and contains labeled news articles. The project involves data preprocessing, model building, evaluation, and selection of the best models for final use.

# Project Structure

dataset/: Contains the dataset files used in the project.

notebooks/: Jupyter notebooks with code for preprocessing, model building, evaluation, and analysis.

requirements.txt: List of dependencies required to run the project.

README.md: Overview and instructions for the project.

# Installation

To run this project locally, you'll need to install the required Python packages. You can do this by creating a virtual environment and installing the dependencies listed in requirements.txt.

# Install dependencies

pip install -r requirements.txt

# Dataset

The dataset is retrieved from Kaggle, and it consists of labeled news articles with the following columns:

title: The title of the news article.

text: The body of the news article.

subject: The subject category of the news article.

date: The publication date of the news article.

label: The label indicating whether the news is "True" (1) or "Fake" (0).

# Data Preprocessing

Before feeding the data into the models, the following preprocessing steps were performed:

Text Cleaning: Tokenization, stopword removal, and lemmatization.

Vectorization: Conversion of text data to numerical format using TF-IDF Vectorizer.

Tokenization and Padding (for DL models): Tokenizing the text and applying padding to ensure uniform input size.

# Models Implemented

## Machine Learning Models

Logistic Regression

Support Vector Machines (SVM)

Random Forest Classifier (Selected as the final ML model)

Gradient Boosting Classifier

Multinomial Naive Bayes (Performed poorly)

## Deep Learning Models

Deep Neural Network (DNN)

LSTM (Long Short-Term Memory)

Bidirectional LSTM (Selected as the final DL model)

# Model Evaluation and Selection

Random Forest Classifier was chosen as the final ML model due to its robustness, reduced overfitting, and feature importance insights.

Bidirectional LSTM was selected as the final DL model due to its superior performance in handling sequential data.

# How to Run the Project

Run the Jupyter Notebooks

Navigate to the notebooks/ directory and open the respective Jupyter notebooks to see the model building, evaluation, and analysis steps.

# Conclusion

This project successfully built and evaluated multiple models for fake news classification. The chosen models, Random Forest Classifier and Bidirectional LSTM, demonstrated high accuracy and reliability, making them suitable for practical deployment in identifying fake news.

# Future Work

Model Deployment: Deploy the selected models to a production environment.

Further Improvements: Experiment with different NLP techniques and models, such as BERT, to potentially improve accuracy.

# References

Kaggle Dataset: https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/data

# Contributors

Shruthi AK - Project Development
