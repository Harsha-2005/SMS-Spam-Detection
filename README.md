# SMS-Spam-Detection
This project involves building a robust SMS spam detection system using a combination of traditional Machine Learning and Deep Learning techniques.
-----
# SMS Spam Detection: Comparative Analysis (ML vs. DL)

## Project Overview

This project implements and compares various Machine Learning (ML) and Deep Learning (DL) models to accurately classify SMS messages as either **"spam"** or **"ham"** (legitimate). The goal is to build an effective text classification system that can be deployed to filter unwanted messages, demonstrating proficiency in NLP techniques, model comparison, and performance optimization.

## Features

  * **Comprehensive Data Preprocessing:** Includes cleaning, tokenization, stemming/lemmatization (implied by the use of NLP libraries), and removal of punctuation, stopwords, and special characters.
  * **Feature Engineering:** Utilizes **Term Frequency-Inverse Document Frequency (TF-IDF)** vectorization for traditional ML models to capture the importance of words.
  * **Comparative Modeling:** Implementation and rigorous evaluation of three classic ML classifiers and one state-of-the-art DL model.
  * **Model Evaluation:** Performance metrics like **Accuracy, Precision, Recall, and F1-Score** are calculated and compared to determine the best-performing model.
  * **Interactive Prediction Function:** A utility function to quickly test new, user-input messages against all trained models.

## Models Implemented

The project focuses on comparing the performance of models across different algorithmic families:

| Category | Model | Feature Representation | Key Insight |
| :--- | :--- | :--- | :--- |
| **Traditional ML** | **Multinomial Naïve Bayes** | TF-IDF | Excellent baseline for text classification due to its simplicity and effectiveness. |
| **Traditional ML** | **Logistic Regression** | TF-IDF | A strong linear classifier, providing probabilistic output and good interpretability. |
| **Traditional ML** | **Support Vector Machine (SVM)** | TF-IDF | Effective in high-dimensional spaces (like a TF-IDF matrix) for finding the optimal separating hyperplane. |
| **Deep Learning** | **Long Short-Term Memory (LSTM)** | Word Embeddings (Keras Tokenizer) | A type of Recurrent Neural Network (RNN) capable of understanding sequence context and dependencies in text data. |

## Dataset

  * **Source:** `spam.csv` (A publicly available dataset containing \~5,572 SMS messages labeled as 'ham' or 'spam').
  * **Link of the DataSet:** https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
  * **Format:** Two columns: the label (`v1`) and the message text (`v2`).
  * **Challenge:** The dataset is highly imbalanced, requiring careful evaluation using metrics beyond simple accuracy.

## Technologies and Libraries Used

  * **Language:** Python
  * **Data Handling:** **Pandas**, **NumPy**
  * **Machine Learning:** **Scikit-learn** (Model Selection, Feature Extraction, Classification Models)
  * **Deep Learning:** **TensorFlow** / **Keras** (LSTM Model, Tokenization, Embedding Layer)
  * **Natural Language Processing (NLP):** **`re`** (Regular Expressions for cleaning), **`string`**
  * **Environment:** Jupyter Notebook

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Harsha-2005/SMS-Spam-Detection.git
    cd SMS-Spam-Detection
    ```
2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # .\venv\Scripts\activate # On Windows
    ```
3.  **Install the required libraries:**
    ```bash
    pip install pandas scikit-learn tensorflow jupyter
    ```
4.  **Run the notebook:**
    ```bash
    jupyter notebook "SMS Spam Detection.ipynb"
    ```
    Follow the steps in the notebook to execute the data loading, preprocessing, model training, evaluation, and testing.
