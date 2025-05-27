# Email Spam-Ham Detection

This project builds a spam detection system to classify text messages or emails as **spam** or **ham** (not spam) using Natural Language Processing (NLP) and machine learning. It employs a [Multinomial Naive Bayes classifier](https://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes) with features extracted by [`TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html). The project includes Jupyter notebooks for preprocessing and training, a Python script for inference, and a labeled dataset. The aim is to improve communication security by detecting unwanted messages.

View the source code on [GitHub](https://github.com/alphacrypto246/Email-Spam-Ham-Detection).

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running the Notebooks](#running-the-notebooks)
- [Using run.py for Inference](#using-runpy-for-inference)
- [Dataset](#dataset)
- [Model and Preprocessing Details](#model-and-preprocessing-details)
- [Troubleshooting](#troubleshooting)
- [Notes](#notes)

## Project Structure

- [**`spam_detection_preprocessing.ipynb`**][preprocessing-notebook]: Loads the [`spam.csv`][dataset] dataset, cleans text (e.g., lowercase, remove punctuation, normalize whitespace), and converts text to numerical features using `TfidfVectorizer(max_features=3000)`. Saves the vectorizer as [`vectorizer.pkl`][vectorizer].
- [**`spam_detection_model_training.ipynb`**][training-notebook]: Trains a `MultinomialNB` classifier on preprocessed data and saves the model as [`multinomial_nb_model.pkl`][model].
- [**`run.py`**][run-script]: A command-line script to load the model and vectorizer, preprocess input text, and predict spam or ham.
- [**`spam.csv`**][dataset]: Dataset of text messages labeled "spam" or "ham" (likely the [SMS Spam Collection][sms-dataset]).
- [**`outputs/models/`**][models-dir]:
  - [**`multinomial_nb_model.pkl`**][model]: Saved Multinomial Naive Bayes model.
  - [**`vectorizer.pkl`**][vectorizer]: Saved `TfidfVectorizer` with `max_features=3000`.
- [**`requirements.txt`**][requirements]: Python dependencies for the project.
- [**`README.md`**][readme]: This documentation file.

## Prerequisites

- **Python**: Version 3.13 ([download](https://www.python.org/downloads/release/python-3130/)).
- **Dependencies**: See [`requirements.txt`][requirements]:
  ```
  scikit-learn==1.5.2
  pandas==2.2.3
  numpy==2.1.1
  jupyter==1.0.0
  ```

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/alphacrypto246/Email-Spam-Ham-Detection.git
   cd Email-Spam-Ham-Detection
   ```
   Or use the local path:
   ```bash
   cd "C:\My Files\Programming Language\Python\NLP\Projects\spam_detection_project"
   ```

2. **Install Dependencies**:
   ```bash
   "C:\Program Files\Python313\python.exe" -m pip install -r requirements.txt
   ```

3. **Verify Files**:
   - Ensure [`spam.csv`][dataset] is in the project root.
   - Check that [`multinomial_nb_model.pkl`][model] and [`vectorizer.pkl`][vectorizer] are in [`outputs/models/`][models-dir]. If missing, generate them via the notebooks (see [Running the Notebooks](#running-the-notebooks)).

## Running the Notebooks

1. **Launch Jupyter Notebook**:
   ```bash
   "C:\Program Files\Python313\python.exe" -m jupyter notebook
   ```

2. **Run [`spam_detection_preprocessing.ipynb`][preprocessing-notebook]**:
   - Open in Jupyter and execute all cells to load [`spam.csv`][dataset], preprocess text, and fit `TfidfVectorizer`.
   - Save the vectorizer:
     ```python
     import pickle
     import os
     os.makedirs('outputs/models', exist_ok=True)
     with open('outputs/models/vectorizer.pkl', 'wb') as vectorizer_file:
         pickle.dump(tfidf, vectorizer_file)
     print("Vectorizer saved as 'outputs/models/vectorizer.pkl'")
     ```
   - Check [`vectorizer.pkl`][vectorizer] in [`outputs/models/`][models-dir] (size ~10-100 KB).

3. **Run [`spam_detection_model_training.ipynb`][training-notebook]**:
   - Open in Jupyter and execute all cells to train the `MultinomialNB` model.
   - Save the model:
     ```python
     import pickle
     import os
     os.makedirs('outputs/models', exist_ok=True)
     with open('outputs/models/multinomial_nb_model.pkl', 'wb') as model_file:
         pickle.dump(model, model_file)
     print("Model saved as 'outputs/models/multinomial_nb_model.pkl'")
     ```
   - Check [`multinomial_nb_model.pkl`][model] in [`outputs/models/`][models-dir] (size ~1-10 KB).

## Using run.py for Inference

The [`run.py`][run-script] script predicts whether new text is spam or ham using the saved model and vectorizer.

1. **Execute the Script**:
   ```bash
   cd "C:\My Files\Programming Language\Python\NLP\Projects\spam_detection_project"
   "C:\Program Files\Python313\python.exe" run.py --text "Win a free iPhone now!"
   ```

2. **Example Output**:
   ```
   Current working directory: C:\My Files\Programming Language\Python\NLP\Projects\spam_detection_project
   Model loaded successfully
   Vectorizer loaded successfully
   Input text: Win a free iPhone now!
   Prediction: Spam
   ```

3. **Usage Notes**:
   - Use the `--text` argument to input text.
   - Without text:
     ```bash
     python run.py
     ```
     Output:
     ```
     Please provide text to classify using the --text argument.
     Example: python run.py --text 'Win a free iPhone now!'
     ```

## Dataset

- **File**: [`spam.csv`][dataset]
- **Description**: Likely the [SMS Spam Collection][sms-dataset], with ~5,572 text messages (~13.4% spam, ~86.6% ham). Spam messages often include keywords like "free," "win," or "txt."
- **Format**: CSV with columns (e.g., "Category" for "spam"/"ham", "Message" for text).
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

## Model and Preprocessing Details

- **Vectorizer**: [`TfidfVectorizer`][tfidf] with `max_features=3000` creates TF-IDF features.
- **Model**: [`MultinomialNB`][naive-bayes] classifier predicts binary labels (0 = ham, 1 = spam).
- **Preprocessing**: Text is lowercased, punctuation is removed, and whitespace is normalized. Ensure `preprocess_text` in [`run.py`][run-script] aligns with [`spam_detection_preprocessing.ipynb`][preprocessing-notebook].
