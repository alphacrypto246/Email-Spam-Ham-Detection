Email Spam-Ham Detection
This project implements a spam detection system to classify text messages or emails as "spam" or "ham" (not spam) using Natural Language Processing (NLP) and machine learning. It leverages a Multinomial Naive Bayes classifier with features extracted via TfidfVectorizer. The project includes Jupyter notebooks for data preprocessing and model training, a Python script for inference, and a dataset of labeled messages. The goal is to enhance communication security by accurately identifying unwanted or harmful messages.
Project Structure

spam_detection_preprocessing.ipynb: Loads and preprocesses the spam.csv dataset, applies text cleaning (e.g., lowercase, remove punctuation, normalize whitespace), and transforms text into numerical features using TfidfVectorizer(max_features=3000). Saves the vectorizer as vectorizer.pkl.
spam_detection_model_training.ipynb: Trains a MultinomialNB classifier on the preprocessed data and saves the model as multinomial_nb_model.pkl.
run.py: A command-line script to load the trained model and vectorizer, preprocess new text inputs, and predict spam or ham.
spam.csv: Dataset containing text messages labeled as "spam" or "ham" (likely the SMS Spam Collection dataset).
outputs/models/:
multinomial_nb_model.pkl: Saved Multinomial Naive Bayes model.
vectorizer.pkl: Saved TfidfVectorizer with max_features=3000.


requirements.txt: Lists Python dependencies required for the project.
README.md: This file, providing project overview and instructions.

Prerequisites

Python: Version 3.13 (used to create pickle files and run scripts).
Dependencies: Listed in requirements.txt:scikit-learn==1.5.2
pandas==2.2.3
numpy==2.1.1
jupyter==1.0.0



Setup

Clone the Repository:
git clone https://github.com/alphacrypto246/Email-Spam-Ham-Detection.git
cd Email-Spam-Ham-Detection


Install Dependencies:
"C:\Program Files\Python313\python.exe" -m pip install -r requirements.txt


Ensure Dataset and Pickle Files:

Verify spam.csv is in the project root (Email-Spam-Ham-Detection/).
Ensure multinomial_nb_model.pkl and vectorizer.pkl are in outputs/models/. If missing, run the notebooks to generate them (see below).



Running the Notebooks

Start Jupyter Notebook:
"C:\Program Files\Python313\python.exe" -m jupyter notebook


Run spam_detection_preprocessing.ipynb:

Open the notebook in Jupyter.
Execute all cells to load spam.csv, preprocess text (e.g., lowercase, remove punctuation, normalize whitespace), and fit TfidfVectorizer.
Save the vectorizer:import pickle
import os
os.makedirs('outputs/models', exist_ok=True)
with open('outputs/models/vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf, vectorizer_file)
print("Vectorizer saved as 'outputs/models/vectorizer.pkl'")


Verify the file appears in outputs/models/ and has a non-zero size (e.g., ~10-100 KB).


Run spam_detection_model_training.ipynb:

Open the notebook in Jupyter.
Execute all cells to train the MultinomialNB model on the preprocessed data.
Save the model:import pickle
import os
os.makedirs('outputs/models', exist_ok=True)
with open('outputs/models/multinomial_nb_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
print("Model saved as 'outputs/models/multinomial_nb_model.pkl'")


Verify the file appears in outputs/models/ and has a non-zero size (e.g., ~1-10 KB).



Using run.py for Inference
The run.py script classifies new text inputs as spam or ham using the saved model and vectorizer.

Run the Script:
cd "C:\My Files\Programming Language\Python\NLP\Projects\spam_detection_project"
"C:\Program Files\Python313\python.exe" run.py --text "Win a free iPhone now!"


Example Output:
Current working directory: C:\My Files\Programming Language\Python\NLP\Projects\spam_detection_project
Model loaded successfully
Vectorizer loaded successfully
Input text: Win a free iPhone now!
Prediction: Spam


Usage Notes:

Provide text via the --text argument.
If no text is provided:python run.py

Output:Please provide text to classify using the --text argument.
Example: python run.py --text 'Win a free iPhone now!'





Dataset

File: spam.csv (assumed to be the SMS Spam Collection dataset).
Description: Contains 5,572 text messages labeled as "spam" (e.g., promotional messages) or "ham" (legitimate messages). Common spam keywords include "free," "win," and "txt." Approximately 13.4% of messages are spam, and 86.6% are ham.
Format: CSV with columns for text and labels (e.g., "Category" and "Message").

Model and Preprocessing Details

Vectorizer: TfidfVectorizer with max_features=3000 transforms text into TF-IDF features.
Model: MultinomialNB classifier trained to predict binary labels (0 = ham, 1 = spam).
Preprocessing: Text is converted to lowercase, punctuation is removed, and whitespace is normalized. Ensure the preprocess_text function in run.py matches the preprocessing in spam_detection_preprocessing.ipynb.

Troubleshooting

FileNotFoundError:
Ensure spam.csv is in the project root.
Verify multinomial_nb_model.pkl and vectorizer.pkl exist in outputs/models/. If missing, re-run the notebooks.


Pickle Errors (e.g., invalid load key):
Check file sizes:dir "outputs\models"

Files should be non-zero (e.g., vectorizer.pkl ~10-100 KB, multinomial_nb_model.pkl ~1-10 KB).
Recreate files using the notebook code above.
Ensure Python 3.13 is used for saving and loading:"C:\Program Files\Python313\python.exe" -m jupyter notebook


Test loading manually:import pickle
try:
    with open('outputs/models/multinomial_nb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
try:
    with open('outputs/models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("Vectorizer loaded successfully")
except Exception as e:
    print(f"Error loading vectorizer: {str(e)}")




Preprocessing Mismatch: If predictions are inaccurate, ensure run.py’s preprocess_text matches the notebook’s preprocessing (e.g., add stopwords removal if used).

Notes

Dataset: Update the spam.csv description if using a different dataset.
Preprocessing: If spam_detection_preprocessing.ipynb includes additional steps (e.g., stopwords removal, lemmatization), update run.py’s preprocess_text function and note them here.
Contributing: Fork the repository and submit pull requests via GitHub: https://github.com/alphacrypto246/Email-Spam-Ham-Detection.

