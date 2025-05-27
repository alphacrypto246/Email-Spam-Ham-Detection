import pickle
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import os

# Print current working directory for debugging
print(f"Current working directory: {os.getcwd()}")

# Define file paths
MODEL_PATH = 'spam_detection_project/outputs/models/multinomial_nb_model.pkl'
VECTORIZER_PATH = 'spam_detection_project/outputs/models/vectorizer.pkl'

# Check if files exist
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found at 'C:/My Files/Programming Language/Python/NLP/Projects/spam_detection_project/outputs/models/multinomial_nb_model.pkl'.")
    exit(1)
if not os.path.exists(VECTORIZER_PATH):
    print(f"Error: Vectorizer file '{VECTORIZER_PATH}' not found at 'C:/My Files/Programming Language/Python/NLP/Projects/spam_detection_project/outputs/models/vectorizer.pkl'.")
    exit(1)

# Load the trained model
try:
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model file '{MODEL_PATH}': {str(e)}")
    exit(1)

# Load the vectorizer
try:
    with open(VECTORIZER_PATH, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    print("Vectorizer loaded successfully")
except Exception as e:
    print(f"Error loading vectorizer file '{VECTORIZER_PATH}': {str(e)}")
    exit(1)

def preprocess_text(text):
    """Preprocess input text to match training preprocessing."""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_spam(text):
    """Predict if the input text is spam or not."""
    # Preprocess the input text
    processed_text = preprocess_text(text)
    # Transform text using the loaded vectorizer
    text_vector = vectorizer.transform([processed_text])
    # Predict using the loaded model
    prediction = model.predict(text_vector)[0]
    # Return label (assuming 0 = not spam, 1 = spam)
    return "Spam" if prediction == 1 else "Not Spam"

def main():
    """Command-line interface for spam detection."""
    parser = argparse.ArgumentParser(description="Spam Detection using Multinomial Naive Bayes")
    parser.add_argument('--text', type=str, help="Text to classify as spam or not spam")
    args = parser.parse_args()

    if args.text:
        result = predict_spam(args.text)
        print(f"Input text: {args.text}")
        print(f"Prediction: {result}")
    else:
        print("Please provide text to classify using the --text argument.")
        print("Example: python run.py --text 'Win a free iPhone now!'")

if __name__ == "__main__":
    main()