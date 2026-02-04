"""
Utility functions for sentiment analysis application
"""

import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data with better error handling
def download_nltk_data(resource_name, download_name=None):
    """Download NLTK data with error handling for corrupted files"""
    if download_name is None:
        download_name = resource_name.split('/')[-1]
    
    try:
        nltk.data.find(resource_name)
    except (LookupError, Exception):
        # If not found or corrupted, download it
        try:
            nltk.download(download_name, quiet=True)
        except Exception:
            # If download fails, try without quiet mode for debugging
            pass

# Download all required NLTK resources
download_nltk_data('corpora/stopwords', 'stopwords')
download_nltk_data('corpora/wordnet', 'wordnet')
download_nltk_data('tokenizers/punkt', 'punkt')
download_nltk_data('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
download_nltk_data('corpora/omw-1.4', 'omw-1.4')


# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Remove negation words from stop words (important for sentiment)
negation_words = {'not', 'no', 'nor', 'neither', 'never', 'none', 'nothing', 'nobody', 'nowhere'}
stop_words = stop_words - negation_words


def clean_text(text):
    """
    Comprehensive text cleaning:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove HTML tags
    4. Remove special characters and digits
    5. Remove extra whitespace
    """
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits (keep only letters)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def remove_stopwords(text):
    """
    Remove stop words while preserving negation words
    """
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)


def lemmatize_text(text):
    """
    Lemmatize words to their base form
    """
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)


def preprocess_text(text):
    """
    Complete preprocessing pipeline
    """
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text


def load_model_and_vectorizer(model_path='models/best_model.pkl', 
                              vectorizer_path='models/vectorizer.pkl'):
    """
    Load the trained model and vectorizer
    
    Args:
        model_path: Path to the saved model
        vectorizer_path: Path to the saved vectorizer
        
    Returns:
        tuple: (model, vectorizer)
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        return model, vectorizer
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model or vectorizer file not found: {e}")
    except Exception as e:
        raise Exception(f"Error loading model or vectorizer: {e}")


def predict_sentiment(text, model, vectorizer):
    """
    Predict sentiment for a given text
    
    Args:
        text: Input review text
        model: Trained classification model
        vectorizer: Fitted vectorizer
        
    Returns:
        dict: Dictionary containing prediction and confidence scores
    """
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Check if text is empty after preprocessing
    if not processed_text.strip():
        return {
            'sentiment': 'Unknown',
            'confidence': 0.0,
            'positive_prob': 0.0,
            'negative_prob': 0.0,
            'error': 'Text is empty after preprocessing'
        }
    
    # Transform using vectorizer
    text_vector = vectorizer.transform([processed_text])
    
    # Make prediction
    prediction = model.predict(text_vector)[0]
    probabilities = model.predict_proba(text_vector)[0]
    
    # Prepare results
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    confidence = probabilities[prediction] * 100
    
    return {
        'sentiment': sentiment,
        'confidence': confidence,
        'positive_prob': probabilities[1] * 100,
        'negative_prob': probabilities[0] * 100,
        'processed_text': processed_text
    }


def load_model_metadata(metadata_path='models/model_metadata.pkl'):
    """
    Load model metadata
    
    Args:
        metadata_path: Path to metadata file
        
    Returns:
        dict: Model metadata
    """
    try:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        return metadata
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None


def get_sentiment_color(sentiment):
    """
    Get color for sentiment visualization
    
    Args:
        sentiment: 'Positive' or 'Negative'
        
    Returns:
        str: Hex color code
    """
    if sentiment == 'Positive':
        return '#2ecc71'  # Green
    elif sentiment == 'Negative':
        return '#e74c3c'  # Red
    else:
        return '#95a5a6'  # Gray


def get_confidence_level(confidence):
    """
    Get confidence level description
    
    Args:
        confidence: Confidence percentage (0-100)
        
    Returns:
        str: Confidence level description
    """
    if confidence >= 90:
        return "Very High"
    elif confidence >= 75:
        return "High"
    elif confidence >= 60:
        return "Moderate"
    else:
        return "Low"
