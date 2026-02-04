"""
Quick test script to verify the sentiment analysis system is working
Run this after training the models to ensure everything is set up correctly
"""

import os
import sys

def test_imports():
    """Test if all required packages are installed"""
    print("Testing imports...")
    try:
        import pandas
        import numpy
        import nltk
        import sklearn
        import streamlit
        import plotly
        import xgboost
        import gensim
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        return False

def test_nltk_data():
    """Test if NLTK data is downloaded"""
    print("\nTesting NLTK data...")
    try:
        import nltk
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('tokenizers/punkt')
        print("✓ NLTK data is available")
        return True
    except LookupError:
        print("✗ NLTK data missing. Run: python -c \"import nltk; nltk.download('all')\"")
        return False

def test_data_files():
    """Test if preprocessed data files exist"""
    print("\nTesting data files...")
    data_files = [
        'data/processed_reviews.csv',
        'data/train_reviews.csv',
        'data/test_reviews.csv'
    ]
    
    all_exist = True
    for filepath in data_files:
        if os.path.exists(filepath):
            print(f"✓ {filepath} exists")
        else:
            print(f"✗ {filepath} not found - Run notebook 2")
            all_exist = False
    
    return all_exist

def test_model_files():
    """Test if model files exist"""
    print("\nTesting model files...")
    model_files = [
        'models/best_model.pkl',
        'models/vectorizer.pkl'
    ]
    
    all_exist = True
    for filepath in model_files:
        if os.path.exists(filepath):
            print(f"✓ {filepath} exists")
        else:
            print(f"✗ {filepath} not found - Run notebook 4")
            all_exist = False
    
    return all_exist

def test_model_loading():
    """Test if model can be loaded and used"""
    print("\nTesting model loading...")
    try:
        from utils import load_model_and_vectorizer, predict_sentiment
        
        model, vectorizer = load_model_and_vectorizer()
        print("✓ Model and vectorizer loaded successfully")
        
        # Test prediction
        test_review = "This product is amazing!"
        result = predict_sentiment(test_review, model, vectorizer)
        
        print(f"✓ Test prediction successful")
        print(f"  Review: '{test_review}'")
        print(f"  Sentiment: {result['sentiment']}")
        print(f"  Confidence: {result['confidence']:.2f}%")
        
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Flipkart Sentiment Analysis - System Test")
    print("=" * 60)
    print()
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("NLTK Data", test_nltk_data()))
    results.append(("Data Files", test_data_files()))
    results.append(("Model Files", test_model_files()))
    results.append(("Model Loading", test_model_loading()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("=" * 60)
    if all_passed:
        print("✓ All tests passed! System is ready.")
        print("\nYou can now run: streamlit run app.py")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Run notebooks 1-4 to generate data and models")
        print("- Install dependencies: pip install -r requirements.txt")
        print("- Download NLTK data: python -c \"import nltk; nltk.download('all')\"")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
