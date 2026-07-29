import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import os
import sys

try:
    print("Loading raw data...")
    # Read the raw scraper file
    data = pd.read_csv("hasil_scraper_ulasan_app_seabank.csv")
    
    # Drop rows without text
    data = data.dropna(subset=['Review Text'])
    data = data.sample(n=2000, random_state=42)

    
    # Simple labelling based on Rating
    def get_sentiment(rating):
        try:
            r = int(rating)
            if r < 3: return 'Negatif'
            elif r == 3: return 'Netral'
            else: return 'Positif'
        except:
            return 'Netral'
            
    data['Sentiment'] = data['Rating'].apply(get_sentiment)
    
    X = data['Review Text']
    y = data['Sentiment']

    print(f"Data size: {len(data)} rows")
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Vectorizing...")
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=3, max_features=3000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)

    print("SMOTE oversampling...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_vec, y_train)

    print("Initializing model and calculating learning curve...")
    # Use the new hyperparameters
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

    train_sizes, train_scores, val_scores = learning_curve(
        svm_model, X_train_smote, y_train_smote, cv=3, scoring='accuracy'
    )

    train_scores_mean = train_scores.mean(axis=1)
    val_scores_mean = val_scores.mean(axis=1)

    print("Plotting...")
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', label="Train Accuracy")
    plt.plot(train_sizes, val_scores_mean, 'o-', label="Validation Accuracy")
    plt.legend(loc='best')
    plt.xlabel("Train Set Size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve - SVM with SMOTE (RBF Kernel)")
    plt.grid(True)
    
    save_path = os.path.abspath('learning_curve.png')
    plt.savefig(save_path)
    print(f"Plot saved successfully to {save_path}")
    
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
