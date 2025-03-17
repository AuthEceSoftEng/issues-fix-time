"""
Jira Issue Resolution Time Predictor: TF-IDF Based Models
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

from config import RANDOM_STATE, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, RF_N_ESTIMATORS

def train_tfidf_models(train_df, test_df, output_dir):
    """
    Train and evaluate TF-IDF based models
    
    Args:
        train_df: Training data DataFrame
        test_df: Test/validation data DataFrame
        output_dir: Directory to save models
        
    Returns:
        Dictionary of models and their accuracies
    """
    print("\n=== Training TF-IDF Based Models ===")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Prepare inputs and targets
    X_train = train_df['combined_text']
    y_train = train_df['time_category']
    X_test = test_df['combined_text']
    y_test = test_df['time_category']

    # Dictionary to store models and their accuracies
    models = {}

    # 1. TF-IDF + Logistic Regression
    print("\nTraining TF-IDF + Logistic Regression...")
    tfidf_lr_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE)),
        ('classifier', LogisticRegression(C=1.0, class_weight='balanced', solver='liblinear', max_iter=1000, random_state=RANDOM_STATE))
    ])

    tfidf_lr_pipeline.fit(X_train, y_train)
    y_pred = tfidf_lr_pipeline.predict(X_test)
    lr_accuracy = accuracy_score(y_test, y_pred)

    # Save model
    joblib.dump(tfidf_lr_pipeline, f"{output_dir}/tfidf_lr_model.pkl")

    # Save classification report
    lr_report = classification_report(y_test, y_pred)
    with open(f"{output_dir}/tfidf_lr_report.txt", 'w') as f:
        f.write(f"Accuracy: {lr_accuracy:.4f}\n\n")
        f.write(lr_report)

    models["TF-IDF + Logistic Regression"] = (tfidf_lr_pipeline, lr_accuracy)
    print(f"Accuracy: {lr_accuracy:.4f}")

    # 2. TF-IDF + Random Forest
    print("\nTraining TF-IDF + Random Forest...")
    tfidf_rf_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE)),
        ('classifier', RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, max_depth=None, min_samples_split=2, random_state=RANDOM_STATE))
    ])

    tfidf_rf_pipeline.fit(X_train, y_train)
    y_pred = tfidf_rf_pipeline.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred)

    # Save model
    joblib.dump(tfidf_rf_pipeline, f"{output_dir}/tfidf_rf_model.pkl")

    # Save classification report
    rf_report = classification_report(y_test, y_pred)
    with open(f"{output_dir}/tfidf_rf_report.txt", 'w') as f:
        f.write(f"Accuracy: {rf_accuracy:.4f}\n\n")
        f.write(rf_report)

    models["TF-IDF + Random Forest"] = (tfidf_rf_pipeline, rf_accuracy)
    print(f"Accuracy: {rf_accuracy:.4f}")

    # 3. TF-IDF + SVM
    print("\nTraining TF-IDF + SVM...")
    tfidf_svm_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE)),
        ('classifier', SVC(C=1.0, kernel='linear', probability=True, random_state=RANDOM_STATE))
    ])

    tfidf_svm_pipeline.fit(X_train, y_train)
    y_pred = tfidf_svm_pipeline.predict(X_test)
    svm_accuracy = accuracy_score(y_test, y_pred)

    # Save model
    joblib.dump(tfidf_svm_pipeline, f"{output_dir}/tfidf_svm_model.pkl")

    # Save classification report
    svm_report = classification_report(y_test, y_pred)
    with open(f"{output_dir}/tfidf_svm_report.txt", 'w') as f:
        f.write(f"Accuracy: {svm_accuracy:.4f}\n\n")
        f.write(svm_report)

    models["TF-IDF + SVM"] = (tfidf_svm_pipeline, svm_accuracy)
    print(f"Accuracy: {svm_accuracy:.4f}")

    # 4. TF-IDF + Ensemble (Voting Classifier)
    print("\nTraining TF-IDF + Ensemble (Voting)...")

    # Create vectorizer and transform the data
    tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Create base classifiers
    lr = LogisticRegression(C=1.0, class_weight='balanced', solver='liblinear', max_iter=1000, random_state=RANDOM_STATE)
    rf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, max_depth=None, min_samples_split=2, random_state=RANDOM_STATE)
    svm_clf = SVC(C=1.0, kernel='linear', probability=True, random_state=RANDOM_STATE)

    # Create ensemble
    tfidf_ensemble = VotingClassifier(
        estimators=[
            ('lr', lr),
            ('rf', rf),
            ('svm', svm_clf)
        ],
        voting='soft'
    )

    # Train and evaluate
    tfidf_ensemble.fit(X_train_tfidf, y_train)
    y_pred = tfidf_ensemble.predict(X_test_tfidf)
    ensemble_accuracy = accuracy_score(y_test, y_pred)

    # Save vectorizer and model separately
    joblib.dump(tfidf, f"{output_dir}/tfidf_vectorizer.pkl")
    joblib.dump(tfidf_ensemble, f"{output_dir}/tfidf_ensemble_model.pkl")

    # Save classification report
    ensemble_report = classification_report(y_test, y_pred)
    with open(f"{output_dir}/tfidf_ensemble_report.txt", 'w') as f:
        f.write(f"Accuracy: {ensemble_accuracy:.4f}\n\n")
        f.write(ensemble_report)

    models["TF-IDF + Ensemble"] = ((tfidf, tfidf_ensemble), ensemble_accuracy)
    print(f"Accuracy: {ensemble_accuracy:.4f}")

    # Compare models and select the best one
    print("\n=== TF-IDF Model Accuracy Comparison ===")
    for model_name, (_, accuracy) in sorted(models.items(), key=lambda x: x[1][1], reverse=True):
        print(f"{model_name}: {accuracy:.4f}")

    # Find the best model
    best_model_name = max(models, key=lambda k: models[k][1])
    best_model, best_accuracy = models[best_model_name]

    print(f"\nBest TF-IDF model: {best_model_name} with accuracy {best_accuracy:.4f}")

    return models