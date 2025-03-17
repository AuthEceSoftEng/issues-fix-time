"""
Jira Issue Resolution Time Predictor: BERT Based Models
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from config import RANDOM_STATE, RF_N_ESTIMATORS

def train_bert_models(train_df, test_df, output_dir):
    """
    Train and evaluate BERT-based models if sentence-transformers is available
    
    Args:
        train_df: Training data DataFrame
        test_df: Test/validation data DataFrame
        output_dir: Directory to save models
        
    Returns:
        Dictionary of models and their accuracies, or None if BERT not available
    """
    try:
        from sentence_transformers import SentenceTransformer
        print("\n=== Training BERT-Based Models ===")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Prepare inputs and targets
        X_train = train_df['combined_text'].tolist()
        y_train = train_df['time_category']
        X_test = test_df['combined_text'].tolist()
        y_test = test_df['time_category']

        # Dictionary to store models and their accuracies
        models = {}

        # Load BERT model and create embeddings
        print("Creating BERT embeddings (this may take a while)...")
        model_name = 'all-MiniLM-L6-v2'
        bert_model = SentenceTransformer(model_name)

        # Create embeddings (batched to manage memory)
        batch_size = 32
        X_train_embeddings = []

        for i in tqdm(range(0, len(X_train), batch_size), desc="Encoding training data"):
            batch_texts = X_train[i:i+batch_size]
            batch_embeddings = bert_model.encode(batch_texts, show_progress_bar=False)
            X_train_embeddings.extend(batch_embeddings)

        X_train_embeddings = np.array(X_train_embeddings)

        X_test_embeddings = []
        for i in tqdm(range(0, len(X_test), batch_size), desc="Encoding test data"):
            batch_texts = X_test[i:i+batch_size]
            batch_embeddings = bert_model.encode(batch_texts, show_progress_bar=False)
            X_test_embeddings.extend(batch_embeddings)

        X_test_embeddings = np.array(X_test_embeddings)

        # Save embeddings for future use
        np.save(f"{output_dir}/bert_embeddings_train.npy", X_train_embeddings)
        np.save(f"{output_dir}/bert_embeddings_test.npy", X_test_embeddings)
        np.save(f"{output_dir}/bert_labels_train.npy", y_train.values if hasattr(y_train, 'values') else y_train)
        np.save(f"{output_dir}/bert_labels_test.npy", y_test.values if hasattr(y_test, 'values') else y_test)

        # 1. BERT + Logistic Regression
        try:
            print("\nTraining BERT + Logistic Regression...")
            bert_lr = LogisticRegression(C=1.0, class_weight='balanced', solver='liblinear', max_iter=1000, random_state=RANDOM_STATE)
            bert_lr.fit(X_train_embeddings, y_train)
            y_pred = bert_lr.predict(X_test_embeddings)
            lr_accuracy = accuracy_score(y_test, y_pred)

            # Save model
            joblib.dump(bert_lr, f"{output_dir}/bert_lr_model.pkl")

            # Save classification report
            lr_report = classification_report(y_test, y_pred)
            with open(f"{output_dir}/bert_lr_report.txt", 'w') as f:
                f.write(f"Accuracy: {lr_accuracy:.4f}\n\n")
                f.write(lr_report)

            models["BERT + Logistic Regression"] = (bert_lr, lr_accuracy)
            print(f"Accuracy: {lr_accuracy:.4f}")
        except Exception as e:
            print(f"Error training BERT + Logistic Regression: {e}")

        # 2. BERT + Random Forest
        try:
            print("\nTraining BERT + Random Forest...")
            bert_rf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, max_depth=None, min_samples_split=2, random_state=RANDOM_STATE)
            bert_rf.fit(X_train_embeddings, y_train)
            y_pred = bert_rf.predict(X_test_embeddings)
            rf_accuracy = accuracy_score(y_test, y_pred)

            # Save model
            joblib.dump(bert_rf, f"{output_dir}/bert_rf_model.pkl")

            # Save classification report
            rf_report = classification_report(y_test, y_pred)
            with open(f"{output_dir}/bert_rf_report.txt", 'w') as f:
                f.write(f"Accuracy: {rf_accuracy:.4f}\n\n")
                f.write(rf_report)

            models["BERT + Random Forest"] = (bert_rf, rf_accuracy)
            print(f"Accuracy: {rf_accuracy:.4f}")
        except Exception as e:
            print(f"Error training BERT + Random Forest: {e}")

        # 3. BERT + Neural Network
        try:
            print("\nTraining BERT + Neural Network...")
            bert_nn = MLPClassifier(hidden_layer_sizes=(100,), alpha=0.0001, learning_rate='adaptive', max_iter=1000, random_state=RANDOM_STATE)
            bert_nn.fit(X_train_embeddings, y_train)
            y_pred = bert_nn.predict(X_test_embeddings)
            nn_accuracy = accuracy_score(y_test, y_pred)

            # Save model
            joblib.dump(bert_nn, f"{output_dir}/bert_nn_model.pkl")

            # Save classification report
            nn_report = classification_report(y_test, y_pred)
            with open(f"{output_dir}/bert_nn_report.txt", 'w') as f:
                f.write(f"Accuracy: {nn_accuracy:.4f}\n\n")
                f.write(nn_report)

            models["BERT + Neural Network"] = (bert_nn, nn_accuracy)
            print(f"Accuracy: {nn_accuracy:.4f}")
        except Exception as e:
            print(f"Error training BERT + Neural Network: {e}")

        # 4. BERT + Ensemble (Voting Classifier)
        # Only proceed with ensemble if we have at least 2 base models
        if len(models) >= 2:
            try:
                print("\nTraining BERT + Ensemble (Voting)...")
                
                estimators = []
                if "BERT + Logistic Regression" in models:
                    estimators.append(('lr', models["BERT + Logistic Regression"][0]))
                if "BERT + Random Forest" in models:
                    estimators.append(('rf', models["BERT + Random Forest"][0]))
                if "BERT + Neural Network" in models:
                    estimators.append(('nn', models["BERT + Neural Network"][0]))
                
                # Create ensemble
                bert_ensemble = VotingClassifier(
                    estimators=estimators,
                    voting='soft'
                )

                # Train and evaluate
                bert_ensemble.fit(X_train_embeddings, y_train)
                y_pred = bert_ensemble.predict(X_test_embeddings)
                ensemble_accuracy = accuracy_score(y_test, y_pred)

                # Save model
                joblib.dump(bert_ensemble, f"{output_dir}/bert_ensemble_model.pkl")

                # Save classification report
                ensemble_report = classification_report(y_test, y_pred)
                with open(f"{output_dir}/bert_ensemble_report.txt", 'w') as f:
                    f.write(f"Accuracy: {ensemble_accuracy:.4f}\n\n")
                    f.write(ensemble_report)

                models["BERT + Ensemble"] = (bert_ensemble, ensemble_accuracy)
                print(f"Accuracy: {ensemble_accuracy:.4f}")
            except Exception as e:
                print(f"Error training BERT + Ensemble: {e}")

        # Compare models and select the best one if we have any
        if models:
            print("\n=== BERT Model Accuracy Comparison ===")
            for model_name, (_, accuracy) in sorted(models.items(), key=lambda x: x[1][1], reverse=True):
                print(f"{model_name}: {accuracy:.4f}")

            # Find the best model
            best_model_name = max(models, key=lambda k: models[k][1])
            best_model, best_accuracy = models[best_model_name]

            print(f"\nBest BERT model: {best_model_name} with accuracy {best_accuracy:.4f}")
        else:
            print("\nNo BERT models were successfully trained.")

        return models

    except ImportError:
        print("\nSkipping BERT models (sentence-transformers not installed)")
        print("Install with: pip install sentence-transformers")
        return None