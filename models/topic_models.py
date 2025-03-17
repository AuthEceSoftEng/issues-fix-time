"""
Jira Issue Resolution Time Predictor: Topic Models
"""

import os
import joblib
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def train_topic_models(train_df, test_df, output_dir):
    """
    Extract topics from issue text and build topic-based prediction models
    
    Args:
        train_df: Training data DataFrame
        test_df: Test/validation data DataFrame
        output_dir: Directory to save models
        
    Returns:
        Dictionary with topic model and accuracy metrics
    """
    try:
        from bertopic import BERTopic
        print("\n=== Training Topic-Based Models ===")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Clean text function for topic modeling
        def clean_text(text):
            if not isinstance(text, str) or pd.isna(text):
                return ""

            # Remove HTML tags
            cleaned_text = re.sub(r'<[^>]+>', '', text)

            # Replace newlines with spaces
            cleaned_text = cleaned_text.replace('\n', ' ')

            # Remove links
            cleaned_text = re.sub(r'http\S+', '', cleaned_text)

            # Additional preprocessing
            cleaned_text = re.sub(r'[^\w\s]', ' ', cleaned_text.lower())

            # Remove numbers
            cleaned_text = re.sub(r'\d+', '', cleaned_text)

            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            tokens = cleaned_text.split()
            tokens = [word for word in tokens if word not in stop_words]

            return ' '.join(tokens)

        # Combine text columns and clean
        print("Preparing issue text for topic modeling...")

        # Process training data
        train_texts = []
        for _, row in train_df.iterrows():
            combined_text = " ".join([str(row.get('summary', "")), str(row.get('description', ""))])
            cleaned_text = clean_text(combined_text)
            train_texts.append(cleaned_text)

        # Process test data
        test_texts = []
        for _, row in test_df.iterrows():
            combined_text = " ".join([str(row.get('summary', "")), str(row.get('description', ""))])
            cleaned_text = clean_text(combined_text)
            test_texts.append(cleaned_text)

        # Filter out empty texts from training
        non_empty_indices = [i for i, text in enumerate(train_texts) if text.strip()]
        filtered_train_texts = [train_texts[i] for i in non_empty_indices]
        filtered_train_categories = train_df['time_category'].iloc[non_empty_indices].reset_index(drop=True)

        print(f"Using {len(filtered_train_texts)} non-empty texts for topic modeling")

        # Apply topic modeling
        print("\nApplying BERTopic for topic modeling...")
        vectorizer_model = CountVectorizer(stop_words="english")
        topic_model = BERTopic(
            verbose=True,
            vectorizer_model=vectorizer_model,
            calculate_probabilities=True,
            nr_topics="auto"
        )

        # Fit model and transform training data
        train_topics, train_probs = topic_model.fit_transform(filtered_train_texts)

        # Get topic info
        topic_info = topic_model.get_topic_info()

        # Display top topics
        print("\nTop 10 topics:")
        for i, row in topic_info.head(10).iterrows():
            topic_words = []
            if row['Topic'] != -1:  # Skip -1 "outlier" topic
                topic = topic_model.get_topic(row['Topic'])
                topic_words = [word for word, _ in topic[:5]]
            print(f"Topic {row['Topic']}: {', '.join(topic_words)}")

        # Save topic model
        joblib.dump(topic_model, f"{output_dir}/topic_model.pkl")

        # Calculate topic-resolution relationships
        print("\nCalculating topic-resolution relationships...")

        # Create DataFrame with topics and resolution categories
        topic_df = pd.DataFrame({
            'topic': train_topics,
            'time_category': filtered_train_categories
        })

        # Filter out outlier topic (-1)
        topic_df_filtered = topic_df[topic_df['topic'] != -1].copy()
        print(f"Filtered out {len(topic_df) - len(topic_df_filtered)} outlier issues (topic -1)")

        # Calculate resolution distribution by topic
        topic_resolution = pd.crosstab(
            topic_df_filtered['topic'],
            topic_df_filtered['time_category'],
            normalize='index'
        ) * 100

        # Add issue count for each topic
        topic_counts = topic_df_filtered['topic'].value_counts()
        topic_resolution['count'] = topic_counts

        # Save topic resolution probabilities
        topic_resolution.to_csv(f"{output_dir}/topic_resolution_percentages.csv")

        # Test on the test set
        print("\nTesting topic model on test set...")

        # Transform test data
        test_topics, test_probs = topic_model.transform(test_texts)

        # Create dictionary to hold predicted categories
        topic_predictions = []

        # For each test issue, get the topic and its resolution probability distribution
        for i, (topic, topic_prob) in enumerate(zip(test_topics, test_probs)):
            if topic != -1 and topic in topic_resolution.index:
                # Get probability distribution for this topic
                prob_dist = topic_resolution.loc[topic].drop('count').to_dict()

                # Get most likely resolution category
                most_likely = max(prob_dist, key=prob_dist.get)

                # Store prediction
                topic_predictions.append({
                    'index': i,
                    'topic': topic,
                    'predicted_category': most_likely,
                    'actual_category': test_df['time_category'].iloc[i],
                    'probabilities': prob_dist
                })
            else:
                # For outlier topics, we can't make a prediction
                topic_predictions.append({
                    'index': i,
                    'topic': topic,
                    'predicted_category': None,
                    'actual_category': test_df['time_category'].iloc[i],
                    'probabilities': {}
                })

        # Calculate topic model accuracy (excluding outliers)
        valid_predictions = [p for p in topic_predictions if p['predicted_category'] is not None]
        if valid_predictions:
            correct = sum(1 for p in valid_predictions if p['predicted_category'] == p['actual_category'])
            accuracy = correct / len(valid_predictions)
            print(f"Topic model accuracy on test set: {accuracy:.4f} (on {len(valid_predictions)} non-outlier topics)")
        else:
            print("No valid predictions from topic model (all test issues were outliers)")
            accuracy = 0

        # Save topic predictions
        pd.DataFrame(topic_predictions).to_csv(f"{output_dir}/topic_predictions.csv", index=False)

        return {
            'topic_model': topic_model,
            'topic_resolution': topic_resolution,
            'accuracy': accuracy
        }

    except ImportError:
        print("\nSkipping topic models (bertopic not installed)")
        print("Install with: pip install bertopic")
        return None