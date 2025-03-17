"""
Jira Issue Resolution Time Predictor: Stacked ML Predictor

This module implements a stacked machine learning approach for predicting
Jira issue resolution times by combining multiple base models.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

from config import RANDOM_STATE, RF_N_ESTIMATORS

class ResolutionTimeStackedPredictor:
    """
    Stacked machine learning approach for predicting Jira issue resolution times.
    Combines multiple base models using a meta-learner.
    """

    def __init__(self, output_dir=None):
        """
        Initialize the predictor

        Args:
            output_dir: Directory to save models and results
        """
        self.output_dir = output_dir

        # Base models and distributions
        self.text_model = None
        self.topic_model = None
        self.topic_resolution = None
        self.label_distributions = None
        self.component_distributions = None
        self.assignee_distributions = None
        self.priority_distributions = None

        # Meta-learner
        self.meta_learner = None
        self.resolution_categories = ['Less than 0.5 days', '0.5-2 days', '2-5 days', 'More than 5 days']

    def load_models(self, text_model_path=None, topic_model_path=None, topic_resolution_path=None):
        """
        Load the text-based ML model and topic model

        Args:
            text_model_path: Path to the saved text model
            topic_model_path: Path to the saved topic model
            topic_resolution_path: Path to the topic resolution probabilities
        """
        # Load text model if provided
        if text_model_path and os.path.exists(text_model_path):
            try:
                print(f"Loading text model from {text_model_path}")
                self.text_model = joblib.load(text_model_path)
                print(f"Successfully loaded text model of type: {type(self.text_model).__name__}")

                # Check if the model has predict_proba method
                if hasattr(self.text_model, 'predict_proba'):
                    print(f"Model has predict_proba method")
                else:
                    print(f"WARNING: Model does not have predict_proba method")

                # Check if the model has classes_ attribute
                if hasattr(self.text_model, 'classes_'):
                    print(f"Model has classes: {self.text_model.classes_}")
                else:
                    print(f"WARNING: Model does not have classes_ attribute")
            except Exception as e:
                print(f"Error loading text model: {e}")
                import traceback
                traceback.print_exc()

        # Load topic model if provided
        if topic_model_path and os.path.exists(topic_model_path):
            try:
                self.topic_model = joblib.load(topic_model_path)
                print(f"Loaded topic model from {topic_model_path}")
            except Exception as e:
                print(f"Error loading topic model: {e}")

        # Load topic resolution probabilities if provided
        if topic_resolution_path and os.path.exists(topic_resolution_path):
            try:
                self.topic_resolution = pd.read_csv(topic_resolution_path, index_col=0)
                print(f"Loaded topic resolution probabilities from {topic_resolution_path}")
            except Exception as e:
                print(f"Error loading topic resolution probabilities: {e}")

    def load_distributions(self, label_dist_path=None, component_dist_path=None,
                         assignee_dist_path=None, priority_dist_path=None):
        """
        Load the probability distributions for labels, components, assignees, and priorities

        Args:
            label_dist_path: Path to label distribution CSV
            component_dist_path: Path to component distribution CSV
            assignee_dist_path: Path to assignee distribution CSV
            priority_dist_path: Path to priority distribution CSV
        """
        # Load label distributions
        if label_dist_path and os.path.exists(label_dist_path):
            try:
                self.label_distributions = pd.read_csv(label_dist_path, index_col=0)
                print(f"Loaded label distributions from {label_dist_path}")
            except Exception as e:
                print(f"Error loading label distributions: {e}")

        # Load component distributions
        if component_dist_path and os.path.exists(component_dist_path):
            try:
                self.component_distributions = pd.read_csv(component_dist_path, index_col=0)
                print(f"Loaded component distributions from {component_dist_path}")
            except Exception as e:
                print(f"Error loading component distributions: {e}")

        # Load assignee distributions
        if assignee_dist_path and os.path.exists(assignee_dist_path):
            try:
                self.assignee_distributions = pd.read_csv(assignee_dist_path, index_col=0)
                print(f"Loaded assignee distributions from {assignee_dist_path}")
            except Exception as e:
                print(f"Error loading assignee distributions: {e}")

        # Load priority distributions
        if priority_dist_path and os.path.exists(priority_dist_path):
            try:
                self.priority_distributions = pd.read_csv(priority_dist_path, index_col=0)
                print(f"Loaded priority distributions from {priority_dist_path}")
            except Exception as e:
                print(f"Error loading priority distributions: {e}")

    def _get_base_model_predictions(self, issue_text, issue_labels=None, issue_components=None,
                                  issue_assignee=None, issue_priority=None, issue_type=None):
        """
        Get predictions from all base models without aggregation

        Args:
            issue_text: Issue text (summary + description)
            issue_labels: List of labels for the issue
            issue_components: List of components for the issue
            issue_assignee: Assignee username
            issue_priority: Priority level
            issue_type: Issue type

        Returns:
            Dictionary with predictions from each model
        """
        predictions = {}

        # 1. Text-based ML model prediction
        if self.text_model:
            try:
                # Prepare input text
                text_input = issue_text if isinstance(issue_text, list) else [issue_text]

                # Initialize prediction dict
                text_model_pred = {}

                # Try different methods to get predictions
                if hasattr(self.text_model, 'predict_proba'):
                    text_probs = self.text_model.predict_proba(text_input)[0]

                    # Convert to dictionary with category names
                    if hasattr(self.text_model, 'classes_'):
                        for i, category in enumerate(self.text_model.classes_):
                            text_model_pred[category] = text_probs[i] * 100
                    else:
                        # Use generic categories
                        for i, category in enumerate(self.resolution_categories):
                            if i < len(text_probs):
                                text_model_pred[category] = text_probs[i] * 100

                elif hasattr(self.text_model, 'steps') and len(self.text_model.steps) > 1:
                    # This is likely a Pipeline with a classifier as the last step
                    classifier = self.text_model.steps[-1][1]
                    if hasattr(classifier, 'predict_proba'):
                        # Get the transformed data from earlier pipeline steps
                        transformed_data = text_input
                        for name, transform in self.text_model.steps[:-1]:
                            transformed_data = transform.transform(transformed_data)

                        # Use the classifier directly
                        text_probs = classifier.predict_proba(transformed_data)[0]
                        for i, category in enumerate(classifier.classes_):
                            text_model_pred[category] = text_probs[i] * 100
                    else:
                        # Use predict instead of predict_proba if unavailable
                        prediction = self.text_model.predict(text_input)[0]
                        # Set 100% for the predicted class
                        text_model_pred[prediction] = 100.0

                else:
                    # Fallback to predict if predict_proba is not available
                    prediction = self.text_model.predict(text_input)[0]
                    # Set 100% for the predicted class
                    text_model_pred[prediction] = 100.0

                # Only add to predictions if we actually got some predictions
                if text_model_pred:
                    predictions['text_model'] = text_model_pred

            except Exception as e:
                print(f"Error getting text model predictions: {e}")

        # 2. Topic-based model prediction
        if self.topic_model and self.topic_resolution is not None:
            try:
                # Get topic prediction
                topics, _ = self.topic_model.transform([issue_text])
                topic = topics[0]

                # Get resolution probabilities for this topic
                if topic in self.topic_resolution.index:
                    # Create a copy of the row without modifying the original
                    topic_row = self.topic_resolution.loc[topic].copy()
                    # Drop the count column if it exists
                    if 'count' in topic_row.index:
                        topic_row = topic_row.drop('count')
                    # Convert to dictionary
                    topic_pred = topic_row.to_dict()
                    predictions['topic_model'] = topic_pred
            except Exception as e:
                print(f"Warning: Error getting topic model predictions: {e}")

        # 3. Label distribution prediction
        if self.label_distributions is not None and issue_labels:
            try:
                # Get probabilities for each label and average them
                valid_labels = [label for label in issue_labels if label in self.label_distributions.index]

                if valid_labels:
                    label_pred = {}

                    # Get mean probability across all valid labels
                    for category in self.resolution_categories:
                        if category in self.label_distributions.columns:
                            probs = [self.label_distributions.loc[label, category] for label in valid_labels]
                            label_pred[category] = sum(probs) / len(probs)

                    predictions['label_dist'] = label_pred
            except Exception as e:
                print(f"Warning: Error getting label distribution predictions: {e}")

        # 4. Component distribution prediction
        if self.component_distributions is not None and issue_components:
            try:
                valid_components = [comp for comp in issue_components if comp in self.component_distributions.index]

                if valid_components:
                    component_pred = {}

                    for category in self.resolution_categories:
                        if category in self.component_distributions.columns:
                            probs = [self.component_distributions.loc[comp, category] for comp in valid_components]
                            component_pred[category] = sum(probs) / len(probs)

                    predictions['component_dist'] = component_pred
            except Exception as e:
                print(f"Warning: Error getting component distribution predictions: {e}")

        # 5. Assignee distribution prediction
        if self.assignee_distributions is not None and issue_assignee:
            try:
                if issue_assignee in self.assignee_distributions.index:
                    assignee_pred = {}

                    for category in self.resolution_categories:
                        if category in self.assignee_distributions.columns:
                            assignee_pred[category] = self.assignee_distributions.loc[issue_assignee, category]

                    predictions['assignee_dist'] = assignee_pred
            except Exception as e:
                print(f"Warning: Error getting assignee distribution predictions: {e}")

        # 6. Priority distribution prediction
        if self.priority_distributions is not None and issue_priority:
            try:
                if issue_priority in self.priority_distributions.index:
                    priority_pred = {}

                    for category in self.resolution_categories:
                        if category in self.priority_distributions.columns:
                            priority_pred[category] = self.priority_distributions.loc[issue_priority, category]

                    predictions['priority_dist'] = priority_pred
            except Exception as e:
                print(f"Warning: Error getting priority distribution predictions: {e}")

        return predictions

    def _create_meta_features(self, base_predictions, issue_priority=None, issue_type=None):
        """
        Convert base model predictions to a feature vector for the meta-learner

        Args:
            base_predictions: Dictionary with predictions from each model
            issue_priority: Priority level
            issue_type: Issue type

        Returns:
            List with feature values
        """
        feature_vector = []

        # Add text model predictions
        if 'text_model' in base_predictions:
            for category in self.resolution_categories:
                feature_vector.append(base_predictions['text_model'].get(category, 0.0))
        else:
            feature_vector.extend([0.0] * len(self.resolution_categories))

        # Add topic model predictions
        if 'topic_model' in base_predictions:
            for category in self.resolution_categories:
                feature_vector.append(base_predictions['topic_model'].get(category, 0.0))
        else:
            feature_vector.extend([0.0] * len(self.resolution_categories))

        # Add label distribution predictions
        if 'label_dist' in base_predictions:
            for category in self.resolution_categories:
                feature_vector.append(base_predictions['label_dist'].get(category, 0.0))
        else:
            feature_vector.extend([0.0] * len(self.resolution_categories))

        # Add component distribution predictions
        if 'component_dist' in base_predictions:
            for category in self.resolution_categories:
                feature_vector.append(base_predictions['component_dist'].get(category, 0.0))
        else:
            feature_vector.extend([0.0] * len(self.resolution_categories))

        # Add assignee distribution predictions
        if 'assignee_dist' in base_predictions:
            for category in self.resolution_categories:
                feature_vector.append(base_predictions['assignee_dist'].get(category, 0.0))
        else:
            feature_vector.extend([0.0] * len(self.resolution_categories))

        # Add priority distribution predictions
        if 'priority_dist' in base_predictions:
            for category in self.resolution_categories:
                feature_vector.append(base_predictions['priority_dist'].get(category, 0.0))
        else:
            feature_vector.extend([0.0] * len(self.resolution_categories))

        # Add metadata features for priority
        if issue_priority:
            feature_vector.append(1.0 if issue_priority == 'Blocker' else 0.0)
            feature_vector.append(1.0 if issue_priority == 'Critical' else 0.0)
            feature_vector.append(1.0 if issue_priority == 'Major' else 0.0)
            feature_vector.append(1.0 if issue_priority == 'Minor' else 0.0)
            feature_vector.append(1.0 if issue_priority == 'Trivial' else 0.0)
        else:
            feature_vector.extend([0.0] * 5)  # 5 priority levels

        # Add metadata features for issue type
        if issue_type:
            feature_vector.append(1.0 if issue_type == 'Bug' else 0.0)
            feature_vector.append(1.0 if issue_type == 'Improvement' else 0.0)
            feature_vector.append(1.0 if issue_type == 'New Feature' else 0.0)
            feature_vector.append(1.0 if issue_type == 'Task' else 0.0)
            feature_vector.append(1.0 if issue_type == 'Sub-task' else 0.0)
        else:
            feature_vector.extend([0.0] * 5)  # 5 issue types

        return feature_vector

    def train_meta_learner(self, validation_df):
        """
        Train a meta-learner model that learns how to combine predictions from base models

        Args:
            validation_df: DataFrame with validation data to train the meta-learner
            
        Returns:
            Accuracy of the meta-learner on the validation set
        """
        print("\n=== Training Meta-Learner ===")

        # Collect meta-features (predictions from base models)
        meta_features = []
        targets = []

        # Process each validation issue
        for _, row in validation_df.iterrows():
            # Get issue properties
            issue_text = row.get('combined_text', '')
            issue_labels = row.get('labels', [])
            issue_components = row.get('components', [])
            issue_assignee = row.get('assignee')
            issue_priority = row.get('priority')
            issue_type = row.get('issuetype')
            actual_category = row.get('time_category')

            # Get predictions from all base models
            base_predictions = self._get_base_model_predictions(
                issue_text,
                issue_labels,
                issue_components,
                issue_assignee,
                issue_priority,
                issue_type
            )

            # Convert predictions to a feature vector
            feature_vector = self._create_meta_features(
                base_predictions,
                issue_priority,
                issue_type
            )

            # Add the feature vector and target
            meta_features.append(feature_vector)
            targets.append(actual_category)

        # Train the meta-learner
        print(f"Training meta-learner on {len(meta_features)} validation examples")
        self.meta_learner = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        )
        self.meta_learner.fit(meta_features, targets)

        # Evaluate the meta-learner on validation data
        predictions = self.meta_learner.predict(meta_features)
        accuracy = accuracy_score(targets, predictions)
        print(f"Meta-learner validation accuracy: {accuracy:.4f}")
        print(classification_report(targets, predictions))

        # Print feature importances
        feature_names = []
        model_names = ['text_model', 'topic_model', 'label_dist', 'component_dist', 'assignee_dist', 'priority_dist']
        for model in model_names:
            for category in self.resolution_categories:
                feature_names.append(f"{model}_{category}")

        # Add metadata feature names
        feature_names.extend(['priority_Blocker', 'priority_Critical', 'priority_Major', 'priority_Minor', 'priority_Trivial'])
        feature_names.extend(['type_Bug', 'type_Improvement', 'type_NewFeature', 'type_Task', 'type_Subtask'])

        # Get feature importances
        importances = self.meta_learner.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': feature_names[:len(importances)],  # In case we have fewer importances than feature names
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        print("\nTop 10 most important features:")
        print(feature_importance.head(10).to_string(index=False))

        # Save the meta-learner
        if self.output_dir:
            joblib.dump(self.meta_learner, f"{self.output_dir}/meta_learner_model.pkl")
            print(f"Meta-learner saved to {self.output_dir}/meta_learner_model.pkl")

            # Save feature importances
            feature_importance.to_csv(f"{self.output_dir}/meta_learner_feature_importances.csv", index=False)

        return accuracy

    def predict(self, issue_text, issue_labels=None, issue_components=None,
               issue_assignee=None, issue_priority=None, issue_type=None):
        """
        Make a prediction using the meta-learner or weighted averaging

        Args:
            issue_text: Issue text (summary + description)
            issue_labels: List of labels for the issue
            issue_components: List of components for the issue
            issue_assignee: Assignee username
            issue_priority: Priority level
            issue_type: Issue type

        Returns:
            Dictionary with prediction results
        """
        result = {}

        # Get base model predictions
        base_predictions = self._get_base_model_predictions(
            issue_text,
            issue_labels,
            issue_components,
            issue_assignee,
            issue_priority,
            issue_type
        )

        # Store individual predictions
        result['individual_predictions'] = base_predictions

        # If we have a meta-learner, use it
        if self.meta_learner:
            # Prepare feature vector for meta-learner
            feature_vector = self._create_meta_features(
                base_predictions,
                issue_priority,
                issue_type
            )

            # Get prediction probabilities from meta-learner
            meta_probs = self.meta_learner.predict_proba([feature_vector])[0]

            # Format as dictionary with category names
            aggregated_probs = {}
            for i, category in enumerate(self.meta_learner.classes_):
                aggregated_probs[category] = round(meta_probs[i] * 100, 1)

            result['aggregated_prediction'] = aggregated_probs
            result['model_contributions'] = {'meta_learner': 1.0}

        else:
            # Fall back to simple averaging if no meta-learner
            aggregated_probs = {category: 0.0 for category in self.resolution_categories}
            model_weights = {
                'text_model': 0.3,
                'topic_model': 0.2,
                'label_dist': 0.2,
                'component_dist': 0.15,
                'assignee_dist': 0.075,
                'priority_dist': 0.075
            }

            model_contributions = {}

            # Apply weighted averaging
            for model, predictions in base_predictions.items():
                weight = model_weights.get(model, 0.0)
                if weight > 0:
                    model_contributions[model] = weight
                    for category, prob in predictions.items():
                        if category in aggregated_probs:
                            aggregated_probs[category] += prob * weight

            # Normalize if we have any contributions
            total_weight = sum(model_contributions.values())
            if total_weight > 0 and total_weight != 1.0:
                for category in aggregated_probs:
                    aggregated_probs[category] /= total_weight

            # Round probabilities
            for category in aggregated_probs:
                aggregated_probs[category] = round(aggregated_probs[category], 1)

            result['aggregated_prediction'] = aggregated_probs
            result['model_contributions'] = model_contributions

        return result

    def evaluate(self, test_df):
        """
        Evaluate the predictor on a test set

        Args:
            test_df: DataFrame with test data

        Returns:
            Dictionary with evaluation metrics
        """
        print("\n=== Evaluating Stacked Predictor ===")

        # Initialize counters
        correct = 0
        total = 0
        predictions = []

        # Process each test issue
        for _, row in test_df.iterrows():
            # Get issue properties
            issue_text = row.get('combined_text', '')
            issue_labels = row.get('labels', [])
            issue_components = row.get('components', [])
            issue_assignee = row.get('assignee')
            issue_priority = row.get('priority')
            issue_type = row.get('issuetype')
            actual_category = row.get('time_category')

            # Make prediction
            prediction = self.predict(
                issue_text,
                issue_labels=issue_labels,
                issue_components=issue_components,
                issue_assignee=issue_assignee,
                issue_priority=issue_priority,
                issue_type=issue_type
            )

            # Get most likely category
            probs = prediction['aggregated_prediction']
            predicted_category = max(probs, key=probs.get) if probs else None

            # Skip if no prediction could be made
            if predicted_category is None:
                continue

            # Store prediction
            pred_record = {
                'issue_key': row.get('issue_key', ''),
                'actual_category': actual_category,
                'predicted_category': predicted_category,
                'correct': predicted_category == actual_category
            }

            # Add probabilities
            for category, prob in probs.items():
                pred_record[f'prob_{category}'] = prob

            predictions.append(pred_record)

            # Update counters
            if predicted_category == actual_category:
                correct += 1
            total += 1

        # Calculate overall accuracy
        accuracy = correct / total if total > 0 else 0

        print(f"Overall stacked model accuracy: {accuracy:.4f}")

        # Create confusion matrix
        y_true = [p['actual_category'] for p in predictions]
        y_pred = [p['predicted_category'] for p in predictions]

        # Convert predictions to DataFrame
        predictions_df = pd.DataFrame(predictions)

        # Save predictions
        if self.output_dir:
            predictions_df.to_csv(f"{self.output_dir}/stacked_predictions.csv", index=False)

        # Calculate confusion matrix
        conf_matrix = pd.crosstab(
            pd.Series(y_true, name='Actual'),
            pd.Series(y_pred, name='Predicted')
        )

        print("\nConfusion Matrix:")
        print(conf_matrix)

        # Generate classification report
        class_report = classification_report(y_true, y_pred)
        print("\nClassification Report:")
        print(class_report)

        # Save classification report
        if self.output_dir:
            with open(f"{self.output_dir}/stacked_classification_report.txt", 'w') as f:
                f.write(f"Accuracy: {accuracy:.4f}\n\n")
                f.write("Confusion Matrix:\n")
                f.write(str(conf_matrix))
                f.write("\n\nClassification Report:\n")
                f.write(class_report)

        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'predictions': predictions_df
        }