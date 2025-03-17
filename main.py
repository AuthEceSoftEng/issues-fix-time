"""
Jira Issue Resolution Time Predictor: Main Script

This script implements the full pipeline for analyzing and predicting 
Jira issue resolution times using a stacked machine learning approach.
"""

import os
import argparse
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import configuration
from config import *

# Import project modules
from data.data_retrieval import get_project_data
from data.data_processing import categorize_resolution_times, prepare_text_data
from data.data_analysis import analyze_distributions

from visualization.resolution_charts import analyze_resolution_times
from visualization.heatmaps import create_heatmap_visualizations

from models.text_models import train_tfidf_models
from models.bert_models import train_bert_models
from models.topic_models import train_topic_models
from models.stacked_predictor import ResolutionTimeStackedPredictor

from utils.project_stats import count_project_issues

# Suppress warnings
warnings.filterwarnings("ignore")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Jira Resolution Time Predictor')
    
    parser.add_argument('--project', type=str, required=True,
                        help='Jira project name to analyze')
    
    parser.add_argument('--mongo-uri', type=str, default=DEFAULT_MONGO_URI,
                        help='MongoDB connection URI')
    
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                        help='Directory to save results')
    
    parser.add_argument('--max-days', type=int, default=MAX_RESOLUTION_DAYS,
                        help='Maximum resolution time in days')
    
    parser.add_argument('--list-projects', action='store_true',
                        help='List available projects and exit')
    
    return parser.parse_args()

def main():
    """
    Main function to run the full Jira resolution time analysis and prediction pipeline
    with stacked machine learning
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Set output directory
    main_output_dir = args.output_dir
    models_dir = f"{main_output_dir}/models"
    visualizations_dir = f"{main_output_dir}/visualizations"
    distributions_dir = f"{main_output_dir}/distributions"
    predictions_dir = f"{main_output_dir}/predictions"

    # Create output directories
    for directory in [main_output_dir, models_dir, visualizations_dir, distributions_dir, predictions_dir]:
        os.makedirs(directory, exist_ok=True)
        
    # If --list-projects flag is used, show available projects and exit
    if args.list_projects:
        projects_df = count_project_issues(args.mongo_uri, DEFAULT_MONGO_DB)
        if projects_df is not None:
            print("\n=== Available Projects ===")
            print(projects_df.head(20))
            print(f"\nTotal projects: {len(projects_df)}")
        return

    print(f"Starting Jira resolution time analysis for project: {args.project}")
    print(f"Output directory: {main_output_dir}")

    # 1. Get project data
    df = get_project_data(
        args.project, 
        args.mongo_uri, 
        max_resolution_days=args.max_days
    )

    if df.empty or len(df) < 10:
        print("Not enough data to continue. Exiting.")
        return

    # Basic statistics
    print("\nResolution time statistics:")
    print(f"Mean: {df['resolution_hours'].mean():.2f} hours")
    print(f"Median: {df['resolution_hours'].median():.2f} hours")
    print(f"Min: {df['resolution_hours'].min():.2f} hours")
    print(f"Max: {df['resolution_hours'].max():.2f} hours")

    # 2. Categorize resolution times
    df = categorize_resolution_times(df)

    # 3. Create visualizations
    analyze_resolution_times(df, visualizations_dir)

    # 4. Create heatmap visualizations for labels, components, and assignees
    heatmap_dir = create_heatmap_visualizations(df, visualizations_dir)

    # 5. Prepare text data
    df = prepare_text_data(df)

    # 6. Split into train, validation and test sets
    # First split into temp (train+validation) and test
    temp_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df['time_category']
    )

    # Then split temp into train and validation
    train_df, validation_df = train_test_split(
        temp_df,
        test_size=VALID_SIZE,  # 0.2 * 0.8 = 0.16 of original data
        random_state=RANDOM_STATE,
        stratify=temp_df['time_category']
    )

    print(f"\nSplit data into {len(train_df)} training samples, {len(validation_df)} validation samples, and {len(test_df)} test samples")

    # Save the split datasets
    train_df.to_csv(f"{main_output_dir}/train_data.csv", index=False)
    validation_df.to_csv(f"{main_output_dir}/validation_data.csv", index=False)
    test_df.to_csv(f"{main_output_dir}/test_data.csv", index=False)

    # 7. Analyze distributions
    distribution_tables = analyze_distributions(train_df, validation_df, distributions_dir)

    # 8. Train TF-IDF based models on training set, evaluate on validation set
    tfidf_models = train_tfidf_models(train_df, validation_df, models_dir)

    # 9. Train BERT-based models (if available) on training set, evaluate on validation set
    bert_models = train_bert_models(train_df, validation_df, models_dir)

    # 10. Train topic models (if available) on training set
    topic_models = train_topic_models(train_df, validation_df, models_dir)

    # 11. Create stacked predictor
    print("\n=== Setting up Stacked Predictor ===")
    stacked_predictor = ResolutionTimeStackedPredictor(output_dir=predictions_dir)

    # Select best text model (BERT or TF-IDF)
    best_text_model_path = None
    if bert_models:
        # Get best BERT model
        best_bert_name = max(bert_models, key=lambda k: bert_models[k][1])
        model_type = best_bert_name.split(' + ')[1].lower().replace(' ', '_')
        best_text_model_path = f"{models_dir}/bert_{model_type}_model.pkl"
        print(f"Using best BERT model: {best_bert_name}")
    else:
        # Get best TF-IDF model
        best_tfidf_name = max(tfidf_models, key=lambda k: tfidf_models[k][1])
        if "Ensemble" in best_tfidf_name:
            best_text_model_path = f"{models_dir}/tfidf_ensemble_model.pkl"
        else:
            model_type = best_tfidf_name.split(' + ')[1].lower().replace(' ', '_')
            best_text_model_path = f"{models_dir}/tfidf_{model_type}_model.pkl"
        print(f"Using best TF-IDF model: {best_tfidf_name}")

    # Load models and distributions
    topic_model_path = f"{models_dir}/topic_model.pkl" if topic_models else None
    topic_resolution_path = f"{models_dir}/topic_resolution_percentages.csv" if topic_models else None

    stacked_predictor.load_models(
        text_model_path=best_text_model_path,
        topic_model_path=topic_model_path,
        topic_resolution_path=topic_resolution_path
    )

    stacked_predictor.load_distributions(
        label_dist_path=f"{distributions_dir}/label_resolution_percentages.csv",
        component_dist_path=f"{distributions_dir}/component_resolution_percentages.csv",
        assignee_dist_path=f"{distributions_dir}/assignee_resolution_percentages.csv",
        priority_dist_path=f"{distributions_dir}/priority_resolution_percentages.csv"
    )

    # 12. Train the meta-learner on validation set
    meta_accuracy = stacked_predictor.train_meta_learner(validation_df)

    # 13. Evaluate the stacked model on test set
    evaluation_results = stacked_predictor.evaluate(test_df)
    aggregated_accuracy = evaluation_results['accuracy']

    # 14. Create accuracy comparison table
    print("\n=== Model Accuracy Comparison ===")

    accuracy_data = []

    # Add TF-IDF models
    if tfidf_models:
        for model_name, (_, accuracy) in tfidf_models.items():
            accuracy_data.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'Type': 'Text-based'
            })

    # Add BERT models
    if bert_models:
        for model_name, (_, accuracy) in bert_models.items():
            accuracy_data.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'Type': 'Text-based'
            })

    # Add Topic model
    if topic_models and 'accuracy' in topic_models:
        accuracy_data.append({
            'Model': 'Topic Model',
            'Accuracy': topic_models['accuracy'],
            'Type': 'Topic-based'
        })

    # Add meta-learner (validation accuracy)
    accuracy_data.append({
        'Model': 'Meta-Learner (Validation)',
        'Accuracy': meta_accuracy,
        'Type': 'Stacked ML'
    })

    # Add stacked model (test accuracy)
    accuracy_data.append({
        'Model': 'Stacked ML (Test)',
        'Accuracy': aggregated_accuracy,
        'Type': 'Stacked ML'
    })

    # Create DataFrame and sort by accuracy
    accuracy_df = pd.DataFrame(accuracy_data)
    accuracy_df = accuracy_df.sort_values('Accuracy', ascending=False)

    # Display the accuracy comparison table
    print("\nAccuracy Comparison Table:")
    print(accuracy_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Save the accuracy comparison table
    accuracy_df.to_csv(f"{main_output_dir}/model_accuracy_comparison.csv", index=False)

    # Create accuracy visualization
    plt.figure(figsize=(12, 6))
    bar_plot = sns.barplot(
        x='Model',
        y='Accuracy',
        hue='Type',
        data=accuracy_df,
        palette='viridis'
    )
    plt.title('Model Accuracy Comparison', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{visualizations_dir}/model_accuracy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 15. Calculate and display breakdown of accuracy by category
    print("\n=== Accuracy Breakdown by Category ===")

    # Get prediction results from the evaluation
    y_true = evaluation_results['predictions']['actual_category']
    y_pred = evaluation_results['predictions']['predicted_category']

    # Calculate accuracy for each category
    category_accuracies = {}
    category_counts = {}
    for category in test_df['time_category'].unique():
        # Filter to just this category
        category_indices = y_true == category
        if sum(category_indices) > 0:
            category_acc = accuracy_score(
                y_true[category_indices],
                y_pred[category_indices]
            )
            category_accuracies[category] = category_acc
            category_counts[category] = sum(category_indices)

    # Display category accuracies
    print("\nAccuracy by Resolution Category:")
    for category, acc in sorted(category_accuracies.items()):
        count = category_counts[category]
        print(f"  {category}: {acc:.4f} (n={count})")

    print(f"\nAnalysis complete! All results saved to {main_output_dir}")
    print(f"Visualizations saved to {visualizations_dir}")
    print(f"Heatmaps saved to {heatmap_dir}")
    print(f"Stacked ML model saved to {predictions_dir}/meta_learner_model.pkl")

if __name__ == "__main__":
    main()