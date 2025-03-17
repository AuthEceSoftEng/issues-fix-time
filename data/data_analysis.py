"""
Jira Issue Resolution Time Predictor: Distribution Analysis Functions
"""

import pandas as pd
import numpy as np
import os
from collections import defaultdict, Counter

def analyze_distributions(train_df, test_df, output_dir):
    """
    Analyze resolution time distributions by component, label, priority, and assignee
    
    Args:
        train_df: Training data DataFrame
        test_df: Test data DataFrame
        output_dir: Directory to save distribution tables
        
    Returns:
        Dictionary with all distribution tables
    """
    print("\n=== Analyzing Resolution Time Distributions ===")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Combine train and test for distribution analysis
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    distribution_tables = {}

    # 1. Analyze components
    print("\nAnalyzing components...")

    # Extract all components (they are in lists)
    component_stats = defaultdict(lambda: defaultdict(int))
    component_totals = Counter()

    # Count occurrences of each component in each resolution category
    for _, row in full_df.iterrows():
        time_category = row['time_category']
        if isinstance(row['components'], list):
            for component in row['components']:
                component_stats[component][time_category] += 1
                component_totals[component] += 1

    # Convert to DataFrame
    component_df = pd.DataFrame(component_stats).T

    # Fill NaN values with 0
    for category in full_df['time_category'].unique():
        if category not in component_df.columns:
            component_df[category] = 0

    # Add total counts
    component_df['Total'] = pd.Series(component_totals)

    # Sort by total count descending
    component_df = component_df.sort_values('Total', ascending=False)

    # Calculate percentages
    component_pct_df = component_df.copy()
    for category in full_df['time_category'].unique():
        if category in component_pct_df.columns:
            component_pct_df[category] = (component_pct_df[category] / component_pct_df['Total'] * 100).round(1)

    # Save tables
    component_df.to_csv(f"{output_dir}/component_resolution_counts.csv")
    component_pct_df.to_csv(f"{output_dir}/component_resolution_percentages.csv")

    # Store for return
    distribution_tables['component_counts'] = component_df
    distribution_tables['component_percentages'] = component_pct_df

    # 2. Analyze labels
    print("Analyzing labels...")

    # Extract all labels (they are in lists)
    label_stats = defaultdict(lambda: defaultdict(int))
    label_totals = Counter()

    # Count occurrences of each label in each resolution category
    for _, row in full_df.iterrows():
        time_category = row['time_category']
        if isinstance(row['labels'], list):
            for label in row['labels']:
                label_stats[label][time_category] += 1
                label_totals[label] += 1

    # Convert to DataFrame
    label_df = pd.DataFrame(label_stats).T

    # Fill NaN values with 0
    for category in full_df['time_category'].unique():
        if category not in label_df.columns:
            label_df[category] = 0

    # Add total counts
    label_df['Total'] = pd.Series(label_totals)

    # Sort by total count descending
    label_df = label_df.sort_values('Total', ascending=False)

    # Calculate percentages
    label_pct_df = label_df.copy()
    for category in full_df['time_category'].unique():
        if category in label_pct_df.columns:
            label_pct_df[category] = (label_pct_df[category] / label_pct_df['Total'] * 100).round(1)

    # Save tables
    label_df.to_csv(f"{output_dir}/label_resolution_counts.csv")
    label_pct_df.to_csv(f"{output_dir}/label_resolution_percentages.csv")

    # Store for return
    distribution_tables['label_counts'] = label_df
    distribution_tables['label_percentages'] = label_pct_df

    # 3. Analyze priority
    print("Analyzing priority levels...")

    # Create distribution by priority
    priority_counts = pd.crosstab(full_df['priority'], full_df['time_category'])

    # Add total counts
    priority_counts['Total'] = priority_counts.sum(axis=1)

    # Calculate percentages
    priority_pct = priority_counts.copy()
    for column in priority_counts.columns:
        if column != 'Total':
            priority_pct[column] = (priority_counts[column] / priority_counts['Total'] * 100).round(1)

    # Save tables
    priority_counts.to_csv(f"{output_dir}/priority_resolution_counts.csv")
    priority_pct.to_csv(f"{output_dir}/priority_resolution_percentages.csv")

    # Store for return
    distribution_tables['priority_counts'] = priority_counts
    distribution_tables['priority_percentages'] = priority_pct

    # 4. Analyze assignee
    print("Analyzing assignees...")

    # Get assignees with at least 5 issues
    assignee_issue_counts = full_df['assignee'].value_counts()
    active_assignees = assignee_issue_counts[assignee_issue_counts >= 5].index.tolist()

    # Filter to active assignees
    assignee_df = full_df[full_df['assignee'].isin(active_assignees)]

    # Create distribution by assignee
    assignee_counts = pd.crosstab(assignee_df['assignee'], assignee_df['time_category'])

    # Add total counts
    assignee_counts['Total'] = assignee_counts.sum(axis=1)

    # Calculate percentages
    assignee_pct = assignee_counts.copy()
    for column in assignee_counts.columns:
        if column != 'Total':
            assignee_pct[column] = (assignee_counts[column] / assignee_counts['Total'] * 100).round(1)

    # Save tables
    assignee_counts.to_csv(f"{output_dir}/assignee_resolution_counts.csv")
    assignee_pct.to_csv(f"{output_dir}/assignee_resolution_percentages.csv")

    # Store for return
    distribution_tables['assignee_counts'] = assignee_counts
    distribution_tables['assignee_percentages'] = assignee_pct

    print(f"\nAll distribution tables saved to {output_dir}")

    return distribution_tables