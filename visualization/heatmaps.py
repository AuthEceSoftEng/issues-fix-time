"""
Jira Issue Resolution Time Predictor: Heatmap Visualizations
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_heatmap_visualizations(df, output_dir):
    """
    Create heatmap visualizations for labels, components, and assignees
    
    Args:
        df: DataFrame with issue data
        output_dir: Base directory for visualizations
        
    Returns:
        Path to the heatmap directory
    """
    print("\n=== Creating Heatmap Visualizations ===")
    heatmap_dir = f"{output_dir}/heatmaps"
    os.makedirs(heatmap_dir, exist_ok=True)

    # Define a colormap for heatmaps
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # 1. Label Heatmap
    print("Creating label heatmap...")
    try:
        # Get all unique labels
        all_labels = set()
        for labels_list in df['labels']:
            if isinstance(labels_list, list):
                all_labels.update(labels_list)

        # Count occurrences of each label in each resolution category
        label_data = {}
        for label in all_labels:
            label_data[label] = {}
            for category in df['time_category'].unique():
                # Count issues with this label and category
                count = sum(1 for _, row in df.iterrows()
                          if isinstance(row['labels'], list)
                          and label in row['labels']
                          and row['time_category'] == category)
                label_data[label][category] = count

        # Convert to DataFrame
        label_df = pd.DataFrame(label_data).T

        # Fill NaN values with 0
        label_df = label_df.fillna(0)

        # Calculate percentages (normalize by row)
        label_pct = label_df.div(label_df.sum(axis=1), axis=0) * 100

        # Select top 20 labels by total count
        top_labels = label_df.sum(axis=1).sort_values(ascending=False).head(20).index
        label_pct = label_pct.loc[top_labels]

        # Create heatmap
        plt.figure(figsize=(12, 14))
        sns.heatmap(label_pct, annot=True, fmt='.1f', cmap=cmap, linewidths=0.5, center=25)
        plt.title('Resolution Time Categories by Label (%)', fontsize=16)
        plt.ylabel('Label', fontsize=12)
        plt.xlabel('Resolution Category', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{heatmap_dir}/label_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved label heatmap to {heatmap_dir}/label_heatmap.png")
    except Exception as e:
        print(f"Error creating label heatmap: {e}")

    # 2. Component Heatmap
    print("Creating component heatmap...")
    try:
        # Get all unique components
        all_components = set()
        for components_list in df['components']:
            if isinstance(components_list, list):
                all_components.update(components_list)

        # Count occurrences of each component in each resolution category
        component_data = {}
        for component in all_components:
            component_data[component] = {}
            for category in df['time_category'].unique():
                # Count issues with this component and category
                count = sum(1 for _, row in df.iterrows()
                          if isinstance(row['components'], list)
                          and component in row['components']
                          and row['time_category'] == category)
                component_data[component][category] = count

        # Convert to DataFrame
        component_df = pd.DataFrame(component_data).T

        # Fill NaN values with 0
        component_df = component_df.fillna(0)

        # Calculate percentages (normalize by row)
        component_pct = component_df.div(component_df.sum(axis=1), axis=0) * 100

        # Select top 20 components by total count
        top_components = component_df.sum(axis=1).sort_values(ascending=False).head(20).index
        component_pct = component_pct.loc[top_components]

        # Create heatmap
        plt.figure(figsize=(12, 14))
        sns.heatmap(component_pct, annot=True, fmt='.1f', cmap=cmap, linewidths=0.5, center=25)
        plt.title('Resolution Time Categories by Component (%)', fontsize=16)
        plt.ylabel('Component', fontsize=12)
        plt.xlabel('Resolution Category', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{heatmap_dir}/component_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved component heatmap to {heatmap_dir}/component_heatmap.png")
    except Exception as e:
        print(f"Error creating component heatmap: {e}")

    # 3. Assignee Heatmap
    print("Creating assignee heatmap...")
    try:
        # Get assignees with at least 5 issues
        assignee_counts = df['assignee'].value_counts()
        active_assignees = assignee_counts[assignee_counts >= 5].index

        # Filter to active assignees
        assignee_df = df[df['assignee'].isin(active_assignees)]

        # Create crosstab
        assignee_table = pd.crosstab(assignee_df['assignee'], assignee_df['time_category'])

        # Calculate percentages (normalize by row)
        assignee_pct = assignee_table.div(assignee_table.sum(axis=1), axis=0) * 100

        # Select top 20 assignees by total count
        top_assignees = assignee_table.sum(axis=1).sort_values(ascending=False).head(20).index
        assignee_pct = assignee_pct.loc[top_assignees]

        # Create heatmap
        plt.figure(figsize=(12, 14))
        sns.heatmap(assignee_pct, annot=True, fmt='.1f', cmap=cmap, linewidths=0.5, center=25)
        plt.title('Resolution Time Categories by Assignee (%)', fontsize=16)
        plt.ylabel('Assignee', fontsize=12)
        plt.xlabel('Resolution Category', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{heatmap_dir}/assignee_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved assignee heatmap to {heatmap_dir}/assignee_heatmap.png")
    except Exception as e:
        print(f"Error creating assignee heatmap: {e}")

    # 4. Priority Heatmap
    print("Creating priority heatmap...")
    try:
        # Create crosstab
        priority_table = pd.crosstab(df['priority'], df['time_category'])

        # Calculate percentages (normalize by row)
        priority_pct = priority_table.div(priority_table.sum(axis=1), axis=0) * 100

        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(priority_pct, annot=True, fmt='.1f', cmap=cmap, linewidths=0.5, center=25)
        plt.title('Resolution Time Categories by Priority (%)', fontsize=16)
        plt.ylabel('Priority', fontsize=12)
        plt.xlabel('Resolution Category', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{heatmap_dir}/priority_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved priority heatmap to {heatmap_dir}/priority_heatmap.png")
    except Exception as e:
        print(f"Error creating priority heatmap: {e}")

    # 5. Issue Type Heatmap
    print("Creating issue type heatmap...")
    try:
        # Create crosstab
        issuetype_table = pd.crosstab(df['issuetype'], df['time_category'])

        # Calculate percentages (normalize by row)
        issuetype_pct = issuetype_table.div(issuetype_table.sum(axis=1), axis=0) * 100

        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(issuetype_pct, annot=True, fmt='.1f', cmap=cmap, linewidths=0.5, center=25)
        plt.title('Resolution Time Categories by Issue Type (%)', fontsize=16)
        plt.ylabel('Issue Type', fontsize=12)
        plt.xlabel('Resolution Category', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{heatmap_dir}/issuetype_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved issue type heatmap to {heatmap_dir}/issuetype_heatmap.png")
    except Exception as e:
        print(f"Error creating issue type heatmap: {e}")

    return heatmap_dir