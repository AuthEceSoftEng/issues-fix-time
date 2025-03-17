"""
Jira Issue Resolution Time Predictor: Visualization Functions
"""

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import os
import pandas as pd
import numpy as np

def analyze_resolution_times(df, output_dir):
    """
    Create resolution time visualizations with clear color distinctions
    
    Args:
        df: DataFrame with time_category column
        output_dir: Directory to save visualizations
        
    Returns:
        DataFrame with analyzed data
    """
    figsize=(4.4, 2.64)
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Make sure resolution_hours is numeric
    df['resolution_hours'] = pd.to_numeric(df['resolution_hours'])

    # Define a color palette with clearly distinct colors
    category_colors = {
        "Less than 0.5 days": "#3949AB",    # Indigo
        "0.5-2 days": "#00897B",            # Teal
        "2-5 days": "#7CB342",              # Light Green
        "More than 5 days": "#FFA000"       # Amber
    }

    # 1. Create pie chart for overall distribution
    try:
        category_counts = df['time_category'].value_counts().sort_index()
        category_percentages = category_counts / len(df) * 100

        plt.figure(figsize=figsize)
        patches, texts, autotexts = plt.pie(
            category_counts,
            labels=category_counts.index,
            colors=[category_colors[cat] for cat in category_counts.index],
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
            textprops={'fontsize': 12}
        )

        # Customize text appearance
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')

        plt.title('Overall Distribution of Resolution Time Categories', fontsize=16, pad=20)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/resolution_time_pie.pdf", bbox_inches='tight')
        plt.close()
        print(f"✓ Saved pie chart to {output_dir}/resolution_time_pie.pdf")
    except Exception as e:
        print(f"Error creating pie chart: {e}")

    # 2. Categories by Issue Type
    try:
        # Get counts per issue type and category
        issue_counts = df.pivot_table(
            index='issuetype',
            columns='time_category',
            aggfunc='size',
            fill_value=0
        )

        # Calculate percentages
        issue_percentages = issue_counts.div(issue_counts.sum(axis=1), axis=0) * 100

        # Sort by issue type frequency and take top 7
        issue_totals = df['issuetype'].value_counts()
        top_issues = issue_totals.nlargest(7).index
        issue_percentages = issue_percentages.loc[top_issues]

        # Create the horizontal stacked bar chart
        plt.figure(figsize=figsize)

        # Ensure consistent category order
        category_order = ["Less than 0.5 days", "0.5-2 days", "2-5 days", "More than 5 days"]
        # Get available categories that exist in the data
        available_categories = [cat for cat in category_order if cat in issue_percentages.columns]
        available_categories_names = ["<0.5 days", "0.5-2 days", "2-5 days", ">5 days"]

        issue_percentages = issue_percentages[available_categories]

        # Create the plot
        ax = issue_percentages.plot(
            kind='barh',
            stacked=True,
            figsize=figsize,
            color=[category_colors[cat] for cat in available_categories],
            width=0.7
        )
        ax.set_xlim([0, 100])

        #plt.title('Resolution Time Categories by Issue Type', fontsize=16, pad=20)
        plt.xlabel('Percentage', fontsize=12)
        plt.ylabel('Issue Type', fontsize=12)
        plt.tight_layout()

        # Create legend with custom colors
        legend_elements = [
            Patch(facecolor=category_colors[cat], label=available_categories_names[i])
            for i, cat in enumerate(available_categories)
        ]
        plt.legend(
            handles=legend_elements,
            title='Resolution',
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )

        plt.savefig(f"{output_dir}/resolution_by_issuetype.pdf", bbox_inches='tight')
        plt.close()
        print(f"✓ Saved issue type chart to {output_dir}/resolution_by_issuetype.pdf")
    except Exception as e:
        print(f"Error creating issue type chart: {e}")

    # 3. Categories by Priority
    try:
        # Define priority order
        priority_order = ['Blocker', 'Critical', 'Major', 'Minor', 'Trivial']

        # Get counts per priority and category
        priority_counts = df.pivot_table(
            index='priority',
            columns='time_category',
            aggfunc='size',
            fill_value=0
        )

        # Calculate percentages
        priority_percentages = priority_counts.div(priority_counts.sum(axis=1), axis=0) * 100

        # Reorder by standard priority levels
        priority_percentages = priority_percentages.reindex(
            [p for p in priority_order if p in priority_percentages.index]
        )

        # Create the horizontal stacked bar chart
        plt.figure(figsize=figsize)

        # Ensure consistent category order
        category_order = ["Less than 0.5 days", "0.5-2 days", "2-5 days", "More than 5 days"]
        # Get available categories that exist in the data
        available_categories = [cat for cat in category_order if cat in priority_percentages.columns]
        available_categories_names = ["<0.5 days", "0.5-2 days", "2-5 days", ">5 days"]

        priority_percentages = priority_percentages[available_categories]

        # Create the plot
        ax = priority_percentages.plot(
            kind='barh',
            stacked=True,
            figsize=figsize,
            color=[category_colors[cat] for cat in available_categories],
            width=0.7
        )
        ax.set_xlim([0, 100])

        #plt.title('Resolution Time Categories by Priority', fontsize=16, pad=20)
        plt.xlabel('Percentage', fontsize=12)
        plt.ylabel('Priority', fontsize=12)
        plt.tight_layout()

        # Create legend with custom colors
        legend_elements = [
            Patch(facecolor=category_colors[cat], label=available_categories_names[i])
            for i, cat in enumerate(available_categories)
        ]
        plt.legend(
            handles=legend_elements,
            title='Resolution',
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )

        plt.savefig(f"{output_dir}/resolution_by_priority.pdf", bbox_inches='tight')
        plt.close()
        print(f"✓ Saved priority chart to {output_dir}/resolution_by_priority.pdf")
    except Exception as e:
        print(f"Error creating priority chart: {e}")

    return df