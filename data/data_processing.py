"""
Jira Issue Resolution Time Predictor: Data Processing Functions
"""

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from config import RESOLUTION_CATEGORIES, TEST_SIZE, RANDOM_STATE

def categorize_resolution_times(df):
    """
    Categorize resolution times into buckets based on configuration
    
    Args:
        df: DataFrame with resolution_hours column
        
    Returns:
        DataFrame with added time_category column
    """
    # Make sure resolution_hours is numeric
    df['resolution_hours'] = pd.to_numeric(df['resolution_hours'])

    # Create time categories
    category_names = list(RESOLUTION_CATEGORIES.keys())
    category_bins = [0] + [upper for _, upper in RESOLUTION_CATEGORIES.values()]
    
    df['time_category'] = pd.cut(
        df['resolution_hours'],
        bins=category_bins,
        labels=category_names
    )

    print("\nResolution time categories distribution:")
    print(df['time_category'].value_counts())

    return df

def prepare_text_data(df):
    """
    Clean and prepare text data for modeling
    
    Args:
        df: DataFrame with text columns to clean
        
    Returns:
        DataFrame with cleaned text columns added
    """
    # Define text cleaning function
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

        # Return the cleaned text with normalized whitespace
        return " ".join(cleaned_text.split())

    # Clean text columns
    df['clean_summary'] = df['summary'].apply(clean_text)

    if 'description' in df.columns:
        df['clean_description'] = df['description'].apply(clean_text)
        df['combined_text'] = df['clean_summary'] + " " + df['clean_description']
    else:
        df['combined_text'] = df['clean_summary']

    return df

def split_train_test(df):
    """
    Split the dataset into training and test sets
    
    Args:
        df: DataFrame to split
        
    Returns:
        Tuple of (train_df, test_df)
    """
    # Stratify by time_category to ensure balanced classes in both sets
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df['time_category']
    )

    print(f"\nSplit data into {len(train_df)} training samples and {len(test_df)} test samples")

    return train_df, test_df