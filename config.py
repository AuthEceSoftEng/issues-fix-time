"""
Configuration settings for the Jira Resolution Time Predictor
"""

# Output directory for results
OUTPUT_DIR = "jira_analysis_results"

# Machine learning parameters
TEST_SIZE = 0.2  # 20% of data for testing
VALID_SIZE = 0.2  # 20% of training data for validation
RANDOM_STATE = 42

# Resolution time categories (in hours)
RESOLUTION_CATEGORIES = {
    "Less than 0.5 days": (0, 12),
    "0.5-2 days": (12, 48),
    "2-5 days": (48, 120),
    "More than 5 days": (120, float('inf'))
}

# MongoDB settings (default values, can be overridden via CLI)
DEFAULT_MONGO_URI = ''
DEFAULT_MONGO_DB = ''

# Project data filters
MAX_RESOLUTION_DAYS = 30
MIN_ASSIGNEE_CONTRIBUTIONS = 10

# Model parameters
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
RF_N_ESTIMATORS = 100