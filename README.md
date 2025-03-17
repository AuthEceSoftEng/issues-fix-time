# Jira Issue Resolution Time Predictor

This project analyzes and predicts resolution times for Jira issues using machine learning techniques.

## Features

- Retrieves issue data from MongoDB database
- Categorizes resolution times into meaningful buckets
- Analyzes resolution patterns by issue attributes (components, labels, priority, etc.)
- Creates visualizations to understand resolution time distributions
- Implements multiple prediction models:
  - Text-based models (TF-IDF and BERT)
  - Topic models
  - Distribution-based predictions
  - Stacked machine learning approach

## Installation

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Optional: For BERT and topic models, install additional dependencies:
   ```
   pip install sentence-transformers bertopic
   ```

## Usage

```python
python main.py --project PROJECTNAME --mongo-uri "mongodb://user:password@host:port/"
```

## Configuration

Edit `config.py` to customize:
- Output directory
- Resolution time categories
- Model parameters
- Test/train split ratio

## Output

The tool generates:
- Visualizations of resolution time distributions
- Heatmaps showing resolution patterns by different attributes
- Trained ML models for future predictions
- Analysis reports with accuracy metrics
- Comparison of different prediction approaches

