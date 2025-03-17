"""
Jira Project Issue Counter

This module connects to a MongoDB database containing Jira data,
retrieves all unique project names, and counts how many issues
each project has.
"""

import pymongo
import pandas as pd
from collections import Counter

def count_project_issues(mongo_uri, database_name="jidata"):
    """
    Count the number of issues for each project in the MongoDB database.

    Args:
        mongo_uri: MongoDB connection string
        database_name: Name of the database (default: jidata)

    Returns:
        DataFrame with project names and issue counts
    """
    try:
        # Connect to MongoDB
        print(f"Connecting to MongoDB...")
        client = pymongo.MongoClient(mongo_uri)
        db = client[database_name]

        # Check if connection is successful
        client.server_info()
        print(f"Successfully connected to MongoDB database: {database_name}")

        # Check if the issues collection exists
        if "issues" not in db.list_collection_names():
            print("Error: 'issues' collection not found in the database")
            return None

        # Get the count of issues by project
        print("Counting issues by project...")

        # Using aggregation pipeline (most efficient)
        pipeline = [
            {"$group": {"_id": "$projectname", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]

        project_counts = list(db["issues"].aggregate(pipeline))

        # Convert to DataFrame
        df = pd.DataFrame(project_counts)
        df.columns = ["Project", "Issue Count"]

        # Calculate total issues
        total_issues = df["Issue Count"].sum()
        print(f"Total projects found: {len(df)}")
        print(f"Total issues found: {total_issues}")

        return df

    except pymongo.errors.ConnectionFailure as e:
        print(f"Error connecting to MongoDB: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None