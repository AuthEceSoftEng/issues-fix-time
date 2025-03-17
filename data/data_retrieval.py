"""
Jira Issue Resolution Time Predictor: Data Retrieval Functions
"""

import pymongo
import pandas as pd
from datetime import datetime, timedelta
from config import MIN_ASSIGNEE_CONTRIBUTIONS

def get_project_data(project_name, mongo_uri, max_resolution_days=20, min_assignee_contributions=MIN_ASSIGNEE_CONTRIBUTIONS):
    """
    Read data for a specific project and extract In Progress → Resolved transitions
    with resolution time under max_resolution_days, filtering for issues from assignees
    with at least min_assignee_contributions.

    Args:
        project_name: Name of the Jira project
        mongo_uri: MongoDB connection string
        max_resolution_days: Maximum resolution time in days (default: 20)
        min_assignee_contributions: Minimum number of issues an assignee must have (default: 20)

    Returns:
        DataFrame with transition data from qualified assignees
    """
    # MongoDB connection
    client = pymongo.MongoClient(mongo_uri)
    db = client["jidata"]

    print(f"Retrieving data for project: {project_name}")

    # 1. Get all issues for the project
    issues = []
    for issue in db["issues"].find({"projectname": project_name}):
        # Only include issues with components and labels
        if issue.get("components") and issue.get("labels"):
            issues.append({
                "_id": issue.get("_id"),
                "key": issue.get("key"),
                "assignee": issue.get("assignee"),
                "components": [c.get("name") for c in issue.get("components", [])],
                "labels": issue.get("labels", []),
                "issuetype": issue.get("issuetype", {}).get("name"),
                "priority": issue.get("priority", {}).get("name"),
                "created": issue.get("created"),
                "status": issue.get("status", {}).get("name"),
                "summary": issue.get("summary"),
                "description": issue.get("description")
            })

    print(f"Found {len(issues)} issues with components and labels")

    # 1.5. Filter by assignee contribution counts
    assignee_counts = {}
    for issue in issues:
        assignee = issue.get("assignee")
        if assignee:
            assignee_counts[assignee] = assignee_counts.get(assignee, 0) + 1

    # Get qualified assignees (those with minimum required contributions)
    qualified_assignees = {assignee for assignee, count in assignee_counts.items()
                          if count >= min_assignee_contributions}

    # Filter issues to keep only those with qualified assignees
    qualified_issues = [issue for issue in issues if issue.get("assignee") in qualified_assignees]

    print(f"Filtered to {len(qualified_issues)} issues from {len(qualified_assignees)} assignees with >{min_assignee_contributions} contributions")

    # 2. Get issue IDs for lookup (only from qualified issues)
    issue_ids = [issue["_id"] for issue in qualified_issues]
    issue_dict = {issue["_id"]: issue for issue in qualified_issues}

    # 3. Extract status transitions from events
    transitions = []

    for event in db["events"].find({"projectname": project_name, "issue": {"$in": issue_ids}}):
        issue_id = event["issue"]
        # Skip if we don't have this issue in our filtered list
        if issue_id not in issue_dict:
            continue

        # Look for status change events
        for item in event.get("items", []):
            if item.get("field") == "status":
                from_status = item.get("fromString")
                to_status = item.get("toString")

                # Focus on In Progress -> Resolved transitions
                if from_status == "In Progress" and to_status == "Resolved":
                    transitions.append({
                        "issue_id": issue_id,
                        "issue_key": issue_dict[issue_id].get("key"),
                        "from_status": from_status,
                        "to_status": to_status,
                        "transition_time": event["created"],
                        "assignee": issue_dict[issue_id].get("assignee"),
                        "components": issue_dict[issue_id].get("components"),
                        "labels": issue_dict[issue_id].get("labels"),
                        "issuetype": issue_dict[issue_id].get("issuetype"),
                        "priority": issue_dict[issue_id].get("priority"),
                        "summary": issue_dict[issue_id].get("summary"),
                        "description": issue_dict[issue_id].get("description")
                    })

    print(f"Found {len(transitions)} 'In Progress' to 'Resolved' transitions for qualified assignees")

    # 4. Find when issues entered "In Progress" status
    in_progress_times = {}

    for event in db["events"].find({"projectname": project_name, "issue": {"$in": issue_ids}}):
        issue_id = event["issue"]
        # Skip if we don't have this issue in our filtered list
        if issue_id not in issue_dict:
            continue

        # Look for status change events
        for item in event.get("items", []):
            if item.get("field") == "status" and item.get("toString") == "In Progress":
                # Store the earliest time an issue entered In Progress
                if issue_id not in in_progress_times or event["created"] < in_progress_times[issue_id]:
                    in_progress_times[issue_id] = event["created"]

    # 5. Calculate resolution time and filter by max days
    resolution_data = []

    for transition in transitions:
        issue_id = transition["issue_id"]

        # Skip if we don't know when the issue entered In Progress
        if issue_id not in in_progress_times:
            continue

        # Calculate resolution time
        in_progress_time = in_progress_times[issue_id]
        resolved_time = transition["transition_time"]
        resolution_time = resolved_time - in_progress_time
        resolution_hours = resolution_time.total_seconds() / 3600

        # Skip if resolution time exceeds maximum
        if resolution_hours > (max_resolution_days * 24):
            continue

        # Add to final dataset
        resolution_data.append({
            "issue_id": issue_id,
            "issue_key": transition["issue_key"],
            "assignee": transition["assignee"],
            "components": transition["components"],
            "labels": transition["labels"],
            "issuetype": transition["issuetype"],
            "priority": transition["priority"],
            "in_progress_time": in_progress_time,
            "resolved_time": resolved_time,
            "resolution_hours": resolution_hours,
            "summary": transition["summary"],
            "description": transition["description"]
        })

    # Convert to DataFrame
    df = pd.DataFrame(resolution_data)

    print(f"Final dataset has {len(df)} issues from qualified assignees with resolution time under {max_resolution_days} days")

    return df