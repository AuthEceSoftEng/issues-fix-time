"""
Path handling utilities for the Jira Resolution Time Predictor.
"""

import os
import sys
from pathlib import Path

def ensure_dir(directory):
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to ensure exists
        
    Returns:
        Path to the directory
    """
    os.makedirs(directory, exist_ok=True)
    return directory

def get_project_root():
    """
    Get the project root directory.
    
    Returns:
        Path object for the project root
    """
    # This will work whether run as a module or standalone script
    return Path(__file__).resolve().parent.parent

def get_data_dir():
    """
    Get the data directory path.
    
    Returns:
        Path object for the data directory
    """
    data_dir = get_project_root() / 'data'
    ensure_dir(data_dir)
    return data_dir

def get_output_dir(project_name, base_dir=None):
    """
    Get the output directory for a specific project.
    
    Args:
        project_name: Name of the Jira project
        base_dir: Base directory for outputs (default: project_root/output)
        
    Returns:
        Path object for the project output directory
    """
    if base_dir is None:
        base_dir = get_project_root() / 'output'
    
    project_dir = Path(base_dir) / project_name
    ensure_dir(project_dir)
    return project_dir

def setup_output_subdirs(output_dir):
    """
    Set up standard subdirectories in the output directory.
    
    Args:
        output_dir: Base output directory
        
    Returns:
        Dictionary of paths to subdirectories
    """
    subdirs = {
        'models': ensure_dir(os.path.join(output_dir, 'models')),
        'visualizations': ensure_dir(os.path.join(output_dir, 'visualizations')),
        'distributions': ensure_dir(os.path.join(output_dir, 'distributions')),
        'predictions': ensure_dir(os.path.join(output_dir, 'predictions'))
    }
    
    # Also create a heatmaps subdirectory under visualizations
    subdirs['heatmaps'] = ensure_dir(os.path.join(subdirs['visualizations'], 'heatmaps'))
    
    return subdirs