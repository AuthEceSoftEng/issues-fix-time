"""
Utility modules
"""
from .project_stats import count_project_issues
from .paths import (
    ensure_dir, 
    get_project_root, 
    get_data_dir, 
    get_output_dir, 
    setup_output_subdirs
)