"""
Dashboard Pages Package

File Responsibility:
    Exports page rendering functions.
"""

from components.pages.ai_assistant import render_ai_assistant_page
from components.pages.data_upload import render_data_upload_page

__all__ = [
    'render_ai_assistant_page',
    'render_data_upload_page'
]
