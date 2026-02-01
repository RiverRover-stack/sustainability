"""
API Package

File Responsibility:
    Export FastAPI application and route handlers.

Inputs:
    None (package initialization)

Outputs:
    Public API for REST endpoints

Assumptions:
    - FastAPI and uvicorn installed
    - Models are trained and available

Failure Modes:
    - ImportError if fastapi not installed
"""

from .main import app

__all__ = ['app']
