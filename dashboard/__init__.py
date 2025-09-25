"""
Dashboard Module
================

Web dashboard and API for railway traffic management system.
"""

from .app import create_app
from .api import trains, optimizer, health

__all__ = ['create_app', 'trains', 'optimizer', 'health']
