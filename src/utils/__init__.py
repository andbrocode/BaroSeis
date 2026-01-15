"""
Utility functions and classes for baroseis package.
"""

from .decimator import Decimator
from .coherence import compute_coherence_windows

__all__ = [
    'Decimator',
    'compute_coherence_windows',
]