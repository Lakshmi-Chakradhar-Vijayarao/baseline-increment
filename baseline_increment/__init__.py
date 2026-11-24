"""baseline-increment: does a detector's score add anything beyond cheap baselines?"""
from .core import audit, auc
from .checks import check

__all__ = ["audit", "auc", "check", "__version__"]
from .core import VERSION as __version__
