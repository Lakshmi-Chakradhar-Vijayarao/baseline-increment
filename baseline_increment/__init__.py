"""baseline-increment: does a detector's score add anything beyond cheap baselines?"""
from .core import audit, auc
from .checks import check
from .compare import compare

__all__ = ["audit", "auc", "check", "compare", "__version__"]
from .core import VERSION as __version__
