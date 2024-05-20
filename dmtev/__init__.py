from .dmtev_ import DMTEV

import pkg_resources

__all__ = ["DMTEV"]

try:
    __version__ = pkg_resources.get_distribution("dmtev-learn").version
except pkg_resources.DistributionNotFound:
    __version__ = "0.0.1-dev"
