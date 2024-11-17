from .dmt_ import DMT_DV
import os
import sys

__all__ = ["DMT"]
__version__ = '0.0.8'

if os.name != 'posix':
    sys.exit("This package can only be installed on Linux systems.")

# try:
#     __version__ = pkg_resources.get_distribution("dmtev-learn").version
# except pkg_resources.DistributionNotFound:
