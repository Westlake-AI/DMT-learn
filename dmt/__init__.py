from .dmt_ppi import DMT_PPI
import os
import sys

__all__ = ["DMT_PPI"]
__version__ = '0.0.1'

if os.name != 'posix':
    sys.exit("This package can only be installed on Linux systems.")

# try:
#     __version__ = pkg_resources.get_distribution("dmtev-learn").version
# except pkg_resources.DistributionNotFound:
