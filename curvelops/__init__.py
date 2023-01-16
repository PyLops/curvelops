from .curvelops import *


try:
    from ._version import __version__
except ImportError:
    from datetime import datetime

    __version__ = "0.0.unknown+" + datetime.today().strftime("%Y%m%d")
