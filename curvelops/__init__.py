"""
``curvelops``
=============

Python wrapper for CurveLab's 2D and 3D curvelet transforms.
"""
from .curvelops import *
from .utils import *
from .plot import *
from .typing import *


try:
    from ._version import __version__
except ImportError:
    from datetime import datetime

    __version__ = "0.0.unknown+" + datetime.today().strftime("%Y%m%d")
