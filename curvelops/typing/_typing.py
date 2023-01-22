__all__ = ["FDCTStructLike", "RecursiveListNDArray"]
from typing import List, Sequence, Union

import numpy as np
from numpy.typing import NDArray

FDCTStructLike = List[List[NDArray]]
RecursiveListNDArray = Union[List[NDArray], List["RecursiveListNDArray"]]
