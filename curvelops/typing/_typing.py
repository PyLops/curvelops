__all__ = ["InputDimsLike", "FDCTStructLike", "RecursiveListNDArray"]
from typing import List, Sequence, Union

import numpy as np
from numpy.typing import NDArray

InputDimsLike = Union[Sequence[int], NDArray[np.int_]]
FDCTStructLike = List[List[NDArray]]
RecursiveListNDArray = Union[List[NDArray], List["RecursiveListNDArray"]]
