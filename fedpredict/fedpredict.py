import sys
import copy
import numpy as np
from .utils.compression_methods.parameters_svd import parameter_svd_write, inverse_parameter_svd_reading, if_reduces_size
from .utils.compression_methods.sparsification import sparse_crs_top_k, to_dense, sparse_matrix, get_not_zero_values
from .utils.compression_methods.fedkd import fedkd_compression
import os

import math
import torch

# try:
#     import flwr
#     from flwr.common import (
#         EvaluateIns,
#         ndarrays_to_parameters,
#     )
# except ImportError:
#     _has_flwr = False
# else:
#     _has_flwr = True
try:
    import torch
except ImportError:
    _has_torch = False
else:
    _has_torch = True
try:
    import tensorflow as tf
except ImportError:
    _has_tf = False
else:
    _has_tf = True

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

NDArray = npt.NDArray[Any]
NDArrayInt = npt.NDArray[np.int8]
NDArrayFloat = npt.NDArray[np.float32]
NDArrays = List[NDArray]


# ===========================================================================================

def fedpredict_client(local_model: torch.nn.Module, global_parameters: NDArrays,
                      current_proportion: List[float], t: int, T: int, nt: int, M: List,
                      local_client_information: Dict, filename: str='', knowledge_distillation=False,
                      decompress=False, dynamic=False) -> torch.nn.Module:
    """
        FedPredict v.1 presented in https://ieeexplore.ieee.org/abstract/document/10257293
        This method combines global and local parameters, providing both generalization and personalization.
        It also prevents drop in the accuracy when new/untrained clients are evaluated

        FedPredict v.3 (FedPredict-Dynamic)
        This method combines global and local parameters, providing both generalization and personalization.
        Differently from v.1, it takes into account the similarity between last and current dataset to weight the
        combination and provide support for heterogeneous and non-stationary/dynamic data.
        It also prevents drop in the accuracy when new/untrained clients are evaluated
       Args:
           local_model: torch.nn.Module, required
           global_parameters: list[np.array], required
           current_proportion: list[float], required
           t: int, required
           T: int, required
           nt: int, required
           M: list, required
           local_client_information: dict, required
           filename: str, optional. Default=''
           knowledge_distillation: bool, optional. Default=False
               If the model has knowledge distillation, then set True, to indicate that the global model parameters have
               to be combined with the student model
           decompress: bool, optional. Default=False
               Whether or not to decompress global model parameters in case a previous compression was applied. Only set
               True if using "FedPredict_server" and compressing the shared parameters.
            dynamic: bool, optional. Default=False
                If True, it uses the FedPredict dynamic. If False, it uses the traditional FedPredict.

       Returns: torch.nn.Module
           The combined model

       """

    try:

        if not dynamic:

            return fedpredict_client_traditional(local_model=local_model, global_model=global_parameters, t=t, T=T,
                                                 nt=nt, M=M, filename=filename,
                                                 knowledge_distillation=knowledge_distillation, decompress=decompress)

        else:

            return fedpredict_client_dynamic(local_model=local_model, global_model=global_parameters,
                                             current_proportion=current_proportion,
                                             t=t, T=T, nt=nt, M=M, local_client_information=local_client_information,
                                             filename=filename, knowledge_distillation=knowledge_distillation)

    except Exception as e:
        print("FedPredict client")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
