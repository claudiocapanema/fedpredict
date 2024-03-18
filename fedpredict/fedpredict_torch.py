import sys
import copy
import numpy as np
from .utils.compression_methods.parameters_svd import parameter_svd_write, inverse_parameter_svd_reading, if_reduces_size
from .utils.compression_methods.sparsification import sparse_crs_top_k, to_dense, sparse_matrix, get_not_zero_values
from .utils.compression_methods.fedkd import fedkd_compression
import os

from fedpredict.fedpredict_core import CKA, fedpredict_core, fedpredict_dynamic_core, fedpredict_core_compredict, fedpredict_core_layer_selection

import math
import torch

try:
    import flwr
    from flwr.common import (
        EvaluateIns,
        ndarrays_to_parameters,
    )
except ImportError:
    _has_flwr = False
else:
    _has_flwr = True
try:
    import torch
except ImportError:
    _has_torch = False
else:
    _has_torch = True

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

NDArray = npt.NDArray[Any]
NDArrayInt = npt.NDArray[np.int_]
NDArrayFloat = npt.NDArray[np.float_]
NDArrays = List[NDArray]

# ===========================================================================================

# FedPredict client

def fedpredict_client_weight_predictions_torch(output: torch.Tensor, t: int, current_proportion: np.array, similarity: float) -> np.array:
    """
        This function gives more weight to the predominant classes in the current dataset. This function is part of
        FedPredict-Dynamic
    Args:
        output: torch.Tensor, required
            The output of the model after applying 'softmax' activation function.
        t: int, required
            The current round.
        current_proportion:  np.array, required
            The classes proportion in the current training data.
        similarity: float, required
            The similarity between the old data (i.e., the one that the local model was previously trained on) and the new
        data. Note that s \in [0, 1].

    Returns:
        np.array containing the weighted predictions

    """

    try:
        if similarity != 1 and t > 10:
            if _has_torch:
                output = torch.multiply(output, torch.from_numpy(current_proportion * (1 - similarity)))
            else:
                raise ValueError("Framework 'torch' not found")

        return output

    except Exception as e:
        print("FedPredict client weight prediction")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

def fedpredict_client_torch(local_model: torch.nn.Module,
                            global_model: Union[torch.nn.Module, List[NDArrays]],
                            t: int,
                            T: int,
                            nt: int,
                            M: List=[],
                            similarity: float=1,
                            fraction_of_classes: float=-1,
                            filename: str='',
                            current_proportion: List[float]=[],
                            knowledge_distillation=False,
                            decompress=False,
                            dynamic=False) -> torch.nn.Module:
    """
    This function includes two versions of FedPredict.

 Version 1: FedPredict presented in https://ieeexplore.ieee.org/abstract/document/10257293
     Configuration: this version is the default.
     This method combines global and local parameters, providing both generalization and personalization.
     It also prevents the drop in the accuracy when new/untrained clients are evaluated

 Version 2: FedPredict-Dynamic
     Configuration: set 'dynamic=True' and inform the hyperparameters 'similarity' and 'fraction_of_classes'.
     This method has the same benefits as the v.1.
     However, differently from v.1, in the model combination process, it takes into account the similarity between
     the last and current datasets to weigh the combination and provide support for heterogeneous and
     non-stationary/dynamic data.
Args:
    local_model: torch.nn.Module, required.
    global_model: list[np.array], required.
    current_proportion: list[float], required.
    t: int, required.
        The current round.
    T: int, required.
        The total rounds.
    nt: int, required.
        The number of rounds the client is without training.
    M: list, optional. Default=[]
        The list of indexes of the shared layers.
    similarity: float, optional. Default=1. Used when Dynamic=True.
        The similarity between the old data (i.e., the one that the local model was previously trained on) and the new
        data.Note that s \in [0, 1].
    fraction_of_classes: float, optional. Default=-1. Used when Dynamic=True.
        The fraction of classes in the local data.
    filename: str, optional. Default=''
        The filename where the local model is saved.
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

        if not _has_torch:
            raise ValueError("Framework 'torch' not found")

        if not dynamic or len(current_proportion) > 0 or fraction_of_classes==-1:

            return fedpredict_client_traditional_torch(local_model=local_model,
                                                       global_model=global_model,
                                                       t=t,
                                                       T=T,
                                                       nt=nt,
                                                       M=M,
                                                       filename=filename,
                                                       knowledge_distillation=knowledge_distillation,
                                                       decompress=decompress)

        else:

            return fedpredict_client_dynamic_torch(local_model=local_model,
                                                   global_model=global_model,
                                                   current_proportion=current_proportion,
                                                   t=t,
                                                   T=T,
                                                   nt=nt,
                                                   M=M,
                                                   similarity=similarity,
                                                   fraction_of_classes=fraction_of_classes,
                                                   filename=filename,
                                                   knowledge_distillation=knowledge_distillation,
                                                   decompress=decompress)

    except Exception as e:
        print("FedPredict client dynamic")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


def fedpredict_client_traditional_torch(local_model: torch.nn.Module,
                                        global_model: Union[torch.nn.Module,
                                        List[NDArrays]],
                                        t: int,
                                        T: int,
                                        nt: int,
                                        M: List=[],
                                        filename: str='',
                                        knowledge_distillation=False,
                                        decompress=False) -> torch.nn.Module:
    """
        FedPredict v.1 presented in https://ieeexplore.ieee.org/abstract/document/10257293
        This method combines global and local parameters, providing both generalization and personalization.
        It also prevents drop in the accuracy when new/untrained clients are evaluated
        Args:
            local_model: torch.nn.Module, required
            global_model: list[np.array], required
            t: int, required
            T: int, required
            nt: int, required
            M: list, optional. Default
                The list of the indexes of the shared global model layers
            filename: str, optional. Default=''
            knowledge_distillation: bool, optional. Default=False
                If the model has knowledge distillation, then set True, to indicate that the global model parameters have
                to be combined with the student model
            decompress: bool, optional. Default=False
               Whether or not to decompress global model parameters in case a previous compression was applied. Only set
               True if using "FedPredict_server" and compressing the shared parameters.

        Returns: torch.nn.Module
            The combined model

        """
    # Using 'torch.load'
    try:
        if knowledge_distillation:
            model_shape = [i.detach().cpu().numpy().shape for i in local_model.student.parameters()]
        else:
            model_shape = [i.detach().cpu().numpy().shape for i in local_model.parameters()]
        if type(global_model) != list:
            global_model = torch_to_list_of_numpy(global_model)
        if len(M) == 0:
            M = [i for i in range(len(global_model))]
        # print("comprimido: ", len(model_shape))
        global_model = decompress_global_parameters(global_model, model_shape, M, decompress)
        # print("shape modelo: ", model_shape)
        # print("descomprimido: ", [i.shape for i in global_model])

        if len(global_model) != len(M):
            raise Exception("""Lenght of parameters of the global model is {} and is different from the M {}""".format(len(global_model), len(M)))

        if os.path.exists(filename):
            # Load local parameters to 'self.model'
            local_model.load_state_dict(torch.load(filename))
            local_model = fedpredict_combine_models(global_model, local_model, t, T, nt, M)
        else:
            if not knowledge_distillation:
                for old_param, new_param in zip(local_model.parameters(), global_model):
                    old_param.data = new_param.data.clone()
            else:
                local_model.new_client = True
                for old_param, new_param in zip(local_model.student.parameters(), global_model):
                    old_param.data = new_param.data.clone()

        return local_model

    except Exception as e:
        print("FedPredict client traditional")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


def fedpredict_client_dynamic_torch(local_model: torch.nn.Module,
                                    global_model: Union[torch.nn.Module,
                                    List[NDArrays]],
                                    current_proportion: List[float],
                                    t: int,
                                    T: int,
                                    nt: int,
                                    M:List,
                                    similarity: float,
                                    fraction_of_classes: float,
                                    filename: str='',
                                    knowledge_distillation=False,
                                    decompress=False) -> torch.nn.Module:
    """

    Args:
        local_model: torch.nn.Module, required
        global_model: torch.nn.Module or List[NDArrays], required
        current_proportion: List[float], required
        t: int, required
        T: int, required
        nt: int, required
        M: list, required
            The list of the indexes of the shared global model layers
        local_client_information: dict, required
        filename: str, optional. Default=''
        knowledge_distillation: bool, optional. Default=False
            If the model has knowledge distillation, then set True, to indicate that the global model parameters have
            to be combined with the student model
        decompress: bool, optional. Default=False
           Whether or not to decompress global model parameters in case a previous compression was applied. Only set
               True if using "FedPredict_server" and compressing the shared parameters.

    Returns: torch.nn.Module
        The combined model

    """
    # Using 'torch.load'
    try:

        if knowledge_distillation:
            model_shape = [i.detach().cpu().numpy().shape for i in local_model.student.parameters()]
        else:
            model_shape = [i.detach().cpu().numpy().shape for i in local_model.parameters()]
        print("a d: ", type(global_model))
        if type(global_model) != list:
            global_model = torch_to_list_of_numpy(global_model)
        if len(M) == 0:
            M = [i for i in range(len(global_model))]
        global_model = decompress_global_parameters(global_model, model_shape, M, decompress)

        if len(global_model) != len(M):
            raise Exception("""Lenght of parameters of the global model is {} and is different from the M {}""".format(
                len(global_model), len(M)))

        if os.path.exists(filename):
            # Load local parameters to 'self.model'
            # print("existe modelo local")
            local_model.load_state_dict(torch.load(filename))
            local_model = fedpredict_dynamic_combine_models(global_model, local_model, t, T, nt, M,
                                                            similarity, fraction_of_classes)
        else:
            # print("usar modelo global: ", cid)
            if not knowledge_distillation:
                for old_param, new_param in zip(local_model.parameters(), global_model):
                    old_param.data = new_param.data.clone()
            else:
                local_model.new_client = True
                for old_param, new_param in zip(local_model.student.parameters(), global_model):
                    old_param.data = new_param.data.clone()

        return local_model

    except Exception as e:
        print("FedPredict dynamic client")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

def torch_to_list_of_numpy(model: torch.nn.Module):
    try:
        parameters = [i.detach().cpu().numpy() for i in model.parameters()]
        return parameters
    except Exception as e:
        print("Error on FedPredict's method: torch_to_list_of_numpy")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


def decompress_global_parameters(compressed_global_model_parameters, model_shape, M, decompress):
    try:
        if decompress and len(compressed_global_model_parameters) > 0:
            decompressed_gradients = inverse_parameter_svd_reading(compressed_global_model_parameters, model_shape,
                                                                   len(M))
            parameters = [torch.Tensor(i.tolist()) for i in decompressed_gradients]
        else:
            parameters = [torch.Tensor(i.tolist()) for i in compressed_global_model_parameters]

        return parameters

    except Exception as e:
        print("decompress global parameters")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


def fedpredict_combine_models(global_parameters, model, t, T, nt, M):
    try:

        local_model_weights, global_model_weight = fedpredict_core(t, T, nt)
        count = 0
        for new_param, old_param in zip(global_parameters, model.parameters()):
            if count in M:
                if new_param.shape == old_param.shape:
                    old_param.data = (
                            global_model_weight * new_param.data.clone() + local_model_weights * old_param.data.clone())
                else:
                    print("Not combined, CNN student: ", new_param.shape, " CNN 3 proto: ", old_param.shape)
            count += 1

        return model

    except Exception as e:
        print("FedPredict combine models")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


def fedpredict_dynamic_combine_models(global_parameters, model, t, T, nt, M, similarity, fraction_of_classes):
    try:

        local_model_weights, global_model_weight = fedpredict_dynamic_core(t, T, nt, similarity, fraction_of_classes)
        count = 0
        for new_param, old_param in zip(global_parameters, model.parameters()):
            if count in M:
                if new_param.shape == old_param.shape:
                    old_param.data = (
                            global_model_weight * new_param.data.clone() + local_model_weights * old_param.data.clone())
                else:
                    print("Não combinou, CNN student: ", new_param.shape, " CNN 3 proto: ", old_param.shape)
            count += 1

        return model

    except Exception as e:
        print("FedPredict dynamic combine models")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
