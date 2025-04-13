import sys
import copy
from .utils.compression_methods.parameters_svd import parameter_svd_write, inverse_parameter_svd_reading, if_reduces_size
from .utils.compression_methods.sparsification import sparse_crs_top_k, to_dense, sparse_matrix, get_not_zero_values
from .utils.compression_methods.fedkd import fedkd_compression
import os

from fedpredict.fedpredict_core import CKA, fedpredict_core, fedpredict_dynamic_core, fedpredict_core_compredict, fedpredict_core_layer_selection

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

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

NDArray = npt.NDArray[Any]
NDArrayInt = npt.NDArray[np.int8]
NDArrayFloat = npt.NDArray[np.float32]
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
                            device="cpu",
                            M: List=[],
                            s: float=1,
                            fc: dict[str,float]=None,
                            il: dict[str,float]=None,
                            dh: dict[str,float]=None,
                            ps: dict[str,float]=None,
                            knowledge_distillation:bool=False,
                            decompress: bool=False,
                            logs: bool=False
                            )-> torch.nn.Module:
    """
        This function includes three versions of FedPredict.

         Version 1: FedPredict https://ieeexplore.ieee.org/abstract/document/10257293 and https://ieeexplore.ieee.org/abstract/document/10713874
             Configuration: this version is the default.
             This method combines global and local parameters, providing both generalization and personalization.
             It also prevents the drop in the accuracy when new/untrained clients are evaluated

         Version 2: FedPredict-Dynamic https://ieeexplore.ieee.org/abstract/document/10621488
             Configuration: set 'dynamic=True' and inform the hyperparameters 'similarity' and 'fraction_of_classes'.
             This method has the same benefits as the v.1.
             However, differently from v.1, in the model combination process, it takes into account the similarity between
             the last and current datasets to weigh the combination and provide support for heterogeneous and
             non-stationary/dynamic data.
         Version 3: MultiFedPredict
            Usage:
        Args:
            local_model: torch.nn.Module, required.
            global_model: list[np.array], required.
            p: list[float], required.
            t: int, required.
                The current round.
            T: int, required.
                The total rounds.
            nt: int, required.
                The number of rounds the client is without training.
            M: list, optional. Default=[]
                The list of indexes of the shared layers.
            s: float, optional. Default=1. Used when Dynamic=True.
                The similarity between the old data (i.e., the one that the local model was previously trained on) and the new
                data.Note that s \in [0, 1].
            fc: dict[str,float], optional. Default=None. Used when Dynamic=True.
                The fraction of classes in the data that generated the global model.
                Example:
                    fc = {'global': 0.5, 'reference': 0.5}
            il: dict[str,float], optional. Default=None.
                The imbalance level in the data that generated the global model.
                Example:
                    il = {'global': 0.5, 'reference': 0.5}
            dh: dict[str,float], optional. Default=None.
                The degree of data homogeneity that generated the global model.
                Example:
                    dh = {'global': 0.5, 'reference': 0.5}
            ps: dict[str,float], optional. Default=None.
                The probability of data shift.
                Example:
                    ps = {'global': 0.5, 'reference': 0.5}
            knowledge_distillation: bool, optional. Default=False
                If the model has knowledge distillation, then set True, to indicate that the global model parameters have
                to be combined with the student model
            decompress: bool, optional. Default=False
                Whether or not to decompress global model parameters in case a previous compression was applied. Only set
                True if using "FedPredict_server" and compressing the shared parameters.
            logs: bool, optional. Default=False
                Whether or not to log the results of the combination process.

        Returns: torch.nn.Module
            The combined model


   """


    try:

        if not _has_torch:
            raise ValueError("Framework 'torch' not found")

        local_model = copy.deepcopy(local_model)
        global_model = copy.deepcopy(global_model)

        # if s is None and fc is None and il is None and dh is None and ps is None:
        #
        #     combined_local_model = fedpredict_client_traditional_torch(local_model=local_model,
        #                                                                global_model=global_model,
        #                                                                t=t,
        #                                                                T=T,
        #                                                                nt=nt,
        #                                                                device=device,
        #                                                                M=M,
        #                                                                knowledge_distillation=knowledge_distillation,
        #                                                                decompress=decompress,
        #                                                                fc=fc,
        #                                                                il=il)

        assert t > 0, f"t must be greater than 0, but you passed {t}"
        assert (T > t and T >= 0), f"T must be greater than t, but you passed t: {t} and T: {T}"
        assert nt >= 0, f"nt must be greater than 0, but you passed {nt}"
        assert (s >= 0 and s <= 1), f"s must be between 0 and 1, but you passed {s}"

        if s == 1:
            version = "Version 1: FedPredict"
        elif s >= 0 and s < 1:
            version = "Version 2: FedPredict-Dynamic"

        if fc is not None and il is not None and dh is not None and ps is not None:

            assert s >= 0 and s <= 1 and fc["global"] >= 0 and fc["global"] <= 1 and fc["reference"] >= 0 and fc["reference"] <= 1 and il[
                "global"] >= 0 and il["global"] <= 1 and ps["global"] >= 0 and ps["global"] <= 1 and ps[
                       "reference"] >= 0 and ps["reference"] <= 1 and dh["global"] >= 0 and dh["global"] <= 1 and dh[
                       "reference"] >= 0 and dh["reference"] <= 1, f"Metrics fc, il, dh, and ps must be between 0 and 1, but you passed {fc}, {il}, {dh}, and {ps}"

            version = "Version 3: MultiFedPredict"

        if logs:
            print(f"Using {version}")

        combined_local_model = fedpredict_client_versions_torch(local_model=local_model,
                                                                global_model=global_model,
                                                                t=t,
                                                                T=T,
                                                                nt=nt,
                                                                M=M,
                                                                s=s,
                                                                fc=fc,
                                                                il=il,
                                                                dh=dh,
                                                                ps=ps,
                                                                knowledge_distillation=knowledge_distillation,
                                                                decompress=decompress,
                                                                logs=logs)

        return combined_local_model.to(device)

    except Exception as e:
        print("FedPredict client torch")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


def fedpredict_client_traditional_torch(local_model: torch.nn.Module,
                                        global_model: Union[torch.nn.Module, List[NDArrays]],
                                        t: int,
                                        T: int,
                                        nt: int,
                                        device: str,
                                        M: List=[],
                                        knowledge_distillation=False,
                                        decompress=False,
                                        fc=1.,
                                        il=1.) -> torch.nn.Module:
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
        # if knowledge_distillation:
        #     model_shape = [i.detach().cpu().numpy().shape for i in local_model.student.parameters()]
        # else:
        #     model_shape = [i.detach().cpu().numpy().shape for i in local_model.parameters()]
        # # if type(global_model) != list:
        # #     global_model = torch_to_list_of_numpy(global_model)
        # if len(M) == 0:
        #     M = [i for i in range(len(torch_to_list_of_numpy(global_model)))]
        number_of_layers_local_model = count_layers(local_model)
        number_of_layers_global_model = count_layers(global_model)
        M = [i for i in range(number_of_layers_local_model)]
        # print("comprimido: ", len(model_shape))
        # global_model = decompress_global_parameters(global_model, model_shape, M, decompress)
        global_model = global_model.to(device)
        local_model = local_model.to(device)
        # print("shape modelo: ", model_shape)
        # print("descomprimido: ", [i.shape for i in global_model])

        if number_of_layers_local_model != number_of_layers_global_model:
            raise Exception("""Lenght of parameters of the global model is {} and is different from the M {}""".format(len(global_model), len(M)))


        local_model = local_model.to(device)
        local_model = fedpredict_combine_models(global_model, local_model, t, T, nt, M, knowledge_distillation, fc, il)
        local_model = local_model.to(device)

        return local_model

    except Exception as e:
        print("FedPredict client traditional torch")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


def fedpredict_client_versions_torch(local_model: torch.nn.Module,
                                     global_model: Union[torch.nn.Module, List[NDArrays]],
                                     t: int,
                                     T: int,
                                     nt: int,
                                     M:List,
                                     s: float,
                                     fc: dict[str, float] = None,
                                     il: dict[str, float] = None,
                                     dh: dict[str, float] = None,
                                     ps: dict[str, float] = None,
                                     knowledge_distillation=False,
                                     decompress=False,
                                     logs=False) -> torch.nn.Module:

    # Using 'torch.load'
    try:
        number_of_layers_local_model = count_layers(local_model)
        number_of_layers_global_model = count_layers(global_model)
        M = [i for i in range(number_of_layers_local_model)]
        # print("comprimido: ", len(model_shape))
        # global_model = decompress_global_parameters(global_model, model_shape, M, decompress)
        global_model = global_model.to("cuda")
        # print("shape modelo: ", model_shape)
        # print("descomprimido: ", [i.shape for i in global_model])

        if number_of_layers_local_model != number_of_layers_global_model:
            raise Exception("""Lenght of parameters of the global model is {} and is different from the M {}""".format(
                len(global_model), len(M)))

        local_model = fedpredict_dynamic_combine_models(global_model, local_model, t, T, nt, M,
                                                        s, fc, il, dh, ps, logs)

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


def fedpredict_combine_models(global_parameters, model, t, T, nt, M, knowledge_distillation, fc, il):
    try:

        local_model_weights, global_model_weight = fedpredict_core(t, T, nt, fc, il)
        # global_model_weight = 1
        # local_model_weights = 0
        count = 0
        if knowledge_distillation:
            for old_param, new_param in zip(model.student.parameters(), global_parameters):
                if count in M:
                    if new_param.shape == old_param.shape:
                        old_param.data = (
                                global_model_weight * new_param.data.clone() + local_model_weights * old_param.data.clone())
                    else:
                        print("Not combined, CNN student: ", new_param.shape, " CNN 3 proto: ", old_param.shape)
                count += 1
        else:
            for new_param, old_param in zip(global_parameters.parameters(), model.parameters()):
                # if count in M:
                if new_param.shape == old_param.shape and count in M:
                    old_param.data = (
                            global_model_weight * new_param.data.clone() + local_model_weights * old_param.data.clone())
                else:
                    # print("Not combined, CNN student: ", new_param.shape, " CNN 3 proto: ", old_param.shape)
                    pass
                count += 1

        # print("count: ", count)

        return model

    except Exception as e:
        print("FedPredict combine models")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


def fedpredict_dynamic_combine_models(global_parameters, model, t, T, nt, M, s, fc, il, dh, ps, logs=False):
    try:

        local_model_weights, global_model_weight = fedpredict_dynamic_core(t, T, nt, s, fc, il, dh, ps, logs)
        count = 0
        for new_param, old_param in zip(global_parameters.parameters(), model.parameters()):
            if count in M:
                if new_param.shape == old_param.shape:
                    old_param.data = (
                            global_model_weight * new_param.data.clone() + local_model_weights * old_param.data.clone())
                else:
                    raise ValueError(f"Layer {count} has different shapes: {new_param.shape} and {old_param.shape}")
            count += 1

        return model

    except Exception as e:
        print("FedPredict dynamic combine models")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

def count_layers(model):
    contador = 0
    for nome, modulo in model.named_modules():
        # Contar apenas módulos que são instâncias de camadas com parâmetros (Linear, Conv2d, etc.)
        contador += 1
    return contador
