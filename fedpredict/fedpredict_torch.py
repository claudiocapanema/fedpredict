import sys
import copy
from .utils.compression_methods.parameters_svd import parameter_svd_write, inverse_parameter_svd_reading, if_reduces_size
from .utils.compression_methods.sparsification import sparse_crs_top_k, to_dense, sparse_matrix, get_not_zero_values
from .utils.compression_methods.fedkd import fedkd_compression
import os

from fedpredict.fedpredict_core import fedpredict_dynamic_core

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

import logging
import numpy as np
import numpy.typing as npt

NDArray = npt.NDArray[Any]
NDArrayInt = npt.NDArray[np.int8]
NDArrayFloat = npt.NDArray[np.float32]
NDArrays = List[NDArray]


class CustomFormatter(logging.Formatter):
    """Logging colored formatter, adapted from https://stackoverflow.com/a/56944256/3638629"""

    grey = '\x1b[38;21m'
    blue = '\x1b[38;5;39m'
    yellow = '\x1b[38;5;226m'
    red = '\x1b[38;5;196m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

import datetime

# Create custom logger logging all five levels
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Define format for logs
fmt = '%(asctime)s | %(levelname)8s | %(message)s'

# Create stdout handler for logging to the console (logs all five levels)
stdout_handler = logging.StreamHandler()
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(CustomFormatter(fmt))

# Create file handler for logging to a file (logs all five levels)
today = datetime.date.today()

# Add both handlers to the logger
logger.addHandler(stdout_handler)


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
        logger.critical("Method: fedpredict_client_weight_predictions_torch")
        logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def fedpredict_client_torch(local_model: torch.nn.Module,
                            global_model: Union[torch.nn.Module, List[NDArrays]],
                            t: int,
                            T: int,
                            nt: int,
                            device="cpu",
                            s: float=1,
                            fc: dict[str,float]=None,
                            il: dict[str,float]=None,
                            dh: dict[str,float]=None,
                            ps: dict[str,float]=None,
                            knowledge_distillation:bool=False,
                            global_model_original_shape: list[tuple]=None,
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
            global_model: Union[torch.nn.Module, List[NDArrays]], required.
                It is ``torch.nn.Module" when no compression was applied, while it is ``List[NDArrays]" when compression was applied.
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
            global_model_original_shape: list[tuple]
                The original shape of the global model (without compression). Used when compŕession is applied by ``fedpredict_server``.
            logs: bool, optional. Default=False
                Whether or not to log the results of the combination process.

        Returns: torch.nn.Module
            The combined model


   """


    try:

        if type(local_model).__base__ != torch.nn.Module:
            raise TypeError(f"local_model must be of type torch.nn.Module but got {type(local_model).__base__}")
        if type(local_model).__base__ != torch.nn.Module or type(global_model) == List:
            raise TypeError(f"global_model must be of type torch.nn.Module or List but got {type(global_model)}")
        if type(t) is not int:
            raise TypeError(f"t must be an int but got {type(t)}")
        if type(T) is not int:
            raise TypeError(f"T must be an int but got {type(T)}")
        if type(nt) is not int:
            raise TypeError(f"n must be an int but got {type(nt)}")
        if type(s) is not float and s != 1:
            raise TypeError(f"s must be a float but got {type(s)}")
        if type(fc) not in [dict, type(None)]:
            raise TypeError(f"fc must be a dict but got {type(fc)}")
        if type(il) not in [dict, type(None)]:
            raise TypeError(f"il must be a dict but got {type(il)}")
        if type(dh) not in [dict, type(None)]:
            raise TypeError(f"dh must be a dict but got {type(dh)}")
        if type(ps) not in [dict, type(None)]:
            raise TypeError(f"ps must be a dict but got {type(ps)}")
        if type(logs) is not bool:
            raise TypeError(f"logs must be a bool but got {type(logs)}")

        if not _has_torch:
            raise ImportError("Framework 'torch' not found")
        local_model = copy.deepcopy(local_model).to(device)
        if isinstance(global_model, torch.nn.Module):
            global_model = copy.deepcopy(global_model).to(device)
        elif type(global_model) == list and type(global_model_original_shape) == list:
            assert len(global_model_original_shape) > 0, "original_global_model_shape must not be empty"
            # logger.info(f"glooal model recebido comprimido {len([i.shape for i in global_model])}")
            global_model = decompress_global_parameters(global_model, global_model_original_shape, local_model).to(device)
            # logger.info(f"descomprimido {[i.shape for i in global_model.parameters()]} \n original {global_model_original_shape} o")


        assert t >= 0, f"t must be greater or equal than 0, but you passed {t}"
        assert (T >= t and T >= 0), f"T must be greater than t, but you passed t: {t} and T: {T}"
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
            logger.info(f"Using {version}")

        combined_local_model = fedpredict_client_versions_torch(local_model=local_model,
                                                                global_model=global_model,
                                                                t=t,
                                                                T=T,
                                                                nt=nt,
                                                                s=s,
                                                                fc=fc,
                                                                il=il,
                                                                dh=dh,
                                                                ps=ps,
                                                                knowledge_distillation=knowledge_distillation,
                                                                logs=logs)

        return combined_local_model.to(device)

    except Exception as e:
        logger.critical("Method: fedpredict_client_torch")
        logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


def fedpredict_client_versions_torch(local_model: torch.nn.Module,
                                     global_model: torch.nn.Module,
                                     t: int,
                                     T: int,
                                     nt: int,
                                     s: float,
                                     fc: dict[str, float] = None,
                                     il: dict[str, float] = None,
                                     dh: dict[str, float] = None,
                                     ps: dict[str, float] = None,
                                     knowledge_distillation=False,
                                     logs: bool = False) -> torch.nn.Module:

    # Using 'torch.load'
    try:
        number_of_layers_local_model, M_local = count_layers(local_model)
        number_of_layers_global_model, M_global = count_layers(global_model)
        M = M_global

        # if number_of_layers_local_model != number_of_layers_global_model:
        #     raise Exception("""Lenght of parameters of the global model is {} and is different from the M {}""".format(
        #         number_of_layers_global_model, number_of_layers_local_model))

        local_model = fedpredict_dynamic_combine_models(global_model, local_model, t, T, nt, M_global,
                                                        s, fc, il, dh, ps, logs)

        return local_model

    except Exception as e:
        logger.critical("Method: fedpredict_client_versions_torch")
        logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def torch_to_list_of_numpy(model: torch.nn.Module):
    try:
        parameters = [i.detach().cpu().numpy() for i in model.parameters()]
        return parameters
    except Exception as e:
        logger.critical("Method: torch_to_list_of_numpy")
        logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


def decompress_global_parameters(compressed_global_model_parameters: List[NDArrays], global_model_original_shape: List[Tuple], model_base: torch.nn.Module):
    try:
        if len(compressed_global_model_parameters) == 0 or len(compressed_global_model_parameters) <= len(global_model_original_shape):
            is_layer_selection = True
        else:
            is_layer_selection = False
        for i in range(min([len(compressed_global_model_parameters), len(global_model_original_shape)])):
            compressed_shape = compressed_global_model_parameters[i].shape
            original_shape = global_model_original_shape[i]
            # logger.info(f"comparar layer {i} compressed {compressed_shape} original {original_shape}")
            if compressed_shape != original_shape:
                is_layer_selection = False
                break

        # logger.info(f"is layer selection {is_layer_selection}")
        if len(compressed_global_model_parameters) > 0 and not is_layer_selection:
            # logger.info("vai descomprimir")
            decompressed_gradients = inverse_parameter_svd_reading(compressed_global_model_parameters, global_model_original_shape)
            parameters = [torch.Tensor(i.tolist()) for i in decompressed_gradients]
        else:
            parameters = [torch.Tensor(i.tolist()) for i in compressed_global_model_parameters]

        model_base = copy.deepcopy(model_base)
        size = len(parameters)
        count = 0
        for new_param, old_param in zip(parameters, model_base.parameters()):
            old_param.data = new_param.data.clone()
            count += 1
            if count == size:
                break

        return model_base

    except Exception as e:
        logger.critical("Method: decompress_global_parameters")
        logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


def fedpredict_dynamic_combine_models(global_parameters, model, t, T, nt, M, s, fc, il, dh, ps, logs=False):
    try:

        local_model_weights, global_model_weight = fedpredict_dynamic_core(t, T, nt, s, fc, il, dh, ps, logs)
        count = 0
        # logger.info(f"m: {M} layers {len([i for i in model.parameters()])}")
        for new_param, old_param in zip(global_parameters.parameters(), model.parameters()):
            if count in M:
                if new_param.shape == old_param.shape:
                    old_param.data = (
                            global_model_weight * new_param.data.clone() + local_model_weights * old_param.data.clone())
                else:
                    raise ValueError(f"Layer {count} has different shapes: global model {new_param.shape} and local model {old_param.shape}")
            count += 1

        return model

    except Exception as e:
        logger.critical("Method: fedpredict_dynamic_combine_models")
        logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def count_layers(model):
    count = 0
    M = []
    for i, param in enumerate(model.parameters()):
        # Contar apenas módulos que são instâncias de camadas com parâmetros (Linear, Conv2d, etc.)
        if param.shape[-1] > 0:
            count += 1
            M.append(i)
    return count, M
