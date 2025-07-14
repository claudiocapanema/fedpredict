import sys
import logging
import copy
import numpy as np
from .utils.compression_methods.parameters_svd import parameter_svd_write, inverse_parameter_svd_reading, if_reduces_size
from .utils.compression_methods.sparsification import sparse_crs_top_k, to_dense, sparse_matrix, get_not_zero_values
from .utils.compression_methods.fedkd import fedkd_compression
import os
import torch
import pickle

import math

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

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


class CKA(object):
    def __init__(self):
        pass

    def centering(self, K):
        try:
            n = K.shape[0]
            unit = np.ones([n, n])
            I = np.eye(n)
            H = I - unit / n
            return np.dot(np.dot(H, K), H)
        except Exception as e:
            logger.critical("Method: centering")
            logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def rbf(self, X, sigma=None):
        try:
            GX = np.dot(X, X.T)
            KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
            if sigma is None:
                mdist = np.median(KX[KX != 0])
                sigma = math.sqrt(mdist)
            KX *= - 0.5 / (sigma * sigma)
            KX = np.exp(KX)
            return KX
        except Exception as e:
            logger.critical("Method: rbf")
            logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def kernel_HSIC(self, X, Y, sigma):
        try:
            return np.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))
        except Exception as e:
            logger.critical("Method: kernel_HSIC")
            logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def linear_HSIC(self, X, Y):
        try:
            L_X = X @ X.T
            L_Y = Y @ Y.T
            return np.sum(self.centering(L_X) * self.centering(L_Y))
        except Exception as e:
            logger.critical("Method: linear_HSIC")
            logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def linear_CKA(self, X, Y):
        try:
            hsic = self.linear_HSIC(X, Y)
            var1 = np.sqrt(self.linear_HSIC(X, X))
            var2 = np.sqrt(self.linear_HSIC(Y, Y))

            return hsic / (var1 * var2)
        except Exception as e:
            logger.critical("Method: linear_CKA")
            logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def kernel_CKA(self, X, Y, sigma=None):
        try:
            hsic = self.kernel_HSIC(X, Y, sigma)
            var1 = np.sqrt(self.kernel_HSIC(X, X, sigma))
            var2 = np.sqrt(self.kernel_HSIC(Y, Y, sigma))

            return hsic / (var1 * var2)
        except Exception as e:
            logger.critical("Method: kernel_CKA")
            logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

# ===========================================================================================

# FedPredict server

def get_size(parameter):
    try:
        return parameter.nbytes
    except Exception as e:
        logger.critical("Method: get_size")
        logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def compredict(round_of_last_fit, layers_comppression_range, num_rounds, server_round, M, parameter):

    try:

        nt = server_round - round_of_last_fit
        layers_fraction = []
        if round_of_last_fit >= 1:
            n_components_list = []
            for i in range(M):
                # if i % 2 == 0:
                layer = parameter[i]
                if len(layer.shape) >= 2:

                    compression_range = layers_comppression_range[i]
                    if compression_range > 0:
                        n_components = fedpredict_core_compredict(server_round, num_rounds, nt, layer,
                                                                  compression_range)
                    else:
                        n_components = None
                else:
                    n_components = None

                if n_components is None:
                    layers_fraction.append(1)
                else:
                    layers_fraction.append(n_components / layer.shape[-1])

                n_components_list.append(n_components)

            parameter = parameter_svd_write(parameter, n_components_list)

        else:
            new_parameter = []
            for param in parameter:
                new_parameter.append(param)
                new_parameter.append(np.array([]))
                new_parameter.append(np.array([]))

            parameter = new_parameter

            layers_fraction = [1] * len(parameter)

        return parameter, layers_fraction

    except Exception as e:
        logger.critical("Method: compredict")
        logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


def layer_compression_range(model_shape):
    try:

        layers_range = []
        for shape in model_shape:

            layer_range = 0
            if len(shape) >= 2:
                shape = shape[-2:]

                col = shape[1]
                for n_components in range(1, col + 1):
                    if if_reduces_size(shape, n_components):
                        layer_range = n_components
                    else:
                        break

            layers_range.append(layer_range)

        return layers_range

    except Exception as e:
        logger.critical("Method: layer compression range")
        logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def dls(lt, parameters, server_round, nt, num_rounds, df):
    try:
        M = [i for i in range(len(parameters))]
        M_original = copy.deepcopy(M)
        n_layers = len(parameters) / 2

        size_list = []
        for i in range(len(parameters)):
            tamanho = get_size(parameters[i])
            size_list.append(tamanho)
        if lt >= 1:
            # baixo-cima
            M = fedpredict_core_layer_selection(t=server_round, T=num_rounds, nt=nt, n_layers=n_layers, df=df)

            new_parameters = []
            for i in range(len(parameters)):
                if i in M:
                    new_parameters.append(parameters[i])
            parameters = new_parameters

        size_list = []
        for i in range(len(parameters)):
            tamanho = parameters[i].nbytes
            size_list.append(tamanho)
        # logger.info(f"dls: initial layers {len(M_original)} selected layers {len(M)}")
        return parameters, M

    except Exception as e:
        logger.critical("Method: dls")
        logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def per(lt, parameters):
    """
    This method returns all layers parameters except the last two. It is inspired by the FedPer solution.
    Args:
        lt:
        parameters:

    Returns:

    """
    try:
        M = [i for i in range(len(parameters))]
        n_layers = len(parameters) / 2
        size_list = []
        for i in range(len(parameters)):
            tamanho = get_size(parameters[i])
            size_list.append(tamanho)
        if lt >= 1:
            # baixo-cima
            M = [i for i in range(len(parameters))][:-2]
            new_parameters = []
            for i in range(len(parameters)):
                if i in M:
                    new_parameters.append(parameters[i])
            parameters = new_parameters

        size_list = []
        for i in range(len(parameters)):
            tamanho = parameters[i].nbytes
            size_list.append(tamanho)

        return parameters, M

    except Exception as e:
        logger.critical("Method: per")
        logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


def fedpredict_dynamic_core(t, T, nt, s=1, fc=None, il=None, dh=None, ps=None, logs=False):
    try:
        s = float(np.round(s, 1))

        if nt == 0:
            global_model_weight = 0
        elif nt == t:
            global_model_weight = 1

        if nt > 0:
            update_level = 1 / nt
            evolution_level = t / int(T)
            eq1 = (-update_level - evolution_level) * s
            eq2 = round(np.exp(eq1), 6)
            global_model_weight = eq2

        if fc is not None and il is not None and dh is not None and ps is not None:
            if (fc["global"] > fc["reference"] and il["global"] < il["reference"] and dh["global"] > dh[
                "reference"]) and (nt > 0 and ps["global"] < ps["reference"] and dh["global"] > dh["reference"]):
                global_model_weight = 1

        local_model_weights = 1 - global_model_weight

        if logs:
            logger.info(f"round {t} nt {nt} gw {global_model_weight} lw {local_model_weights}")

        return local_model_weights, global_model_weight

    except Exception as e:
        logger.critical("Method: fedpredict core")
        logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


def fedpredict_core_layer_selection(t, T, nt, n_layers, df):
    try:
        # 9
        if nt == 0:
            shared_layers = 0
        else:
            update_level = 1 / nt
            evolution_level = t / T
            eq1 = (-update_level - evolution_level) * (df)  # v8 ótimo
            eq2 = round(np.exp(eq1), 6)
            shared_layers = int(np.ceil(eq2 * n_layers))

        shared_layers = [i for i in range(shared_layers * 2)]

        # logger.info(f"Shared layers: {shared_layers}, round: {t}")

        return shared_layers

    except Exception as e:
        logger.critical("Method: fedpredict_core_server_layer_selection")
        logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


def fedpredict_core_compredict(t, T, nt, layer, compression_range):
    try:
        updated_level = 1 / nt
        columns = layer.shape[-1]
        fraction = compression_range / columns
        evolution_level = t / T

        eq1 = - evolution_level - updated_level  # v5
        fc_l = round(np.exp(eq1), 6)
        n_components = max(round(fc_l * compression_range), 1)
        logger.info(f"fracao: {fc_l}, fraction: {fraction}, componentes: {n_components}, de {compression_range}")

        return n_components

    except Exception as e:
        logger.critical("Method: fedpredict_core_server_compredict")
        logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def layerwise_similarity(global_parameters, clients_parameters, clients_ids,
                         similarity_per_layer_list):
    try:
        num_layers = len(global_parameters)
        num_clients = len(clients_parameters)
        logger.info(f"global: {num_layers}")
        similarity_per_layer = {i: {} for i in clients_ids}
        # interest_layers = [0, 1, int(num_layers/2)-2, int(num_layers/2)-1, num_layers-2, num_layers-1]
        interest_layers = [0, num_layers - 2]
        difference_per_layer = {i: {j: {'min': [], 'max': []} for j in range(num_layers)} for i in clients_ids}
        difference_per_layer_vector = {j: [] for j in range(num_layers)}
        mean_similarity_per_layer = {i: {'mean': 0, 'ci': 0} for i in range(num_layers)}
        mean_difference_per_layer = {i: {'min': 0, 'max': 0} for i in range(num_layers)}

        for client_id in range(num_clients):

            client_parameters = clients_parameters[client_id]
            client_id = clients_ids[client_id]

            for layer_index in range(num_layers):
                client_layer = client_parameters[layer_index]
                global_layer = global_parameters[layer_index]
                if np.ndim(global_layer) == 1:
                    global_layer = np.reshape(global_layer, (len(global_layer), 1))
                if np.ndim(client_layer) == 1:
                    client_layer = np.reshape(client_layer, (len(client_layer), 1))
                # CNN
                if np.ndim(global_layer) == 4:
                    client_similarity = []
                    client_difference = {'min': [], 'max': []}
                    for k in range(len(global_layer)):
                        global_layer_k = global_layer[k][0]
                        client_layer_k = client_layer[k][0]

                        cka = CKA()
                        if layer_index not in interest_layers:
                            similarity = 0
                            difference = np.array([0])
                        else:
                            similarity = cka.linear_CKA(global_layer_k, client_layer_k)
                            difference = global_layer_k - client_layer_k
                            if np.isnan(similarity):
                                if np.sum(global_layer_k) == 0 or np.sum(client_layer_k) == 0:
                                    similarity = 1

                        client_similarity.append(similarity)
                        client_difference['min'].append(abs(difference.min()))
                        client_difference['max'].append(abs(difference.max()))
                        difference_per_layer_vector[layer_index] += np.absolute(difference).flatten().tolist()

                    if layer_index not in similarity_per_layer[client_id]:
                        similarity_per_layer[client_id][layer_index] = []
                        difference_per_layer[client_id][layer_index]['min'] = []
                        difference_per_layer[client_id][layer_index]['max'] = []

                    similarity_per_layer[client_id][layer_index].append(np.mean(client_similarity))
                    difference_per_layer[client_id][layer_index]['min'].append(abs(np.mean(client_difference['min'])))
                    difference_per_layer[client_id][layer_index]['max'].append(abs(np.mean(client_difference['max'])))
                else:

                    if layer_index not in interest_layers:
                        similarity = 0
                        difference = np.array([0])
                    else:
                        cka = CKA()
                        similarity = cka.linear_CKA(global_layer, client_layer)
                        difference = global_layer - client_layer

                    similarity_per_layer[client_id][layer_index] = similarity

                    client_difference = {'min': [], 'max': []}
                    client_difference['min'].append(abs(difference.min()))
                    client_difference['max'].append(abs(difference.max()))
                    difference_per_layer_vector[layer_index] += np.absolute(difference).flatten().tolist()
                    if layer_index not in similarity_per_layer[client_id]:
                        similarity_per_layer[client_id][layer_index] = []
                        difference_per_layer[client_id][layer_index]['min'] = []
                        difference_per_layer[client_id][layer_index]['max'] = []
                    difference_per_layer[client_id][layer_index]['min'].append(abs(np.mean(client_difference['min'])))
                    difference_per_layer[client_id][layer_index]['max'].append(abs(np.mean(client_difference['max'])))

        layers_mean_similarity = []
        for layer_index in interest_layers:
            similarities = []
            min_difference = []
            max_difference = []
            for client_id in clients_ids:
                similarities.append(similarity_per_layer[client_id][layer_index])
                min_difference += difference_per_layer[client_id][layer_index]['min']
                max_difference += difference_per_layer[client_id][layer_index]['max']

            mean = np.mean(similarities)
            similarity_per_layer_list[layer_index].append(mean)
            layers_mean_similarity.append(mean)
            mean_similarity_per_layer[layer_index]['mean'] = mean
            # mean_similarity_per_layer[layer_index]['ci'] = st.norm.interval(alpha=0.95, loc=np.mean(similarities), scale=st.sem(similarities))[1] - np.mean(similarities)
            # mean_difference_per_layer[layer_index]['min'] = np.mean(min_difference)
            # mean_difference_per_layer[layer_index]['max'] = np.mean(max_difference)
            logger.info("""similaridade (camada {}): {}""".format(layer_index, mean_similarity_per_layer[layer_index]))
        for layer in difference_per_layer_vector:
            if np.sum(difference_per_layer_vector[layer]) == 0:
                continue
            # df = pd.DataFrame({'Difference': difference_per_layer_vector[layer], 'x': [i for i in range(len(difference_per_layer_vectore[layer]))]})
            # box_plot(df=df, base_dir='', file_name="""boxplot_difference_layer_{}_round_{}_dataset_{}_alpha_{}""".format(str(layer), str(server_round), dataset, alpha), x_column=None, y_column='Difference', title='Difference between global and local parameters', y_lim=True, y_max=0.065)

        return similarity_per_layer, mean_similarity_per_layer, np.mean(layers_mean_similarity), similarity_per_layer_list
    except Exception as e:
        logger.critical("Method: fedpredict_layerwise_similarity")
        logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def decimals_per_layer(mean_difference_per_layer):
    window = 1
    precisions = {}
    for layer in mean_difference_per_layer:

        n1 = mean_difference_per_layer[layer]['min']
        n2 = mean_difference_per_layer[layer]['max']
        n = min([n1, n2])
        zeros = 0

        if not np.isnan(n):
            n = str(n)
            n = n.split(".")[1]

            for digit in n:

                if digit == "0":
                    zeros += 1
                else:
                    break

            precisions[layer] = zeros + window

        else:
            precisions[layer] = 9

    return precisions


# ===========================================================================================

# FedPredict server

def get_size(parameter):
    try:
        return parameter.nbytes
    except Exception as e:
        logger.critical("Method: get_size")
        logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def fedpredict_server(global_model_parameters: Union[List[np.array], torch.nn.Module],
                      client_evaluate_list: List[Tuple], t: int, T: int, compression: str, df: float = 0,
                      fl_framework=None, **kwargs) -> Union[List[Dict], Tuple[int, Dict]]:
    """
        This function compresses the global model parameters using the specified "compression" technique.

        Args:
            global_model_parameters: list[np.array], required
                List of numpy arrays.
            client_evaluate_list: list[tuple], required
                List of clients' tuples.
            t: int, required
            T: int, required
            compression: str, required
                Compŕession technique. Possible values: "dls", "compredict", "dls_compredict", "fedkd", "spasification", "per".
            df: float, optional. Default=0
                Layers' similarity difference . It is used when compression=="dls".
            fl_framework: str, default=None
                Possible values: None or "flower".

        Returns: Union[List[Dict], Tuple[int, Dict]]
            List[Dict] when fl_framework==None and Tuple[int, Dict] when fl_framework=="flower".

    """
    try:
        assert isinstance(global_model_parameters, torch.nn.Module) or isinstance(global_model_parameters, list), "global model parameters must be of type torch.nn.Module or list"
        assert type(df) == int or type(df) == float, "df must be of type int or float"
        assert df >= 0 and df <= 1, "df must be in range [0,1]"
        if isinstance(global_model_parameters, torch.nn.Module):
            global_model_parameters = [i.detach().cpu().numpy() for i in global_model_parameters.parameters()]

        if compression == "sparsification":
            assert "k" in kwargs, "kwargs must contain 'k' key"
            assert type(kwargs["k"]) == float, "kwargs['k'] must be of type float"
            assert kwargs["k"] >= 0 and kwargs["k"] <= 1, "kwargs['k'] must be in range [0,1]"

        global_model_original_shape = [i.shape for i in global_model_parameters]
        client_evaluate_list_fedpredict = []
        if compression is None or compression not in ["dls", "compredict", "dls_compredict", "fedkd", "sparsification", "per"]:
            logger.info(f"Compression {compression} is not supported or is invalid. Valid values are: 'dls', 'compredict', 'dls_compredict', 'fedkd', 'spasification', 'per'")
            return client_evaluate_list
        if fl_framework == "flwr":
            client_evaluate_list_fedpredict = client_evaluate_list

        # Reuse previously compressed parameters
        previously_reduced_parameters = {}
        logger.debug(f"round t {t} compression technique: {compression}")
        if compression in ["fedkd", "compredict", "dls_compredict"]:
            layers_compression_range = layer_compression_range(global_model_original_shape)
            logger.info(f"compression range: {layers_compression_range}")
            logger.info(f"global model original shape: {global_model_original_shape}")
            # exit()
        fedkd = None
        original_size = sum([i.nbytes for i in global_model_parameters]) * len(client_evaluate_list)
        compressed_size = 0
        # logger.info(f"t: {t} chaves: {previously_reduced_parameters.keys()} tamanho: {len(client_evaluate_list)}")
        for i in range(len(client_evaluate_list)):
            client_tuple = client_evaluate_list[i]
            if fl_framework =='flwr':
                client = client_tuple[0]
                config = client_tuple[1].config
                nt = config['nt']
                lt = config['lt']
            else:
                client = client_tuple['client']
                nt = client_tuple['nt']
                lt = client_tuple['lt']
                config = {}
                config['nt'] = nt
                config['T'] = T
            # logger.info(f"p1 nt: {nt} chaves {previously_reduced_parameters.keys()}")
            if nt != 0 and nt in previously_reduced_parameters:
                process_parameters = False
            else:
                process_parameters = True
            M = [i for i in range(len(global_model_original_shape))]
            parameters_to_send = None
            # When client trained in the current round (nt=0) it is not needed to send parameters (local model is already updated)
            if nt == 0 and compression not in ["fedkd", "sparsification", "", None]:
                if fl_framework is None:
                    config['parameters'] = []
                    client_evaluate_list_fedpredict.append(config)
                elif fl_framework == 'flwr':
                    if _has_flwr:
                        client_evaluate_list_fedpredict[i][1].parameters = ndarrays_to_parameters([])
                        client_evaluate_list_fedpredict[i][1].config = config
                    else:
                        raise ImportError(
                            "Flower is required. Digit: 'pip install fedpredict[flwr]' or 'pip install fedpredict[full]'")
                compressed_size += 0
                continue
            elif compression == 'fedkd':
                if fedkd is None:
                    parameters_to_send = parameters_to_send if parameters_to_send is not None else global_model_parameters
                    parameters_to_send, layers_fraction = fedkd_compression(lt, layers_compression_range, T, t, len(M),
                                                                            parameters_to_send)
                    fedkd = parameters_to_send
                else:
                    # Reuse compressed parameters
                    parameters_to_send = fedkd
                compressed_size += sum([i.nbytes for i in parameters_to_send])
                if fl_framework is None:
                    config['parameters'] = parameters_to_send
                    client_evaluate_list_fedpredict.append(config)
                elif fl_framework =='flwr':
                    if _has_flwr:
                        client_evaluate_list_fedpredict[i][1].parameters = ndarrays_to_parameters(parameters_to_send)
                        client_evaluate_list_fedpredict[i][1].config = config
                    else:
                        raise ImportError(
                            "Flower is required. Digit: 'pip install fedpredict[flwr]' or 'pip install fedpredict[full]'")
                continue
            elif compression == 'sparsification':
                # Keep top k elements
                k = kwargs["k"]
                parameters_to_send, k_values = sparse_crs_top_k([np.abs(i) for i in global_model_parameters], k)
                compressed_size += sum([i.nbytes for i in parameters_to_send])
                if fl_framework is None:
                    config['parameters'] = parameters_to_send
                    config['global_model_original_shape'] = global_model_original_shape
                    client_evaluate_list_fedpredict.append(config)
                elif fl_framework == 'flwr':
                    if _has_flwr:
                        client_evaluate_list_fedpredict[i][1].parameters = ndarrays_to_parameters(parameters_to_send)
                        client_evaluate_list_fedpredict[i][1].config = config
                    else:
                        raise ImportError(
                            "Flower is required. Digit: 'pip install fedpredict[flwr]' or 'pip install fedpredict[full]'")
                continue

            # elif compression in ["", None]:
            #     logger.info(f"compression None")
            #     config['M'] = [i for i in range(len(global_model_parameters))]
            #     config['decompress'] = False
            #     config['layers_fraction'] = 1
            #     if fl_framework is None:
            #         config['parameters'] = global_model_parameters.tolist()
            #         config['global_model_original_shape'] = global_model_original_shape
            #         client_evaluate_list_fedpredict.append(config)
            #     elif fl_framework == 'flwr':
            #         if _has_flwr:
            #             evaluate_ins = EvaluateIns(ndarrays_to_parameters(global_model_parameters), config)
            #             client_evaluate_list_fedpredict[i] = (client, evaluate_ins)
            #         else:
            #             raise ImportError(
            #                 "Flower is required. Digit: 'pip install fedpredict[flwr]' or 'pip install fedpredict[full]'")
            #     compressed_size += sum(i.nbytes for i in parameters_to_send)
            #     continue
            if process_parameters:
                parameters_to_send = global_model_parameters
                if "dls" in compression:
                    parameters_to_send, M = dls(lt, global_model_parameters, t, nt, T, df)
                    logger.info(f"t: {t} T: {T} df: {df} nt: {nt}\ndls - number of shared layers: {len(M)} of {len(global_model_parameters)} ({100*(round(len(M)/len(global_model_parameters), 2))}%). Round {t}")
                elif "per" in compression:
                    M = len([i for i in range(len(global_model_parameters))])
                    logger.info(f"Original number of layers {M}")
                    parameters_to_send, M = per(lt, global_model_parameters)
                    logger.info(f"per - number of shared layers: {len(M)} of {100*len(global_model_parameters)} ({(round(len(M)/len(global_model_parameters), 2))}%). Round {t}")
                layers_fraction = []
                if 'compredict' in compression:
                    logger.info("compredict")
                    parameters_to_send = parameters_to_send if parameters_to_send is not None else global_model_parameters
                    parameters_to_send, layers_fraction = compredict(
                        lt, layers_compression_range,
                        T, t, len(M), parameters_to_send)
                previously_reduced_parameters[nt] = [copy.deepcopy(parameters_to_send), M, layers_fraction]
            else:
                # logger.info(f"Reused parameters for nt == {nt}")
                parameters_to_send, M, layers_fraction = previously_reduced_parameters[nt]
            parameters_to_send = [np.array(i) for i in parameters_to_send]
            compressed_size += sum([i.nbytes for i in parameters_to_send])
            if fl_framework is None:
                config['parameters'] = parameters_to_send
                client_evaluate_list_fedpredict.append(config)
            elif fl_framework == 'flwr':
                if _has_flwr:
                    client_evaluate_list_fedpredict[i][1].parameters = ndarrays_to_parameters(parameters_to_send)
                    client_evaluate_list_fedpredict[i][1].config = config
                else:
                    raise ImportError("Flower is required. Digit: 'pip install fedpredict[flwr]' or 'pip install fedpredict[full]'")
        logger.info(f"Original size {original_size} compressed size {compressed_size}. Economy of {((original_size - compressed_size) * 100) / original_size}. #Round {t}")
        return client_evaluate_list_fedpredict

    except Exception as e:
        logger.critical("Method: fedpredict server")
        logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


# def compredict(round_of_last_fit, layers_comppression_range, num_rounds, server_round, M, parameter):
#
#     try:
#
#         nt = server_round - round_of_last_fit
#         layers_fraction = []
#         if round_of_last_fit >= 1:
#             n_components_list = []
#             for i in range(M):
#                 # if i % 2 == 0:
#                 layer = parameter[i]
#                 if len(layer.shape) >= 2:
#
#                     compression_range = layers_comppression_range[i]
#                     if compression_range > 0:
#                         n_components = fedpredict_core_compredict(server_round, num_rounds, nt, layer,
#                                                                   compression_range)
#                     else:
#                         n_components = None
#                 else:
#                     n_components = None
#
#                 if n_components is None:
#                     layers_fraction.append(1)
#                 else:
#                     layers_fraction.append(n_components / layer.shape[-1])
#
#                 n_components_list.append(n_components)
#
#             parameter = parameter_svd_write(parameter, n_components_list)
#
#
#             logger.info(f"Compredict model: {[i.shape for i in parameter]}")
#
#         else:
#             new_parameter = []
#             for param in parameter:
#                 new_parameter.append(param)
#                 new_parameter.append(np.array([]))
#                 new_parameter.append(np.array([]))
#
#             parameter = new_parameter
#
#             layers_fraction = [1] * len(parameter)
#
#         return parameter, layers_fraction
#
#     except Exception as e:
#         logger.critical("Method: compredict")
#         logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


# def layer_compression_range(model_shape):
#     try:
#
#         layers_range = []
#         for shape in model_shape:
#
#             layer_range = 0
#             if len(shape) >= 2:
#                 shape = shape[-2:]
#
#                 col = shape[1]
#                 for n_components in range(1, col + 1):
#                     if if_reduces_size(shape, n_components):
#                         layer_range = n_components
#                     else:
#                         break
#
#             layers_range.append(layer_range)
#
#         return layers_range
#
#     except Exception as e:
#         logger.critical("Method: layer compression range")
#         logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


# ===========================================================================================

def per(first_round, parameters):
    try:
        M = [i for i in range(len(parameters))]
        n_layers = len(parameters) / 2

        size_list = []
        for i in range(len(parameters)):
            tamanho = get_size(parameters[i])
            size_list.append(tamanho)
        if first_round != -1:
            # baixo-cima
            M = [i for i in range(len(parameters))][:-2]
            new_parameters = []
            for i in range(len(parameters)):
                if i in M:
                    new_parameters.append(parameters[i])
            parameters = new_parameters

        size_list = []
        for i in range(len(parameters)):
            tamanho = parameters[i].nbytes
            size_list.append(tamanho)

        return parameters, M

    except Exception as e:
        logger.critical("Method: per")
        logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))
