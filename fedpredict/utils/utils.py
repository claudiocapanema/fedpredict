import sys
import copy
import numpy as np
from numpy.linalg import norm
from fedpredict.utils.compression_methods.parameters_svd import parameter_svd_write, inverse_parameter_svd_reading, if_reduces_size
from fedpredict.utils.compression_methods.sparsification import client_model_non_zero_indexes, client_specific_top_k_parameters
from fedpredict.utils.compression_methods.sparsification import sparse_crs_top_k, to_dense, sparse_matrix, get_not_zero_values
from fedpredict.utils.compression_methods.fedkd import fedkd_compression
import os

import math
import numpy as np

import sys
import copy
import os

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

def cosine_similarity(p_1, p_2):

    # compute cosine similarity
    try:
        return np.dot(p_1, p_2) / (norm(p_1) * norm(p_2))
    except Exception as e:
        print("cosine_similairty error")
        print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

class CKA(object):
    def __init__(self):
        pass

    def centering(self, K):
        n = K.shape[0]
        unit = np.ones([n, n])
        I = np.eye(n)
        H = I - unit / n
        return np.dot(np.dot(H, K), H)

    def rbf(self, X, sigma=None):
        GX = np.dot(X, X.T)
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        if sigma is None:
            mdist = np.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = np.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return np.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = X @ X.T
        L_Y = Y @ Y.T
        return np.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = np.sqrt(self.linear_HSIC(X, X))
        var2 = np.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = np.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = np.sqrt(self.kernel_HSIC(Y, Y, sigma))

        return hsic / (var1 * var2)



def fedpredict_layerwise_similarity(global_parameter, clients_parameters, similarity_per_layer_list):

    try:
        num_layers = len(global_parameter)
        num_clients = len(clients_parameters)
        print("global: ", num_layers)
        similarity_per_layer = {i: {} for i in range(num_clients)}
        # interest_layers = [0, 1, int(num_layers/2)-2, int(num_layers/2)-1, num_layers-2, num_layers-1]
        interest_layers = [0, num_layers - 2]
        difference_per_layer = {i: {j: {'min': [], 'max': []} for j in range(num_layers)} for i in range(num_clients)}
        difference_per_layer_vector = {j: [] for j in range(num_layers)}
        mean_similarity_per_layer = {i: {'mean': 0, 'ci': 0} for i in range(num_layers)}
        mean_difference_per_layer = {i: {'min': 0, 'max': 0} for i in range(num_layers)}

        for client_id in range(num_clients):

            client = clients_parameters[client_id]
            logger.info(f"cliente antes: {len(client)}")

            for layer_index in range(num_layers):
                client_layer = client[layer_index]
                global_layer = global_parameter[layer_index]
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
                        # print("do cliente: ", client_layer.shape, " global: ", global_layer.shape)
                        client_layer_k = client_layer[k][0]

                        # if gradient:
                        #     client_layer_k = global_layer_k - client_layer_k
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
                    # if layer_index == 0:
                    #     print("do cliente: ", client_similarity, len(client_similarity))
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
            for client_id in range(num_clients):
                similarities.append(similarity_per_layer[client_id][layer_index])
                min_difference += difference_per_layer[client_id][layer_index]['min']
                max_difference += difference_per_layer[client_id][layer_index]['max']

            mean = np.mean(similarities)
            if layer_index not in similarity_per_layer_list:
                similarity_per_layer_list[layer_index] = []
                mean_similarity_per_layer[layer_index]["mean"] = 0
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
        df = float(max(0, abs(np.mean(similarity_per_layer_list[0]) - np.mean(
                        similarity_per_layer_list[num_layers - 2]))))
        logger.info(f" similaridade {similarity_per_layer_list} df {df}")
        return similarity_per_layer, mean_similarity_per_layer, np.mean(layers_mean_similarity), similarity_per_layer_list, df
    except Exception as e:
        logger.critical("Method: fedpredict_layerwise_similarity")
        logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

