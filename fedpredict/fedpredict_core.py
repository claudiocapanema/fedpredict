import sys
import copy
import numpy as np
from .utils.compression_methods.parameters_svd import parameter_svd_write, inverse_parameter_svd_reading, if_reduces_size
from .utils.compression_methods.sparsification import sparse_crs_top_k, to_dense, sparse_matrix, get_not_zero_values
from .utils.compression_methods.fedkd import fedkd_compression
import os

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

# ===========================================================================================

# FedPredict server

def get_size(parameter):
    try:
        return parameter.nbytes
    except Exception as e:
        print("get_size")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


def fedpredict_server(parameters: np.array, client_evaluate_list: List[Tuple], fedpredict_clients_metrics: Dict,
                      server_round: int, num_rounds: int, model_shape: List, compression_method=None,
                      df: float=0, fl_framework=None):
    """

    Args:
        parameters: list[np.array], required
            List of numpy arrays
        client_evaluate_list: list[tuple], required
            List of clients' tuples
        fedpredict_clients_metrics: dict, required
            Dict of clients' metrics.
        server_round: int, required
        num_rounds: int, required
        df: float, optional. Default=0
            Layers' similarity difference. It is used when compression_method uses the "dls" technique.
        model_shape: list, optional
        compression_method: None, 'dls_compredict', 'dls', 'compredict', 'sparsification', 'fedkd', 'fedper'. Default=None
        fl_framework: None, 'flwr'. Default='flwr'
            Method for compressing the global model parameters before sending to thee clients

    Returns: list[tuple]

    """
    try:

        client_evaluate_list_fedpredict = []
        accuracy = 0
        size_of_parameters = []

        # Reuse previously compressed parameters
        previously_reduced_parameters = {}
        print("compression technique: ", compression_method)
        if compression_method in ["fedkd", "compredict", "dls_compredict"]:
            layers_compression_range = layer_compression_range(model_shape)
        fedkd = None
        for client_tuple in client_evaluate_list:
            client = client_tuple[0]
            client_id = str(client.cid)
            config = copy.copy(client_tuple[1].config)
            client_config = fedpredict_clients_metrics[str(client.cid)]
            nt = client_config['nt']
            if nt != 0 and nt in previously_reduced_parameters:
                process_parameters = False
            else:
                process_parameters = True

            if compression_method is None:
                process_parameters = True
            config['nt'] = nt
            config['metrics'] = client_config
            config['last_global_accuracy'] = accuracy
            config['total_server_rounds'] = num_rounds
            M = [i for i in range(len(parameters))]
            parameters_to_send = None
            if nt == 0 and compression_method not in ["fedkd", None]:
                config['M'] = []
                config['decompress'] = False
                config['layers_fraction'] = 0
                if fl_framework is None:
                    config['parameters'] = np.array([])
                    client_evaluate_list_fedpredict.append((client, config))
                elif fl_framework == 'flwr':
                    if _has_flwr:
                        evaluate_ins = EvaluateIns(ndarrays_to_parameters(np.array([])), config)
                        client_evaluate_list_fedpredict.append((client, evaluate_ins))
                    else:
                        raise ImportError(
                            "Flower is required. Digit: 'pip install fedpredict[flwr]' or 'pip install fedpredict[full]'")
                continue
            elif compression_method == 'fedkd':
                if fedkd is None:
                    parameters_to_send = parameters_to_send if parameters_to_send is not None else parameters
                    print("dentro 1: ", type(parameters_to_send[0]), len(parameters_to_send[0]))
                    parameters_to_send, layers_fraction = fedkd_compression(
                        fedpredict_clients_metrics[str(client_id)]['round_of_last_fit'], layers_compression_range,
                        num_rounds, client_id, server_round, len(M), parameters_to_send)
                    fedkd = parameters_to_send
                else:
                    parameters_to_send = fedkd
                config['decompress'] = True
                config['M'] = M
                config['layers_fraction'] = layers_fraction
                if fl_framework is None:
                    config['parameters'] = parameters_to_send
                    client_evaluate_list_fedpredict.append((client, config))
                elif fl_framework =='flwr':
                    if _has_flwr:
                        evaluate_ins = EvaluateIns(ndarrays_to_parameters(parameters_to_send), config)
                        client_evaluate_list_fedpredict.append((client, evaluate_ins))
                    else:
                        raise ImportError(
                            "Flower is required. Digit: 'pip install fedpredict[flwr]' or 'pip install fedpredict[full]'")
                continue
            elif compression_method == 'sparsification':
                k = 0.3

                t, k_values = sparse_crs_top_k([np.abs(i) for i in parameters], k)
                parameters_to_send = t
                # print("contar2")
                # print([len(i[i == 0]) for i in parameters_to_send])
                config['decompress'] = False
                config['M'] = M
                config['layers_fraction'] = 1
                if fl_framework is None:
                    config['parameters'] = parameters_to_send
                    client_evaluate_list_fedpredict.append((client, config))
                elif fl_framework == 'flwr':
                    if _has_flwr:
                        evaluate_ins = EvaluateIns(ndarrays_to_parameters(parameters_to_send), config)
                        client_evaluate_list_fedpredict.append((client, evaluate_ins))
                    else:
                        raise ImportError(
                            "Flower is required. Digit: 'pip install fedpredict[flwr]' or 'pip install fedpredict[full]'")
                continue

            elif compression_method is None:
                config['M'] = [i for i in range(len(parameters))]
                config['decompress'] = False
                config['layers_fraction'] = 1
                if fl_framework is None:
                    config['parameters'] = parameters
                    client_evaluate_list_fedpredict.append((client, config))
                elif fl_framework == 'flwr':
                    if _has_flwr:
                        evaluate_ins = EvaluateIns(ndarrays_to_parameters(parameters), config)
                        client_evaluate_list_fedpredict.append((client, evaluate_ins))
                    else:
                        raise ImportError(
                            "Flower is required. Digit: 'pip install fedpredict[flwr]' or 'pip install fedpredict[full]'")
                continue

            parameters_to_send = None

            print("Tamanho parametros antes: ", sum([i.nbytes for i in parameters]))

            if process_parameters:
                if "dls" in compression_method:
                    parameters_to_send, M = dls(fedpredict_clients_metrics[client_id]['first_round'],
                                                parameters, server_round, nt, num_rounds, df, size_of_parameters)
                    print("Tamanho parametros als: ", sum(i.nbytes for i in parameters_to_send))
                elif "per" in compression_method:
                    parameters_to_send, M = per(fedpredict_clients_metrics[client_id]['first_round'],
                                                parameters)
                    print("Tamanho parametros per: ", sum(i.nbytes for i in parameters_to_send),
                          len(parameters_to_send), len(M))
                layers_fraction = []
                if 'compredict' in compression_method:
                    parameters_to_send = parameters_to_send if parameters_to_send is not None else parameters
                    parameters_to_send, layers_fraction = compredict(
                        fedpredict_clients_metrics[str(client_id)]['round_of_last_fit'], layers_compression_range,
                        num_rounds, server_round, len(M), parameters_to_send)
                    config['decompress'] = True
                    pass
                else:
                    config['decompress'] = False
                    print("nao igual")
                # config['decompress'] = False
                print("Novos parametros para nt: ", nt)
                decompress = config['decompress']
                previously_reduced_parameters[nt] = [copy.deepcopy(parameters_to_send), M, layers_fraction, decompress]

            else:
                print("Reutilizou parametros de nt: ", nt)
                parameters_to_send, M, layers_fraction, decompress = previously_reduced_parameters[nt]

            parameters_to_send = [np.array(i) for i in parameters_to_send]
            print("Tamanho parametros compredict: ", sum(i.nbytes for i in parameters_to_send))
            for i in range(1, len(parameters)):
                size_of_parameters.append(get_size(parameters[i]))
            fedpredict_clients_metrics[str(client.cid)]['acc_bytes_rate'] = size_of_parameters
            config['M'] = M
            config['decompress'] = decompress
            config['layers_fraction'] = layers_fraction
            if fl_framework is None:
                config['parameters'] = parameters_to_send
                client_evaluate_list_fedpredict.append((client, config))
            elif fl_framework == 'flwr':
                if _has_flwr:
                    evaluate_ins = EvaluateIns(ndarrays_to_parameters(parameters_to_send), config)
                    client_evaluate_list_fedpredict.append((client, evaluate_ins))
                else:
                    raise ImportError("Flower is required. Digit: 'pip install fedpredict[flwr]' or 'pip install fedpredict[full]'")

        return client_evaluate_list_fedpredict

    except Exception as e:
        print("fedpredict server")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


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

            print("Vetor de componentes: ", n_components_list)

            parameter = parameter_svd_write(parameter, n_components_list)

            # print("Client: ", client_id, " round: ", server_round, " nt: ", nt, " norm: ", np.mean(gradient_norm), " camadas: ", M, " todos: ", gradient_norm)
            print("modelo compredict: ", [i.shape for i in parameter])

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
        print("compredict")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


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
        print("layer compression range")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


def dls(first_round, parameters,
        server_round, nt, num_rounds, df, size_of_layers):
    try:
        M = [i for i in range(len(parameters))]
        n_layers = len(parameters) / 2

        # return parameters, M

        size_list = []
        for i in range(len(parameters)):
            tamanho = get_size(parameters[i])
            size_list.append(tamanho)
        if first_round != -1:
            # baixo-cima
            M = fedpredict_core_layer_selection(t=server_round, T=num_rounds, nt=nt, n_layers=n_layers,
                                                size_per_layer=size_of_layers, df=df)
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
        print("dls")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

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
        print("per")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


def fedpredict_core(t, T, nt, fc, il):
    try:

        # # 99 bom com alpha 1.0 e 0.1
        # if fc > 0.94:
        # # if fc > 0.94 and imbalance_level < 0.4:
        #     global_model_weight = 1
        # else:
        #     global_model_weight = 0
        #
        # global_model_weight = 1

        # pior do que 99 com alpha 1.0 e 0.1
        # global_model_weight = 0

        if nt == 0:
            global_model_weight = 0
        elif nt == t or (fc == 1):
            global_model_weight = 1
        elif fc is not None and il is not None and (fc > 0.94 and il < 0.4):
            global_model_weight = 1
        else:
            update_level = 1 / nt
            evolution_level = t / T
            eq1 = (- evolution_level-update_level)
            eq2 = round(np.exp(eq1), 6)
            global_model_weight = eq2

        local_model_weights = 1 - global_model_weight

        # print("rodada: ", t, " rounds sem fit: ", nt, "\npeso global: ", global_model_weight, " peso local: ",
        #       local_model_weights)

        return local_model_weights, global_model_weight

    except Exception as e:
        print("fedpredict core")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


def fedpredict_dynamic_core(t, T, nt, s=1, fc=None, il=None, dh=None, ps=None, logs=False):
    try:
        s = float(np.round(s, 1))

        if nt == 0:
            global_model_weight = 0
        elif nt == t:
            global_model_weight = 1
        # elif similarity != 1:
        #     global_model_weight = 1
        elif fc is not None and il is not None and dh is not None and ps is not None:
            if (fc["global"] > fc["reference"] and il["global"] < il["reference"] and dh["global"] > dh[
                "reference"]) and (nt > 0 and ps["global"] < ps["reference"] and dh["global"] > dh["reference"]):
                global_model_weight = 1
        else:
            update_level = 1 / nt
            evolution_level = t / int(T)
            eq1 = (-update_level - evolution_level) * s
            eq2 = round(np.exp(eq1), 6)
            global_model_weight = eq2

        local_model_weights = 1 - global_model_weight

        if logs:
            print(f"round {t} nt {nt} gw {global_model_weight} lw {local_model_weights}")

        return local_model_weights, global_model_weight

    except Exception as e:
        print("fedpredict core")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


def fedpredict_core_layer_selection(t, T, nt, n_layers, size_per_layer, df):
    try:

        # 9
        if nt == 0:
            shared_layers = 0
        else:
            print("similaridade layer selection: ", df)
            update_level = 1 / nt
            evolution_level = t / T
            eq1 = (-update_level - evolution_level) * (df)  # v8 ótimo
            eq2 = round(np.exp(eq1), 6)
            shared_layers = int(np.ceil(eq2 * n_layers))

        shared_layers = [i for i in range(shared_layers * 2)]

        print("Shared layers: ", shared_layers, " rodada: ", t)

        return shared_layers

    except Exception as e:
        print("fedpredict core server layer selection")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


def fedpredict_core_compredict(t, T, nt, layer, compression_range):
    try:
        updated_level = 1 / nt
        columns = layer.shape[-1]
        fraction = compression_range / columns
        evolution_level = t / T

        eq1 = - evolution_level - updated_level  # v5
        fc_l = round(np.exp(eq1), 6)
        n_components = max(round(fc_l * compression_range), 1)
        print("fracao: ", fc_l, " fraction: ", fraction, " componentes: ", n_components, " de ", compression_range)

        return n_components

    except Exception as e:
        print("fedpredict core server compredict")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


# def set_parameters_to_model(parameters, model_name):
#     # print("tamanho: ", self.input_shape, " dispositivo: ", self.device)

def fedpredict_similarity_per_round_rate(similarities, num_layers, current_round, round_window):
    similarities_list = {i: [] for i in range(num_layers)}
    similarity_rate = {i: 0 for i in range(num_layers)}
    initial_round = 1

    for round in range(initial_round, current_round):

        for layer in range(num_layers):
            similarities_list[round].append(similarities[round])

    for layer in range(num_layers):
        similarity_rate[layer] = (similarities_list[current_round] - similarities_list[initial_round]) / round_window


def fedpredict_layerwise_similarity(global_parameter, clients_parameters, clients_ids,
                                    similarity_per_layer_list):
    num_layers = len(global_parameter)
    num_clients = len(clients_parameters)
    print("global: ", num_layers)
    similarity_per_layer = {i: {} for i in clients_ids}
    # interest_layers = [0, 1, int(num_layers/2)-2, int(num_layers/2)-1, num_layers-2, num_layers-1]
    interest_layers = [0, num_layers - 2]
    difference_per_layer = {i: {j: {'min': [], 'max': []} for j in range(num_layers)} for i in clients_ids}
    difference_per_layer_vector = {j: [] for j in range(num_layers)}
    mean_similarity_per_layer = {i: {'mean': 0, 'ci': 0} for i in range(num_layers)}
    mean_difference_per_layer = {i: {'min': 0, 'max': 0} for i in range(num_layers)}

    for client_id in range(num_clients):

        client = clients_parameters[client_id]
        client_id = clients_ids[client_id]
        print("cliente antes: ", len(client))

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
        print("""similaridade (camada {}): {}""".format(layer_index, mean_similarity_per_layer[layer_index]))
    for layer in difference_per_layer_vector:
        if np.sum(difference_per_layer_vector[layer]) == 0:
            continue
        # df = pd.DataFrame({'Difference': difference_per_layer_vector[layer], 'x': [i for i in range(len(difference_per_layer_vectore[layer]))]})
        # box_plot(df=df, base_dir='', file_name="""boxplot_difference_layer_{}_round_{}_dataset_{}_alpha_{}""".format(str(layer), str(server_round), dataset, alpha), x_column=None, y_column='Difference', title='Difference between global and local parameters', y_lim=True, y_max=0.065)

    return similarity_per_layer, mean_similarity_per_layer, np.mean(layers_mean_similarity), similarity_per_layer_list


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
        print("get_size")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


def fedpredict_server(parameters: np.array, client_evaluate_list: List[Tuple], fedpredict_clients_metrics: Dict,
                      server_round: int, num_rounds: int, model_shape: List, compression_method=None,
                      df: float=0, fl_framework=None):
    """

    Args:
        parameters: list[np.array], required
            List of numpy arrays
        client_evaluate_list: list[tuple], required
            List of clients' tuples
        fedpredict_clients_metrics: dict, required
            Dict of clients' metrics.
        server_round: int, required
        num_rounds: int, required
        df: float, optional. Default=0
            Layers' similarity difference. It is used when compression_method uses the "dls" technique.
        model_shape: list, optional
        compression_method: None, 'dls_compredict', 'dls', 'compredict', 'sparsification', 'fedkd', 'fedper'. Default=None
        fl_framework: None, 'flwr'. Default='flwr'
            Method for compressing the global model parameters before sending to thee clients

    Returns: list[tuple]

    """
    try:

        client_evaluate_list_fedpredict = []
        accuracy = 0
        size_of_parameters = []

        # Reuse previously compressed parameters
        previously_reduced_parameters = {}
        print("compression technique: ", compression_method)
        if compression_method in ["fedkd", "compredict", "dls_compredict"]:
            layers_compression_range = layer_compression_range(model_shape)
        fedkd = None
        for client_tuple in client_evaluate_list:
            client = client_tuple[0]
            client_id = str(client.cid)
            config = copy.copy(client_tuple[1].config)
            client_config = fedpredict_clients_metrics[str(client.cid)]
            nt = client_config['nt']
            if nt != 0 and nt in previously_reduced_parameters:
                process_parameters = False
            else:
                process_parameters = True

            if compression_method is None:
                process_parameters = True
            config['nt'] = nt
            config['metrics'] = client_config
            config['last_global_accuracy'] = accuracy
            config['total_server_rounds'] = num_rounds
            M = [i for i in range(len(parameters))]
            parameters_to_send = None
            if nt == 0 and compression_method not in ["fedkd", None]:
                config['M'] = []
                config['decompress'] = False
                config['layers_fraction'] = 0
                if fl_framework is None:
                    config['parameters'] = np.array([])
                    client_evaluate_list_fedpredict.append((client, config))
                elif fl_framework == 'flwr':
                    if _has_flwr:
                        evaluate_ins = EvaluateIns(ndarrays_to_parameters(np.array([])), config)
                        client_evaluate_list_fedpredict.append((client, evaluate_ins))
                    else:
                        raise ImportError(
                            "Flower is required. Digit: 'pip install fedpredict[flwr]' or 'pip install fedpredict[full]'")
                continue
            elif compression_method == 'fedkd':
                if fedkd is None:
                    parameters_to_send = parameters_to_send if parameters_to_send is not None else parameters
                    print("dentro 1: ", type(parameters_to_send[0]), len(parameters_to_send[0]))
                    parameters_to_send, layers_fraction = fedkd_compression(
                        fedpredict_clients_metrics[str(client_id)]['round_of_last_fit'], layers_compression_range,
                        num_rounds, client_id, server_round, len(M), parameters_to_send)
                    fedkd = parameters_to_send
                else:
                    parameters_to_send = fedkd
                config['decompress'] = True
                config['M'] = M
                config['layers_fraction'] = layers_fraction
                if fl_framework is None:
                    config['parameters'] = parameters_to_send
                    client_evaluate_list_fedpredict.append((client, config))
                elif fl_framework =='flwr':
                    if _has_flwr:
                        evaluate_ins = EvaluateIns(ndarrays_to_parameters(parameters_to_send), config)
                        client_evaluate_list_fedpredict.append((client, evaluate_ins))
                    else:
                        raise ImportError(
                            "Flower is required. Digit: 'pip install fedpredict[flwr]' or 'pip install fedpredict[full]'")
                continue
            elif compression_method == 'sparsification':
                k = 0.3

                t, k_values = sparse_crs_top_k([np.abs(i) for i in parameters], k)
                parameters_to_send = t
                # print("contar2")
                # print([len(i[i == 0]) for i in parameters_to_send])
                config['decompress'] = False
                config['M'] = M
                config['layers_fraction'] = 1
                if fl_framework is None:
                    config['parameters'] = parameters_to_send
                    client_evaluate_list_fedpredict.append((client, config))
                elif fl_framework == 'flwr':
                    if _has_flwr:
                        evaluate_ins = EvaluateIns(ndarrays_to_parameters(parameters_to_send), config)
                        client_evaluate_list_fedpredict.append((client, evaluate_ins))
                    else:
                        raise ImportError(
                            "Flower is required. Digit: 'pip install fedpredict[flwr]' or 'pip install fedpredict[full]'")
                continue

            elif compression_method is None:
                config['M'] = [i for i in range(len(parameters))]
                config['decompress'] = False
                config['layers_fraction'] = 1
                if fl_framework is None:
                    config['parameters'] = parameters
                    client_evaluate_list_fedpredict.append((client, config))
                elif fl_framework == 'flwr':
                    if _has_flwr:
                        evaluate_ins = EvaluateIns(ndarrays_to_parameters(parameters), config)
                        client_evaluate_list_fedpredict.append((client, evaluate_ins))
                    else:
                        raise ImportError(
                            "Flower is required. Digit: 'pip install fedpredict[flwr]' or 'pip install fedpredict[full]'")
                continue

            parameters_to_send = None

            print("Tamanho parametros antes: ", sum([i.nbytes for i in parameters]))

            if process_parameters:
                if "dls" in compression_method:
                    parameters_to_send, M = dls(fedpredict_clients_metrics[client_id]['first_round'],
                                                parameters, server_round, nt, num_rounds, df, size_of_parameters)
                    print("Tamanho parametros als: ", sum(i.nbytes for i in parameters_to_send))
                elif "per" in compression_method:
                    parameters_to_send, M = per(fedpredict_clients_metrics[client_id]['first_round'],
                                                parameters)
                    print("Tamanho parametros per: ", sum(i.nbytes for i in parameters_to_send),
                          len(parameters_to_send), len(M))
                layers_fraction = []
                if 'compredict' in compression_method:
                    parameters_to_send = parameters_to_send if parameters_to_send is not None else parameters
                    parameters_to_send, layers_fraction = compredict(
                        fedpredict_clients_metrics[str(client_id)]['round_of_last_fit'], layers_compression_range,
                        num_rounds, server_round, len(M), parameters_to_send)
                    config['decompress'] = True
                    pass
                else:
                    config['decompress'] = False
                    print("nao igual")
                # config['decompress'] = False
                print("Novos parametros para nt: ", nt)
                decompress = config['decompress']
                previously_reduced_parameters[nt] = [copy.deepcopy(parameters_to_send), M, layers_fraction, decompress]

            else:
                print("Reutilizou parametros de nt: ", nt)
                parameters_to_send, M, layers_fraction, decompress = previously_reduced_parameters[nt]

            parameters_to_send = [np.array(i) for i in parameters_to_send]
            print("Tamanho parametros compredict: ", sum(i.nbytes for i in parameters_to_send))
            for i in range(1, len(parameters)):
                size_of_parameters.append(get_size(parameters[i]))
            fedpredict_clients_metrics[str(client.cid)]['acc_bytes_rate'] = size_of_parameters
            config['M'] = M
            config['decompress'] = decompress
            config['layers_fraction'] = layers_fraction
            if fl_framework is None:
                config['parameters'] = parameters_to_send
                client_evaluate_list_fedpredict.append((client, config))
            elif fl_framework == 'flwr':
                if _has_flwr:
                    evaluate_ins = EvaluateIns(ndarrays_to_parameters(parameters_to_send), config)
                    client_evaluate_list_fedpredict.append((client, evaluate_ins))
                else:
                    raise ImportError("Flower is required. Digit: 'pip install fedpredict[flwr]' or 'pip install fedpredict[full]'")

        return client_evaluate_list_fedpredict

    except Exception as e:
        print("fedpredict server")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


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

            print("Vetor de componentes: ", n_components_list)

            parameter = parameter_svd_write(parameter, n_components_list)

            # print("Client: ", client_id, " round: ", server_round, " nt: ", nt, " norm: ", np.mean(gradient_norm), " camadas: ", M, " todos: ", gradient_norm)
            print("modelo compredict: ", [i.shape for i in parameter])

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
        print("compredict")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


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
        print("layer compression range")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


def dls(first_round, parameters,
        server_round, nt, num_rounds, df, size_of_layers):
    try:
        M = [i for i in range(len(parameters))]
        n_layers = len(parameters) / 2

        # return parameters, M

        size_list = []
        for i in range(len(parameters)):
            tamanho = get_size(parameters[i])
            size_list.append(tamanho)
        if first_round != -1:
            # baixo-cima
            M = fedpredict_core_layer_selection(t=server_round, T=num_rounds, nt=nt, n_layers=n_layers,
                                                size_per_layer=size_of_layers, df=df)
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
        print("dls")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


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
        print("per")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
