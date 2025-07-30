import  sys
import logging
import numpy as np
from sklearn.utils.extmath import randomized_svd

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

def parameter_svd_write(arrays, n_components_list, svd_type='tsvd'):

    try:

        u = []
        vt = []
        sigma_parameters = []
        arrays_compre = []
        for i in range(len(arrays)):
            if type(n_components_list) == list:
                n_components = n_components_list[i]
            else:
                n_components = n_components_list
            # print("Indice da camada: ", i)
            r = parameter_svd(arrays[i], n_components, svd_type)
            arrays_compre += r

        return arrays_compre

    except Exception as e:
        logger.critical("paramete_svd")
        logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def parameter_svd(layer, n_components, svd_type='tsvd'):

    try:
        if np.ndim(layer) == 1 or n_components is None:
            return [layer, np.array([]), np.array([])]
        elif np.ndim(layer) == 2:
            r = svd(layer, n_components, svd_type)
            return r
        elif np.ndim(layer) >= 3:
            u = []
            v = []
            sig = []
            for i in range(len(layer)):
                r = parameter_svd(layer[i], n_components, svd_type)
                u.append(r[0])
                v.append(r[1])
                sig.append(r[2])
            return [np.array(u), np.array(v), np.array(sig)]

    except Exception as e:
        logger.critical("parameter_svd")
        logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


def svd(layer, n_components, svd_type='tsvd'):

    try:
        np.random.seed(0)
        if n_components > 0 and n_components < 1:
            n_components = int(len(layer) * n_components)

        if svd_type == 'tsvd':
            U, Sigma, VT = randomized_svd(layer,
                                          n_components=n_components,
                                          n_iter=5,
                                          random_state=0)
        else:
            U, Sigma, VT = np.linalg.svd(layer, full_matrices=False)
            U = U[:, :n_components]
            Sigma = Sigma[:n_components]
            VT = VT[:n_components, :]

        # print(U.shape, Sigma.shape, VT.T.shape)
        return [U, VT, Sigma]

    except Exception as e:
        logger.critical("svd")
        logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def if_reduces_size(shape, n_components, dtype=np.float64):

    try:
        size = np.array([1], dtype=dtype)
        p = shape[0]
        q = shape[1]
        k = n_components

        if p*k + k*k + k*q < p*q:
            return True
        else:
            return False

    except Exception as e:
        logger.critical("if_reduces_size")
        logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


def inverse_parameter_svd_reading(arrays, model_shape):
    try:
        M = len(model_shape)
        if len(arrays) > M:
            M = int(len(arrays) / 3)
        reconstructed_model = []
        # logger.info(f"arrays {len(arrays)} antes inverse: {len(model_shape)}")

        flag = True
        for i, j in zip(arrays, model_shape):
            if i.shape != j:
                flag = False
        if flag:
            logger.info("nao descomprimiu cliente")
            return arrays

        for i in range(M):
            if i*3 + 2 < len(arrays):
                layer_shape = model_shape[i]
                u = arrays[i*3]
                v = arrays[i*3 + 1]

                si = arrays[i*3 + 2]
                if len(layer_shape) == 1:
                    parameter_layer = inverse_parameter_svd(u, v, layer_shape)
                else:
                    parameter_layer = inverse_parameter_svd(u, v, layer_shape, si)
                if parameter_layer is None:
                    pass
                reconstructed_model.append(parameter_layer)
            else:
                break

        # logger.info(f"original shape: {model_shape}")
        # logger.info(f"depois inverse: {[i.shape for i in reconstructed_model]}")

        return reconstructed_model

    except Exception as e:
        logger.critical("inverse_paramete_svd_reading")
        logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


def inverse_parameter_svd(u, v, layer_index, sigma=None, sig_ind=None):
    try:
        if len(v) == 0:
            return u
        elif len(layer_index) == 1:
            # print("u1")
            return u
        elif len(layer_index) == 2:
            # print("u2")
            return np.matmul(u * sigma, v)
        elif len(layer_index) == 3:
            # print("u3")
            layers_l = []
            for i in range(len(u)):
                layers_l.append(np.matmul(u[i] * sigma[i], v[i]))
            return np.array(layers_l)
        elif len(layer_index) == 4:
            layers_l = []
            # print("u4")
            for i in range(len(u)):
                layers_j = []

                for j in range(len(u[i])):
                    # print("u shape: ", u.shape, " sigma shape: ", sigma.shape, "v shape: ", v.shape, i, j)
                    layers_j.append(np.matmul(u[i][j] * sigma[i][j], v[i][j]))
                layers_l.append(layers_j)
            return np.array(layers_l)

    except Exception as e:
        logger.critical("inverse_parameter_svd")
        logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def test():
    original = [np.random.random((5, 5)) for i in range(1)]

    threshold = 2
    U, Sigma, VT = np.linalg.svd(original[0], full_matrices=False)
    print("compactado shapes 1: ", [i.shape for i in [U, Sigma, VT]])
    U = U[:, :threshold]
    Sigma = Sigma[ : threshold]
    VT = VT[:threshold, :]

    print("compactado shapes 2: ", [i.shape for i in [U, Sigma, VT]])
    print(original[0])

    # svd = parameter_svd_write(original, 0.5)
    # original_ = inverse_parameter_svd_reading(svd, [i.shape for i in original])
    # o = [U[0], Sigma.T, VT[0]]
    # print("numpy shape: ", [i.shape for i in o])
    #
    # print("Original: \n", original, [i.shape for i in original])
    # print("Np recontruido: \n", np.matmul(U*Sigma.T, VT.T))
    # print("Reconstruído: \n", original_)
    print("Reconstruido")
    r = np.matmul(U * Sigma[..., None, :], VT)
    print(r)


if __name__ == "__main__":
    test()