import sys
import logging
from fedpredict.utils.compression_methods.parameters_svd import parameter_svd_write, inverse_parameter_svd_reading, if_reduces_size
import numpy as np

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

def fedkd_compression_core(parameters, energy):
    try:
        for i in range(0, len(parameters), 3):
            u = parameters[i]
            v = parameters[i + 1]
            s = parameters[i + 2]

            if len(v) == 0:
                continue

            if len(u.shape) == 4:
                u = np.transpose(u, (2, 3, 0, 1))
                s = np.transpose(s, (2, 0, 1))
                v = np.transpose(v, (2, 3, 0, 1))
            threshold = 1
            if np.sum(np.square(s)) == 0:
                continue
            else:
                for singular_value_num in range(len(s)):
                    if np.sum(np.square(s[:singular_value_num])) > energy * np.sum(np.square(s)):
                        threshold = singular_value_num
                        break
                u = u[:, :threshold]
                s = s[:threshold]
                v = v[:threshold, :]
                # support high-dimensional CNN param
                if len(u.shape) == 4:
                    u = np.transpose(u, (2, 3, 0, 1))
                    s = np.transpose(s, (1, 2, 0))
                    v = np.transpose(v, (2, 3, 0, 1))

                parameters[i] = u
                parameters[i + 1] = v
                parameters[i + 2] = s

        return parameters


    except Exception as e:

        logger.critical("Method: fedkd_compression_core")

        logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def fedkd_compression(lt, layers_compression_range, T, t, M, parameter):
    try:
        nt = t - lt
        layers_fraction = []
        n_components_list = []
        for i in range(M):
            # if i % 2 == 0:
            layer = parameter[i]
            if len(layer.shape) >= 2:

                if layers_compression_range[i] > 0:
                    n_components = layers_compression_range[i]
                else:
                    n_components = None
            else:
                n_components = None

            n_components_list.append(n_components)

        logger.info(f"Vetor de componentes: {n_components_list}")

        parameter = parameter_svd_write(parameter, n_components_list)
        tmin = 0.01
        tmax = 0.95
        energy = tmin + (tmax-tmin)*(t / T)
        logger.info(f"energy: {energy}")
        parameter = fedkd_compression_core(parameter, energy)
        parameter = [np.array(i) for i in parameter]

        return parameter, layers_fraction
    except Exception as e:
        logger.critical("Method: fedkd_compression")
        logger.critical("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))