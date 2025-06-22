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
try:
    import tensorflow as tf
except ImportError:
    _has_tf = False
else:
    _has_tf = True

from fedpredict.fedpredict_torch import fedpredict_client_torch, fedpredict_client_weight_predictions_torch
from fedpredict.fedpredict_core import fedpredict_server, layerwise_similarity
from fedpredict.utils.utils import fedpredict_layerwise_similarity
from fedpredict.fedpredict_core import layer_compression_range

__all__ = [
    "fedpredict_client_torch",
    "fedpredict_server",
    "fedpredict_client_weight_predictions_torch",
    "layerwise_similarity",
    "fedpredict_layerwise_similarity",
    "layer_compression_range"
]