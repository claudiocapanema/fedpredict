"""pytorch_fedpredict_example: A Flower / PyTorch / FedPredict app."""

import sys
from flwr.client import ClientApp
from flwr.common import ArrayRecord, Context

from pytorch_fedpredict_example.task import load_data
from pytorch_fedpredict_example.clients.fedavg_client import FedAvgClient
from pytorch_fedpredict_example.clients.fedavg_fp_client import FedAvgClientFP

import logging
logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    try:
        # Read the node_config to fetch data partition associated to this node
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]

        # Read run_config to fetch hyperparameters relevant to this run
        batch_size = context.run_config["batch-size"]
        alpha = context.run_config["alpha"]
        strategy = context.run_config["strategy"]
        trainloader, valloader = load_data(partition_id, num_partitions, alpha, batch_size)
        local_epochs = context.run_config["local-epochs"]
        learning_rate = context.run_config["learning-rate"]
        num_rounds = context.run_config["num-server-rounds"]
        client_state = context.state

        # Return Client instance
        if strategy == "FedAvg+FP":
            return FedAvgClientFP(trainloader, valloader, local_epochs, learning_rate, num_rounds, partition_id,
                                client_state).to_client()
        elif strategy == "FedAvg":
            return FedAvgClient(trainloader, valloader, local_epochs, learning_rate, num_rounds, partition_id,
                                  client_state).to_client()
        else:
            raise ValueError("Unknown strategy")
    except Exception as e:
        logger.error("client_fn error")
        logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

# Flower ClientApp
try:
    app = ClientApp(client_fn)
except Exception as e:
    logger.error("app error")
    logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))
