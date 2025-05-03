"""pytorch_fedpredict_example: A Flower / PyTorch / FedPredict app."""

import copy
import sys
import torch
from flwr.client import NumPyClient

from pytorch_fedpredict_example.task import Net, get_weights, load_data, set_weights, test, train
import logging
logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

# Define FedAvg Client
class FedAvgClient(NumPyClient):
    def __init__(self, trainloader, valloader, local_epochs, learning_rate, num_server_rounds, client_id, client_state):
        try:
            self.local_model = Net()
            self.trainloader = trainloader
            self.valloader = valloader
            self.local_epochs = local_epochs
            self.lr = learning_rate
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        except Exception as e:
            logger.error("__init__ error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        try:
            """Train the model with data of this client."""
            set_weights(self.local_model, parameters)
            results = train(
                self.local_model,
                self.trainloader,
                self.valloader,
                self.local_epochs,
                self.lr,
                self.device,
            )
            return get_weights(self.local_model), len(self.trainloader.dataset), results
        except Exception as e:
            logger.error("fit error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        try:
            set_weights(self.local_model, parameters)
            loss, accuracy = test(self.local_model, self.valloader, self.device)
            return loss, len(self.valloader.dataset), {"accuracy": accuracy}
        except Exception as e:
            logger.error("evaluate error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))