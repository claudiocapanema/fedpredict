"""pytorch_fedpredict_example: A Flower / PyTorch / FedPredict app."""

import copy
import sys
from pytorch_fedpredict_example.clients.fedavg_client import FedAvgClient
from fedpredict import fedpredict_client_torch
from flwr.common import ConfigRecord
from flwr.common import ArrayRecord

from pytorch_fedpredict_example.task import set_weights, test
import logging
logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

# Define FedAvg+FP Client
class FedAvgClientFP(FedAvgClient):
    def __init__(self, trainloader, valloader, local_epochs, learning_rate, num_server_rounds, client_id, client_state):
        try:
            super().__init__(trainloader, valloader, local_epochs, learning_rate, num_server_rounds, client_id, client_state)
            self.global_model = copy.deepcopy(self.local_model)
            self.lt = 0 # last round the client trained
            self.num_server_rounds = num_server_rounds
            self.client_state = client_state
        except Exception as e:
            logger.error("__init__ error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        try:
            t = config["server_round"]
            self.lt = t
            results = super().fit(parameters, config)
            self._save_layer_weights_to_state()
            return results
        except Exception as e:
            logger.error("fit error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        try:

            set_weights(self.global_model, parameters)
            t = config["server_round"]
            self._load_layer_weights_from_state()
            # Calculate the number of consecutive rounds the client has not been selected for training (nt)."
            nt = t - self.lt
            # Get the "combined_model" from FedPredict
            combined_model = fedpredict_client_torch(local_model=self.local_model, global_model=self.global_model,
                                                     t=t, T=self.num_server_rounds, nt=nt, device=self.device)
            # Test the "combined_model"
            loss, accuracy = test(combined_model, self.valloader, self.device)
            return loss, len(self.valloader.dataset), {"accuracy": accuracy}
        except Exception as e:
            logger.error("evaluate error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _save_layer_weights_to_state(self):
        """Save last layer weights to state."""
        try:
            arr_record = ArrayRecord(torch_state_dict=self.local_model.state_dict())

            # Add to RecordDict (replace if already exists)
            self.client_state["model"] = arr_record
            self.client_state["lt"] = ConfigRecord(config_dict={"lt": self.lt})
        except Exception as e:
            logger.error("_save_layer_weights_to_state error")
            logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

    def _load_layer_weights_from_state(self):
        """Load last layer weights to state."""
        if "model" not in self.client_state.array_records:
            return

        state_dict = self.client_state["model"].to_torch_state_dict()
        self.lt = self.client_state["lt"]["lt"]

        # apply previously saved classification head by this client
        self.local_model.load_state_dict(state_dict, strict=True)