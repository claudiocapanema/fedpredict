"""pytorch_fedpredict_example: A Flower / PyTorch / FedPredict app.
    This basic version of FedPredict requires a small modification
    on the server side:the server must communicate the current
     training round number to the selected client during each
     training cycle. """

from typing import List, Tuple

from flwr.common import Context, Metrics, Scalar, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common.typing import UserConfig
from pytorch_fedpredict_example.task import Net, get_weights
import logging
logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def on_fit_config_fn(server_round: int) -> dict[str, Scalar]:
    config = {"server_round": server_round}
    return config

def on_evaluate_config_fn(server_round: int) -> dict[str, Scalar]:
    config = {"server_round": server_round}
    return config

def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    try:
        # Read from config
        num_rounds = context.run_config["num-server-rounds"]

        # Initialize model parameters
        ndarrays = get_weights(Net())
        parameters = ndarrays_to_parameters(ndarrays)

        # Define the strategy
        strategy = FedAvg(
            fraction_fit=0.3,
            fraction_evaluate=context.run_config["fraction-evaluate"],
            min_available_clients=2,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            evaluate_metrics_aggregation_fn=weighted_average,
            initial_parameters=parameters,
        )
        config = ServerConfig(num_rounds=num_rounds)

        return ServerAppComponents(strategy=strategy, config=config)
    except Exception as e:
        logger.error("server_fn error")
        logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

# Create ServerApp
app = ServerApp(server_fn=server_fn)
