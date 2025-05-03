---
tags: [quickstart, vision, fds]
dataset: [CIFAR-10]
framework: [torch, torchvision, fedpredict]
---

# Federated Learning with FedPredict, Flower, and Pytorch

This is an introductory example to the [FedPredict](https://github.com/claudiocapanema/fedpredict) plugin, integrated to Flower and Pytorch. The code is adapted from the [PyTorch Quickstart example](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch).

## FedPredict

FedPredict is the **world's first federated learning plugin**.
It can be combined to a variety of strategies/solutions to provide several benefits (e.g., personalization, lower downlink communication cost, data shift support, among others). 

### Overview

In this introductory example, we show the basic usage of the plugin combined with FedAvg (FedAvg+FP) to provide personalization under non-IID data. For this, we use the Dirichlet distribution with α
={0.1, 1.0} to split the Cifar-10 dataset.

### Related papers

This code uses the FedPredict version presented in FedPredict [FedPredict: Combining Global and Local Parameters in the Prediction Step of Federated Learning](https://ieeexplore.ieee.org/abstract/document/10257293) (IEEE DCOSS-IoT 2023) and [A Novel Prediction Technique for Federated Learning](https://ieeexplore.ieee.org/abstract/document/10713874) (IEEE Transactions on Emerging Topics).

## Set up the project

### Clone the project

Start by cloning the example project:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
        && mv _tmp/examples/quickstart-pytorch-fedpredict . \
        && rm -rf _tmp \
        && cd quickstart-pytorch-fedpredict
```

This will create a new directory called `quickstart-pytorch-fedpredict` with the following structure:

```shell
quickstart-pytorch-fedpredict
├── pytorch_fedpredict_example
    ├── clients
        ├── fedavg_client.py
        ├── fedavg_fp_client.py
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `pytorchexample` package.

```bash
pip install -e .
```

## Run the project

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> \[!TIP\]
> This example might run faster when the `ClientApp`s have access to a GPU. If your system has one, you can make use of it by configuring the `backend.client-resources` component in `pyproject.toml`. If you want to try running the example with GPU right away, use the `local-simulation-gpu` federation as shown below. Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more.

```bash
# Run with the default federation (CPU only)
flwr run .
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config "num-server-rounds=5 learning-rate=0.05"
```

Run the project in the `local-simulation-gpu` federation that gives CPU and GPU resources to each `ClientApp`. By default, at most 5x`ClientApp` will run in parallel in the available GPU. You can tweak the degree of parallelism by adjusting the settings of this federation in the `pyproject.toml`.

```bash
# Run with the `local-simulation-gpu` federation
flwr run . local-simulation-gpu
```

> \[!TIP\]
> For a more detailed walk-through check our [quickstart PyTorch tutorial](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html)

### Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be intersted in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

If you are already familiar with how the Deployment Engine works, you may want to learn how to run it using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.
