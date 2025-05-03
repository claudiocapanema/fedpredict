
# Welcome to FedPredict
## The first-ever plugin for Federated Learning!

**FedPredict** is a Federated Learning (FL) plugin designed to enhance existing FL solutions without requiring additional training or computational overhead.
It enables personalization in standard algorithms such as FedAvg and FedYogi, boosting performance in scenarios with non-IID data.

As a modular component, FedPredict operates exclusively during the prediction phase of FL and does not require any modifications to the training process.

This project has been developed through a collaboration between the **WISEMAP Lab (UFMG)**, **H.IAAC Lab (UNICAMP)**, and **NESPED Lab (UFV)**.

The list of projects currently using FedPredict includes (continuously updated):

- [FL-H.IAAC_docker](https://github.com/claudiocapanema/FL-HIAAC_docker): it has the code of the experiments of FedPredict papers in **IEEE DCOSS-IoT [2023](https://ieeexplore.ieee.org/document/10257293) and [2024](https://ieeexplore.ieee.org/abstract/document/10621488)** (i.e., FedPredict and FedPredict-Dynamic), and **IEEE Transactions on Emerging Topics** [2025](https://ieeexplore.ieee.org/abstract/document/10713874) (extended FedPredict).
- PFLib (will be available soon).

## Documentation

Please access the FedPredict [documentation](https://claudiocapanema.github.io/fedpredict/) for tutorials and API details. (updating)

## Why FedPredict?

It is better working with the **prediction stage**. See the comparison below!

![](docs/images/contribu.jpeg)

## How does it work?

FedPredict intelligently combines global and local model parameters, assigning dynamic weights to each based on several factors. 
These factors include the evolution level (el) of the global model, the update level (ul) of the local model, and the similarity (s) between previously seen data (i.e., data used in prior training) and newly acquired data.
Using this adaptive combination, the client generates a personalized model, which is then used for prediction on validation or test data.

![](docs/images/fedpredictv5.jpeg)

## Benefits

The list of benefits of the plugin is as follows:

1. **High performance**: Achieves strong performance in heterogeneous data environments.
2. **High efficiency for FL**: Maintains high performance even with reduced training.
3. **Data shift-awareness**: FedPredict enables near-instant adaptation to new scenarios in the presence of concept drift.
4. **Task independent**: Can be applied to any type of deep neural network task.
5. **Easy to use and modular**: No modifications are required in the training phase of your FL solution.
6. **Lightweight**: Built from simple, efficient operations.
7. **Low downlink communication cost**: The FedPredict server compresses global model parameters to reduce communication overhead.

Just plug and play!

## Installation

FedPredict is compatible with Python ≥ 3.8 and has been tested on the latest versions of Ubuntu.
With your virtual environment activated, if you are using PyTorch, you can install FedPredict from PyPI by running the following command:

```python
    pip install fedpredict[torch]
```

[//]: # (If you are using **Flower** for FL simulation, type:)

[//]: # ()
[//]: # (```python)

[//]: # (    pip install fedpredict[flwr])

[//]: # (```)

## FL requirements

In general, if your solution shares structural similarities with FedAvg, then FedPredict is ready to be integrated.
The requirements are outlined below:

| Requirement | Description                                                                                                                                                        |
| :- |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Sharing all layers | Clients must upload all model layers at each round so that the server can aggregate a global model, which can then be directly utilized by any new client—similar to the standard FedAvg approach. |
| Same model structure | The layers of the global and local models must have matching shapes to enable proper parameter combination.                                                       |
| Predicting using the combined model | On the client side, the original method must be flexible enough to perform inference using the combined model; otherwise, the plugin will have no effect.       |

## Components

Our solution has two main components: FedPredict client and FedPredict server. Their objectives are described below:

| Components                   | Objective                                                                                                           | 
|:-----------------------------|:--------------------------------------------------------------------------------------------------------------------|
| FedPredict Client            | Transfer the knowledge from the updated global model to the client's stale local model                              |
| FedPredict server (optional) | Compresses the updated global model parameters to further send to the clients. Used together with FedPredict client |