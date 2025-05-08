# Installation

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
