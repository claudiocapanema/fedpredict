
# Welcome to FedPredict
## The first ever plugin for Federated Learning!

FedPredict is a Federated Learning (FL) plugin that can significantly improve FL solutions without requiring additional training or expensive processing. 
FedPredict enables personalization for traditional methods, such as FedAvg and FedYogi. 
It is also a modular plugin that operates in the prediction stage of FL without requiring any modification in the training step. 
This project has been developed in the laboratories WISEMAP (UFMG), H.IAAC (UNICAMP), and NESPED (UFV).

The list of projects that use FedPredict is the following (updating):

- [FL-H.IAAC](https://github.com/AllanMSouza/FL-H.IAAC): it has the code of the experiments of FedPredict papers in **DCOSS-IoT 2023 and 2024 (i.e., FedPredict and FedPredict-Dynamic**).
- PFLib (will be available soon).
- PyFlexe (will be available soon).

## How it works?

FedPredict intelligently combines global and local model parameters. In this process,
it assigns more or less weight to each type of parameter according to various factors, such as 
the evolution level (el) of the global model, the update level (ul) of the local model, and the 
similarity (s) between the old data (i.e., the one in which the model was previously trained) and 
the recently acquired data). Then, the client uses the combined model to make predictions over the test/val data.

![](../fedpredict%20v5.jpeg)

## Benefits

The list of benefits of the plugin as listed as follows:

1. **High accuracy**: it pushes up FL accuracy without requiring additional training!
2. **Support for dynamic data**: it is designed for stationary and non-stationary non-IID data.
3. **Concept drift**: FedPredict makes the model almost instantly adapt to the new scenario when concept drift occurs.
4. **Task independent**: apply FedPredict for any type of deep neural network task.
2. **Easy to use**: no modifications are necessary in the training stage of your solution!
3. **Low computational cost**: it is composed of simple operations.

Just plug and play!

## Installation

FedPredict is compatible with Python>=3.8 and is tested on the latest versions of Ubuntu.
With your virtual environment opened, if you are using **torch** type the following command to install FedPredict from Pypi:

```python
    pip install fedpredict[torch]
```

If you are using **Flower** for FL simulation, type:

```python
    pip install fedpredict[flwr]
```

## FL requirements

In general, if your solution shares some level of similarity with FedAvg, then FedPredict is ready to use.
The requirements are described as follows:


1. **Sharing all layers**. The clients have to upload all model layers at every round so the server can aggregate a global model that can be directly leveraged by a new client, as in FedAvg.
2. **Same model structure**. The layers of the global and local models have to have the same shape to allow the combination of parameters.
3.  **Predicting using the combined model**. On the client side, the original method has to be flexible enough to make predictions based on the combined model; otherwise, the plugin will have no effect.