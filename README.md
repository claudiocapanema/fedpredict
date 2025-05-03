
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

### Citing

If FedPredict has been useful to you, please cite our papers.

[FedPredict: Combining Global and Local Parameters in the Prediction Step of Federated Learning](https://ieeexplore.ieee.org/abstract/document/10257293) (original paper):

```
@INPROCEEDINGS{capanema2023fedpredict,
  author={Capanema, Cláudio G. S. and de Souza, Allan M. and Silva, Fabrício A. and Villas, Leandro A. and Loureiro, Antonio A. F.},
  booktitle={2023 19th International Conference on Distributed Computing in Smart Systems and the Internet of Things (DCOSS-IoT)}, 
  title={FedPredict: Combining Global and Local Parameters in the Prediction Step of Federated Learning}, 
  year={2023},
  volume={},
  number={},
  pages={17-24},
  keywords={Federated learning;Computational modeling;Neural networks;Mathematical models;Internet of Things;Distributed computing;Personalized Federated Learning;Neural Networks;Federated Learning Plugin},
  doi={10.1109/DCOSS-IoT58021.2023.00012}}
```
[A Novel Prediction Technique for Federated Learning](https://ieeexplore.ieee.org/abstract/document/10713874) (extended journal paper):
```
@ARTICLE{capanema2025@novel,
  author={Capanema, Cláudio G. S. and de Souza, Allan M. and da Costa, Joahannes B. D. and Silva, Fabrício A. and Villas, Leandro A. and Loureiro, Antonio A. F.},
  journal={IEEE Transactions on Emerging Topics in Computing}, 
  title={A Novel Prediction Technique for Federated Learning}, 
  year={2025},
  volume={13},
  number={1},
  pages={5-21},
  keywords={Servers;Costs;Training;Downlink;Adaptation models;Computational modeling;Federated learning;Quantization (signal);Context modeling;Accuracy;Federated learning plugin;neural networks;personalized federated learning},
  doi={10.1109/TETC.2024.3471458}}
```

[A Modular Plugin for Concept Drift in Federated Learning](https://ieeexplore.ieee.org/abstract/document/10621488) (FedPredict-Dynamic):
```
@INPROCEEDINGS{capanema2024@modular,
  author={Capanema, Cláudio G. S. and Da Costa, Joahannes B. D. and Silva, Fabrício A. and Villas, Leandro A. and Loureiro, Antonio A. F.},
  booktitle={2024 20th International Conference on Distributed Computing in Smart Systems and the Internet of Things (DCOSS-IoT)}, 
  title={A Modular Plugin for Concept Drift in Federated Learning}, 
  year={2024},
  volume={},
  number={},
  pages={101-108},
  keywords={Training;Accuracy;Federated learning;Geology;Concept drift;Data models;Internet of Things;Concept Drift;Personalized Federated Learning;Federated Learning Plugin;Neural Networks},
  doi={10.1109/DCOSS-IoT61029.2024.00024}}
```