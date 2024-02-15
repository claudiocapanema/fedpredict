
# FedPredict

FedPredict is a Federated Learning (FL) plugin that can significantly improve FL solutions without requiring additional training or expensive processing. FedPredict enanles personalization for tradditional methods, such as FedAvg and FedYogi. It is also a modular plugin that operates in the prediction stage of FL without requiring any modification in the training step. 
This project has been developed in the laboratories WISEMAP (UFMG) and H.IAAC (UNICAMP).

## Benefits and use cases

FedPredict has the following benefits:
1. Avoid local training, which is costly and takes more time, by just downloading the current global model and combining it with the local one to perform predictions.
2. Allowing traditional solutions (e.g., FedAvg) to perform well in non-IID scenarios.
3. Reduce downlink communication cost: the proposed techniques for reducing downlink communication are independent and flexible and can be leveraged in other solutions.

The possible use cases are listed as follows:

1. Different topologies: FL topologies that inhered the standard Server-Client can use the plugin, such as Server Clod-Edge Server-Client.
2. Outdated client: a client with an outdated model (i.e., trained long ago).
3. Client with insufficient resources (e.g., data, energy, among others) to frequently perform local training.
4. New client (i.e., dynamicity in FL): a client recently added to the FL system and has not trained yet. These clients suffer from low performance in model personalization-based solutions.

## FL requirements

Are you interested in improving your Federated Learning solution with FedPredict? Take note of the requirements your method should satisfy:

1. **Sharing all layers**. The clients have to upload all model layers at every round so the server can aggregate a global model that can be directly leveraged by a new client, as in FedAvg.
2. **Same model structure**. The layers of the global and local models have to have the same shape to allow the combination of parameters.
3.  **Predicting using the combined model**. On the client side, the original method has to be flexible enough to make predictions based on the combined model; otherwise, the plugin will have no effect.

## Versions

FedPredict-Client is placed on the client-side and the features of its versions are listed below:

| Module | Static clients | Dynamic clients | Static heterogeneous data | Dynamic heterogeneous data | 
| :---         |     :---:      |     :---:     |    :---:   |  :--------------:|
|    FedPredict Client v.1  |    :heavy_check_mark:   |    -   |    -   |   :heavy_check_mark:    |

FedPredict-Server is placed on the server-side and is responsible for compressing the shared global model parameters.

### Citing

If FedPredict has been useful to you, please cite our [paper](https://ieeexplore.ieee.org/abstract/document/10257293). The BibTeX is presented as follows:

```
@inproceedings{capanema2023fedpredict,
  title={FedPredict: Combining Global and Local Parameters in the Prediction Step of Federated Learning},
  author={Capanema, Cl{\'a}udio GS and de Souza, Allan M and Silva, Fabr{\'\i}cio A and Villas, Leandro A and Loureiro, Antonio AF},
  booktitle={2023 19th International Conference on Distributed Computing in Smart Systems and the Internet of Things (DCOSS-IoT)},
  pages={17--24},
  year={2023},
  doi={https://doi.org/10.1109/DCOSS-IoT58021.2023.00012},
  organization={IEEE}
}
```


