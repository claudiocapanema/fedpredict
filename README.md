
# FedPredict

FedPredict is a Federated Learning (FL) plugin that provides generalization and personalization. It is modular and can be added in the prediction stage of several methods without modifying the training step. This project has been developed in the laboratories WISEMAP (UFMG) and H.IAAC (UNICAMP).

## Requirements

The plugin has the following requirements:

- **Req. 1: sharing all layers**. The clients have to upload all model layers at every round so the server can aggregate a global model that can be directly leveraged by a new client, as in FedAvg.
- **Req. 2: same model structure**. The layers of the global and local models have to have the same shape to allow the combination of parameters.
-  **Req. 3: predicting using the combined model**. On the client side, the original method has to be flexible enough to make predictions based on the combined model; otherwise, the plugin will have no effect.

## Versions

| Version | Static clients | Dynamic clients | Static heterogeneous data | Dynamic heterogeneous data | 
| :---         |     :---:      |     :---:     |    :---:   |  :--------------:|
|    v.1   |    :heavy_check_mark:   |    -   |    -   |   :heavy_check_mark:    |

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


