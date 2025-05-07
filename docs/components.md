## Components

Our solution has two main components: FedPredict client and FedPredict server. Their objectives are described below:

| Components                   | Objective                                                                                                           | 
|:-----------------------------|:--------------------------------------------------------------------------------------------------------------------|
| FedPredict Client            | Transfer the knowledge from the updated global model to the client's stale local model                              |
| FedPredict server (optional) | Compresses the updated global model parameters to further send to the clients. Used together with FedPredict client |