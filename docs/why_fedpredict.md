# Why FedPredict?

It is better working with the **prediction stage**. See the comparison below!

![](images/contribu.jpeg)

## How does it work?

FedPredict intelligently combines global and local model parameters, assigning dynamic weights to each based on several factors. 
These factors include the evolution level (el) of the global model, the update level (ul) of the local model, and the similarity (s) between previously seen data (i.e., data used in prior training) and newly acquired data.
Using this adaptive combination, the client generates a personalized model, which is then used for prediction on validation or test data.

![](images/fedpredictv5.jpeg)

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
