# Quick start

## Preliminaries

FedPredict has four main components:

| Method | Description |                       Return                        |   Location   |
| :- | :-: |:---------------------------------------------------:|:------------:|
| fedpredict_client_traditional | Combines global and local parameters. Used when data is stationary and non-IID. |      The combined model as "torch.nn.Module".       | Client-side. |
| fedpredict_client_dynamic | Combines global and local parameters. Used when data is non-stationary and non-IID. |      The combined model as "torch.nn.Module".       | Client-side. |
| fedpredict_client_weight_predictions | Weight predictions to prioritze the most frequent classes in the newest data. Used when data is non-starionary and non-IID. |      The probabilities vectors "Numpy array".       | Client-side. |
| fedpredict_server | Compress the global model parameters for further sending to the clients | The global model parameters as "List[Numpy array]". | Server-side |

## Tutorials

## Simple usage


At the client-side, after receiving the global model parameters in the prediction state, apply FedPredict as follows:
```python
    t = 50 # current round
    T = 100 # total number of rounds
    nt = 3 # number of rounds since the last time the current client trained
    M = len(global_model_parameters) # the number of layers shared by the server
    
    # apply fedpredict
    combined_model = fedpredict_client_traditional(local_model=local_model, 
                                                   global_model_parameters=global_model_parameters,
                                                   t=t, 
                                                   T=T, 
                                                   nt=nt,
                                                   M=M)
    # Use the combined model to perform predictions over the input data
    y_hat = combined_model(X_test)
```




