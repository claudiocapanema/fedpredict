# Hello FedPredict

## Toy example - FedPredict Client


At the client-side, after receiving the global model parameters in the prediction state, apply FedPredict as follows:
```python
    from fedpredict import fedpredict_client_torch

    """Client evaluation"""
    
    t = 50 # current round
    T = 100 # total number of rounds
    nt = 3 # number of rounds since the last time the current client trained
    
    # apply fedpredict
    combinel_model = fedpredict_client_torch(local_model=local_model, 
                                             global_model=global_model, 
                                             t=t, 
                                             T=T, 
                                             nt=nt)
    # Use the combined model to perform predictions over the input data
    y_hat = combined_model(X_test)
```



