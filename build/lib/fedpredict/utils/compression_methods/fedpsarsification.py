
import torch
import numpy as np

from scipy.sparse import coo_array, csr_matrix

def sparse_top_k(x, k):
    vec_x = x.flatten()
    d = int(len(vec_x))
    # print(d)
    k = int(np.ceil(d * k))
    # print(k)
    indices = torch.abs(vec_x).topk(k)[1]
    out_x = torch.zeros_like(vec_x)
    out_x[indices] = vec_x[indices]
    out_x = out_x.reshape(x.shape)
    # print(x.shape)
    return csr_matrix(out_x.numpy())

def fedsparsification_server(parameters, k):

    for i in range(len(parameters)):

        parameters[i] = sparse_top_k(torch.from_numpy(parameters[i]), k)

    return parameters