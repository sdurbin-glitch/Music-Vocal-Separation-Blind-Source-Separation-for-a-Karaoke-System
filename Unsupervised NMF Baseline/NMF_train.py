import numpy as np


def NMF_train(M, B_init, W_init, n_iter):
    epsilon = 1e-10
    B = np.maximum(np.array(B_init, dtype=float, copy=True), epsilon)
    W = np.maximum(np.array(W_init, dtype=float, copy=True), epsilon)

    for _ in range(n_iter):
        ratio = M / np.maximum(B @ W, epsilon)
        B *= (ratio @ W.T) / np.maximum(np.sum(W, axis=1)[None, :], epsilon)

        ratio = M / np.maximum(B @ W, epsilon)
        W *= (B.T @ ratio) / np.maximum(np.sum(B, axis=0)[:, None], epsilon)
    return B, W