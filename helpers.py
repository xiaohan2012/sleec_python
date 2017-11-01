import numpy as np
from scipy.sparse import issparse


def precision_at_ks(true_Y, pred_Y, ks=[1, 2, 3, 4, 5]):
    result = {}
    true_labels = [set(true_Y[i, :].nonzero()[1]) for i in range(true_Y.shape[0])]
    label_ranks = np.fliplr(np.argsort(pred_Y, axis=1))
    for k in ks:
        pred_labels = label_ranks[:, :k]
        precs = [len(t.intersection(set(p))) / k
                 for t, p in zip(true_labels, pred_labels)]
        result[k] = np.mean(precs)
    return result


def print_hdf5_object(o):
    for k in o:
        print('{}: {}'.format(k, o[k].value))


def project(V, x):
    """
    V: (embed dim, feature dim)
    x: (feature_dim, )
    
    return:
    vector of (embed dim, )
    """
    if issparse(x):
        x = x.toarray().T
    if len(x.shape) == 1:
        x = x[:, None]

    return (V @ x).flatten()
