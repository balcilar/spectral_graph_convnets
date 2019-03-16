import tensorflow as tf
import numpy as np
import time
import sys, os
sys.path.insert(0, '..')
from lib import spectGCNN, graph, coarsening, utils



def grid_graph(m, corners=False):
    k=8
    z = graph.grid(m)
    dist, idx = graph.distance_sklearn_metrics(z, k=8, metric='euclidean')
    A = graph.adjacency(dist, idx)

    # Connections are only vertical or horizontal on the grid.
    # Corner vertices are connected to 2 neightbors only.
    if corners:
        import scipy.sparse
        A = A.toarray()
        A[A < A.max()/1.5] = 0
        A = scipy.sparse.csr_matrix(A)
        print('{} edges'.format(A.nnz))

    print("{} > {} edges".format(A.nnz//2, k*m**2//2))
    return A


A = grid_graph(28, corners=False)
# A = graph.replace_random_edges(A, 0)
graphs, perm = coarsening.coarsen(A, levels=4, self_connections=False)
L = [graph.laplacian(A, normalized=True) for A in graphs]


# dir_data=os.path.join('..', 'data', 'mnist')


# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets(dir_data, one_hot=False)

# train_data = mnist.train.images.astype(np.float32)
# val_data = mnist.validation.images.astype(np.float32)
# test_data = mnist.test.images.astype(np.float32)
# train_labels = mnist.train.labels
# val_labels = mnist.validation.labels
# test_labels = mnist.test.labels


# train_data = coarsening.perm_data(train_data, perm)
# val_data = coarsening.perm_data(val_data, perm)
# test_data = coarsening.perm_data(test_data, perm)

# C = max(mnist.train.labels) + 1 

num_examples=55000   # mnist.train.num_examples
C=10

common = {}
common['dir_name']       = 'mnist/'
common['num_epochs']     = 20
common['batch_size']     = 100
common['feature_size']   = 3
common['max_node']       = L[0].shape[0]
common['decay_steps']    = num_examples / common['batch_size']
common['eval_frequency'] = 30 * common['num_epochs']
common['brelu']          = 'b1relu'
common['pool']           = 'mpool1'
common['regularization'] = 5e-4
common['dropout']        = 0.5
common['learning_rate']  = 0.02  # 0.03 in the paper but sgconv_sgconv_fc_softmax has difficulty to converge
common['decay_rate']     = 0.95
common['momentum']       = 0.9
common['F']              = [32, 64]
common['K']              = [25, 25]
common['p']              = [4, 4]
common['M']              = [512, C]
name = 'cgconv_cgconv_fc_softmax'  # 'Chebyshev'
params = common.copy()
params['dir_name'] += name
params['filter'] = 'chebyshev5'
model=spectGCNN.cgcnn(L, **params)

# acc, los, ftime = model.fit(train_data, train_labels, val_data, val_labels)
# msg, tracc, trf1, trloss = model.evaluate(train_data, train_labels)
# print('train {}'.format(msg))
# msg, tsacc, tsf1, tsloss = model.evaluate(test_data, test_labels)
# print('test  {}'.format(msg))
        



