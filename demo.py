#%load_ext autoreload
#%autoreload 2

from lib import models, graph, coarsening, utils
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.sparse
import numpy as np
import time, shutil

#%matplotlib inline


flags = tf.app.flags
FLAGS = flags.FLAGS

# Graphs.
flags.DEFINE_integer('number_edges', 16, 'Graph: minimum number of edges per vertex.')
flags.DEFINE_string('metric', 'cosine', 'Graph: similarity measure (between features).')
# TODO: change cgcnn for combinatorial Laplacians.
flags.DEFINE_bool('normalized_laplacian', True, 'Graph Laplacian: normalized.')
flags.DEFINE_integer('coarsening_levels', 0, 'Number of coarsened graphs.')

flags.DEFINE_string('dir_data', os.path.join('data', 'rcv1'), 'Directory to store data.')
flags.DEFINE_integer('val_size', 400, 'Size of the validation set.')

dataset = utils.TextRCV1()

#dataset = utils.TextRCV1(data_home='data',download_if_missing=False)


a=1

# Selection of classes.
keep = ['C11','C12','C13','C14','C15','C16','C17','C18','C21','C22','C23','C24',
        'C31','C32','C33','C34','C41','C42','E11','E12','E13','E14','E21','E31',
        'E41','E51','E61','E71','G15','GCRIM','GDEF','GDIP','GDIS','GENT','GENV',
        'GFAS','GHEA','GJOB','GMIL','GOBIT','GODD','GPOL','GPRO','GREL','GSCI',
        'GSPO','GTOUR','GVIO','GVOTE','GWEA','GWELF','M11','M12','M13','M14']
assert len(keep) == 55  # There is 55 second-level categories according to LYRL2004.
keep.remove('C15')   # 151785 documents
keep.remove('GMIL')  # 5 documents only

dataset.show_doc_per_class()
dataset.show_classes_per_doc()
dataset.remove_classes(keep)
dataset.show_doc_per_class(True)
dataset.show_classes_per_doc()


# Remove documents with multiple classes.
dataset.select_documents()
dataset.data_info()


#dataset.normalize(norm='l1')
dataset.show_document(1)


perm = np.random.RandomState(seed=42).permutation(dataset.data.shape[0])
Ntest = dataset.data.shape[0] // 2
perm_test = perm[:Ntest]
perm_train = perm[Ntest:]
train_data = dataset.data[perm_train,:].astype(np.float32)
test_data = dataset.data[perm_test,:].astype(np.float32)
train_labels = dataset.labels[perm_train]
test_labels = dataset.labels[perm_test]

if False:
    graph_data = train.embeddings.astype(np.float32)
else:
    graph_data = dataset.data.T.astype(np.float32)

t_start = time.process_time()
dist, idx = graph.distance_lshforest(graph_data.astype(np.float64), k=FLAGS.number_edges, metric=FLAGS.metric)
A = graph.adjacency(dist.astype(np.float32), idx)
print("{} > {} edges".format(A.nnz//2, FLAGS.number_edges*graph_data.shape[0]//2))
A = graph.replace_random_edges(A, 0)
graphs, perm = coarsening.coarsen(A, levels=FLAGS.coarsening_levels, self_connections=False)
L = [graph.laplacian(A, normalized=True) for A in graphs]
print('Execution time: {:.2f}s'.format(time.process_time() - t_start))

assert FLAGS.coarsening_levels is 0


# Training set is shuffled already.
#perm = np.random.permutation(train_data.shape[0])
#train_data = train_data[perm,:]
#train_labels = train_labels[perm]

# Validation set.
if False:
    val_data = train_data[:FLAGS.val_size,:]
    val_labels = train_labels[:FLAGS.val_size]
    train_data = train_data[FLAGS.val_size:,:]
    train_labels = train_labels[FLAGS.val_size:]
else:
    val_data = test_data
    val_labels = test_labels


if False:
    utils.baseline(train_data, train_labels, test_data, test_labels)

common = {}
common['dir_name']       = 'rcv1/'
common['num_epochs']     = 4
common['batch_size']     = 100
common['decay_steps']    = len(train_labels) / common['batch_size']
common['eval_frequency'] = 200
common['filter']         = 'chebyshev5'
common['brelu']          = 'b1relu'
common['pool']           = 'mpool1'
C = max(train_labels) + 1  # number of classes

model_perf = utils.model_perf()

if True:
    name = 'softmax'
    params = common.copy()
    params['dir_name'] += name
    params['regularization'] = 0
    params['dropout']        = 1
    params['learning_rate']  = 1e3
    params['decay_rate']     = 0.95
    params['momentum']       = 0.9
    params['F']              = []
    params['K']              = []
    params['p']              = []
    params['M']              = [C]
    model_perf.test(models.cgcnn(L, **params), name, params,
                    train_data, train_labels, val_data, val_labels, test_data, test_labels)

if True:
    name = 'fc_softmax'
    params = common.copy()
    params['dir_name'] += name
    params['regularization'] = 0
    params['dropout']        = 1
    params['learning_rate']  = 0.1
    params['decay_rate']     = 0.95
    params['momentum']       = 0.9
    params['F']              = []
    params['K']              = []
    params['p']              = []
    params['M']              = [2500, C]
    model_perf.test(models.cgcnn(L, **params), name, params,
                    train_data, train_labels, val_data, val_labels, test_data, test_labels)


if True:
    name = 'fc_fc_softmax'
    params = common.copy()
    params['dir_name'] += name
    params['regularization'] = 0
    params['dropout']        = 1
    params['learning_rate']  = 0.1
    params['decay_rate']     = 0.95
    params['momentum']       = 0.9
    params['F']              = []
    params['K']              = []
    params['p']              = []
    params['M']              = [2500, 500, C]
    model_perf.test(models.cgcnn(L, **params), name, params,
                    train_data, train_labels, val_data, val_labels, test_data, test_labels)

if True:
    name = 'cgconv_softmax'
    params = common.copy()
    params['dir_name'] += name
    params['regularization'] = 1e-3
    params['dropout']        = 1
    params['learning_rate']  = 0.1
    params['decay_rate']     = 0.999
    params['momentum']       = 0
    params['F']              = [1]
    params['K']              = [5]
    params['p']              = [1]
    params['M']              = [C]
    model_perf.test(models.cgcnn(L, **params), name, params,
                    train_data, train_labels, val_data, val_labels, test_data, test_labels)


if True:
    name = 'cgconv_fc_softmax'
    params = common.copy()
    params['dir_name'] += name
    params['regularization'] = 0
    params['dropout']        = 1
    params['learning_rate']  = 0.1
    params['decay_rate']     = 0.999
    params['momentum']       = 0
    params['F']              = [5]
    params['K']              = [15]
    params['p']              = [1]
    params['M']              = [100, C]
    model_perf.test(models.cgcnn(L, **params), name, params,
                    train_data, train_labels, val_data, val_labels, test_data, test_labels)

model_perf.show()


