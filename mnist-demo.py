#%% [markdown]
# # Graph Convolutional Neural Networks


import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import pdb #pdb.set_trace()
import collections
import time
import numpy as np
from sGCNN import Graph_ConvNet_LeNet5
from scipy.sparse import coo_matrix
import sys
sys.path.insert(0, 'lib/')
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if torch.cuda.is_available():
    print('cuda available')
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
    torch.cuda.manual_seed(1)
else:
    print('cuda not available')
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor
    torch.manual_seed(1)
    


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('datasets', one_hot=False) # load data in folder datasets/

train_data = mnist.train.images.astype(np.float32)
val_data = mnist.validation.images.astype(np.float32)
test_data = mnist.test.images.astype(np.float32)
train_labels = mnist.train.labels
val_labels = mnist.validation.labels
test_labels = mnist.test.labels
print(train_data.shape)
print(train_labels.shape)
print(val_data.shape)
print(val_labels.shape)
print(test_data.shape)
print(test_labels.shape)

from lib.grid_graph import grid_graph
from lib.coarsening import coarsen
from lib.coarsening import lmax_L
from lib.coarsening import perm_data
from lib.coarsening import rescale_L

# Construct graph
t_start = time.time()
grid_side = 28
number_edges = 8
metric = 'euclidean'
A = grid_graph(grid_side,number_edges,metric) # create graph of Euclidean grid

# Compute coarsened graphs
coarsening_levels = 4
L, perm = coarsen(A, coarsening_levels)

# Compute max eigenvalue of graph Laplacians
lmax = []
for i in range(coarsening_levels):
    lmax.append(lmax_L(L[i]))    
    L[i] = rescale_L(L[i], lmax[i]) 

print('lmax: ' + str([lmax[i] for i in range(coarsening_levels)]))

# Reindex nodes to satisfy a binary tree structure
train_data = perm_data(train_data, perm)
val_data = perm_data(val_data, perm)
test_data = perm_data(test_data, perm)


# make input data column shows feature 
tr_data=np.reshape(train_data,train_data.shape[0]*train_data.shape[1])
vl_data=np.reshape(val_data,val_data.shape[0]*val_data.shape[1])
ts_data=np.reshape(test_data,test_data.shape[0]*test_data.shape[1])

tr_data=np.expand_dims(tr_data, axis=1)
vl_data=np.expand_dims(vl_data, axis=1)
ts_data=np.expand_dims(ts_data, axis=1)

# make index vector to show which row correspondend to which data
tr_index=np.zeros((train_data.shape[0],train_data.shape[1]))
for i in range(0,train_data.shape[0]):
    tr_index[i,:]=i
vl_index=np.zeros((val_data.shape[0],val_data.shape[1]))
for i in range(0,val_data.shape[0]):
    vl_index[i,:]=i
ts_index=np.zeros((test_data.shape[0],test_data.shape[1]))
for i in range(0,test_data.shape[0]):
    ts_index[i,:]=i


tr_index=np.reshape(tr_index,tr_index.shape[0]*tr_index.shape[1])
vl_index=np.reshape(vl_index,vl_index.shape[0]*vl_index.shape[1])
ts_index=np.reshape(ts_index,ts_index.shape[0]*ts_index.shape[1])

# set number of vertex for each data point
tr_nvertex=train_data.shape[1]*np.ones(train_data.shape[0],dtype=np.int32)
ts_nvertex=test_data.shape[1]*np.ones(test_data.shape[0],dtype=np.int32)
vl_nvertex=val_data.shape[1]*np.ones(val_data.shape[0],dtype=np.int32)


print(train_data.shape)
print(val_data.shape)
print(test_data.shape)

print('Execution time: {:.2f}s'.format(time.time() - t_start))


del perm
# Delete existing network if exists
try:
    del net
    print('Delete existing network\n')
except NameError:
    print('No existing network to delete\n')



# network parameters
D = train_data.shape[1]
CL1_F = 32
CL1_K = 25
CL2_F = 64
CL2_K = 25
FC1_F = 512
FC2_F = 10
net_parameters = [D, CL1_F, CL1_K, CL2_F, CL2_K, FC1_F, FC2_F]


# instantiate the object net of the class 
net = Graph_ConvNet_LeNet5(net_parameters)
if torch.cuda.is_available():
    net.cuda()
print(net)


# Weights
L_net = list(net.parameters())


# learning parameters
learning_rate = 0.05
dropout_value = 0.5
l2_regularization = 5e-4 
batch_size = 20
num_epochs = 20
train_size = train_data.shape[0]
nb_iter = int(num_epochs * train_size) // batch_size
print('num_epochs=',num_epochs,', train_size=',train_size,', nb_iter=',nb_iter)


# Optimizer
global_lr = learning_rate
global_step = 0
decay = 0.95
decay_steps = train_size
lr = learning_rate
optimizer = net.update(lr) 

nfeat=tr_data.shape[1]


size0=0
size2=0
for i in range(batch_size):
            # take current data Laplacian , in this case all laplacian are same
            # normally i tmust be 
            # L0 = L[batch_idx][i][0].tocoo()
            # L2 = L[batch_idx][i][2].tocoo()
            L0 = L[0].tocoo()
            L2 = L[2].tocoo()
            size0+=L0.col.shape[0]
            size2+=L2.col.shape[0]

col0=np.zeros((size0),dtype=np.int32)
row0=np.zeros((size0),dtype=np.int32)
data0=np.zeros((size0))

col2=np.zeros((size2),dtype=np.int32)
row2=np.zeros((size2),dtype=np.int32)
data2=np.zeros((size2))

        

pid0=0
pid2=0
pid22=0
pidd=0
for i in range(batch_size):
            # take current data Laplacian , in this case all laplacian are same
            # normally i tmust be 
            # L0 = L[batch_idx][i][0].tocoo()
            # L2 = L[batch_idx][i][2].tocoo()
            L0 = L[0].tocoo()
            L2 = L[2].tocoo()           
            

            # col0= np.hstack((col0,L0.col+pid0))
            # row0= np.hstack((row0,L0.row+pid0))
            # data0= np.hstack((data0,L0.data))
            col0[pid0:pid0+L0.col.shape[0]]=L0.col+pidd
            row0[pid0:pid0+L0.col.shape[0]]=L0.row+pidd
            data0[pid0:pid0+L0.col.shape[0]]=L0.data
            pid0+=L0.col.shape[0]

            # col2= np.hstack((col2,L2.col+pid2))
            # row2= np.hstack((row2,L2.row+pid2))
            # data2= np.hstack((data2,L2.data))
            col2[pid2:pid2+L2.col.shape[0]]=L2.col+pid22
            row2[pid2:pid2+L2.col.shape[0]]=L2.row+pid22
            data2[pid2:pid2+L2.col.shape[0]]=L2.data
            pid2+=L2.col.shape[0]    
            pid22+=L2.shape[0]   

            pidd+=L0.shape[0]





# loop over epochs
indices = collections.deque()
for epoch in range(num_epochs):  # loop over the dataset multiple times

    # reshuffle 
    indices.extend(np.random.permutation(train_size)) # rand permutation
    
    # reset time
    t_start = time.time()
    
    # extract batches
    running_loss = 0.0
    running_accuray = 0
    running_total = 0
    while len(indices) >= batch_size:
        
        # extract batches
        train_x=np.zeros((0,nfeat))
        
        batch_idx = [indices.popleft() for i in range(batch_size)]

        #t_start1 = time.time()
        
        ntotnode=tr_nvertex[batch_idx].sum()
        train_x=np.zeros((ntotnode,nfeat))

        train_x=tr_data[running_total*batch_size*912:(1+running_total)*batch_size*912,:]
        train_y = train_labels[running_total*batch_size:(1+running_total)*batch_size]

        # size0=0
        # size2=0
        # for i in range(batch_size):
        #     # take current data Laplacian , in this case all laplacian are same
        #     # normally i tmust be 
        #     # L0 = L[batch_idx][i][0].tocoo()
        #     # L2 = L[batch_idx][i][2].tocoo()
        #     L0 = L[0].tocoo()
        #     L2 = L[2].tocoo()
        #     size0+=L0.col.shape[0]
        #     size2+=L2.col.shape[0]

        # col0=np.zeros((size0),dtype=np.int32)
        # row0=np.zeros((size0),dtype=np.int32)
        # data0=np.zeros((size0))

        # col2=np.zeros((size2),dtype=np.int32)
        # row2=np.zeros((size2),dtype=np.int32)
        # data2=np.zeros((size2))

        

        # pid0=0
        # pid2=0
        # pid22=0
        #pidd=0
        #for i in range(batch_size):
        #     # take current data Laplacian , in this case all laplacian are same
        #     # normally i tmust be 
        #     # L0 = L[batch_idx][i][0].tocoo()
        #     # L2 = L[batch_idx][i][2].tocoo()
        #     L0 = L[0].tocoo()
        #     L2 = L[2].tocoo()

        #     a=np.where(tr_index==batch_idx[i])[0]
        #     #train_x= np.vstack((train_x,tr_data[a,:]))
        #     train_x[pidd:pidd+a.shape[0],:]=tr_data[a,:]
            

        #     # col0= np.hstack((col0,L0.col+pid0))
        #     # row0= np.hstack((row0,L0.row+pid0))
        #     # data0= np.hstack((data0,L0.data))
        #     col0[pid0:pid0+L0.col.shape[0]]=L0.col+pidd
        #     row0[pid0:pid0+L0.col.shape[0]]=L0.row+pidd
        #     data0[pid0:pid0+L0.col.shape[0]]=L0.data
        #     pid0+=L0.col.shape[0]

        #     # col2= np.hstack((col2,L2.col+pid2))
        #     # row2= np.hstack((row2,L2.row+pid2))
        #     # data2= np.hstack((data2,L2.data))
        #     col2[pid2:pid2+L2.col.shape[0]]=L2.col+pid22
        #     row2[pid2:pid2+L2.col.shape[0]]=L2.row+pid22
        #     data2[pid2:pid2+L2.col.shape[0]]=L2.data
        #     pid2+=L2.col.shape[0]    
        #     pid22+=L2.shape[0]   

        #     pidd+=a.shape[0]      

        #t_stop1 = time.time() - t_start1

        L2=coo_matrix((data2, (row2, col2)))
        L0=coo_matrix((data0, (row0, col0)))

        

        #train_y = train_labels[batch_idx]

        train_x = Variable( torch.FloatTensor(train_x).type(dtypeFloat) , requires_grad=False) 
        train_y = train_y.astype(np.int64)
        train_y = torch.LongTensor(train_y).type(dtypeLong)
        train_y = Variable( train_y , requires_grad=False) 
            
        # Forward 
        y = net.forward(train_x, dropout_value, L0,L2)
        loss = net.loss(y,train_y,l2_regularization) 
        #loss_train = loss.data[0]
        loss_train = loss.item()
        
        # Accuracy
        acc_train = net.evaluation(y,train_y.data)
        
        # backward
        loss.backward()
        
        # Update 
        global_step += batch_size # to update learning rate
        optimizer.step()
        optimizer.zero_grad()
        
        # loss, accuracy
        running_loss += loss_train
        running_accuray += acc_train
        running_total += 1
        
        #t_stop2 = time.time() - t_start1
        #print(t_stop1/(t_stop2-t_stop1))
        
        # print        
        if not running_total%100: # print every x mini-batches
            print('epoch= %d, i= %4d, loss(batch)= %.4f, accuray(batch)= %.2f' % (epoch+1, running_total*batch_size, loss_train, acc_train))
            print(time.time() - t_start)
       
    # print 
    t_stop = time.time() - t_start
    print('epoch= %d, loss(train)= %.3f, accuracy(train)= %.3f, time= %.3f, lr= %.5f' % 
          (epoch+1, running_loss/running_total, running_accuray/running_total, t_stop, lr))
 

    # update learning rate 
    lr = global_lr * pow( decay , float(global_step// decay_steps) )
    optimizer = net.update_learning_rate(optimizer, lr)
    
    
    # Test set
    running_accuray_test = 0
    running_total_test = 0
    indices_test = collections.deque()
    indices_test.extend(range(test_data.shape[0]))
    t_start_test = time.time()
    while len(indices_test) >= batch_size:
        batch_idx_test = [indices_test.popleft() for i in range(batch_size)]
        test_x, test_y = test_data[batch_idx_test,:], test_labels[batch_idx_test]
        test_x = Variable( torch.FloatTensor(test_x).type(dtypeFloat) , requires_grad=False) 
        y = net.forward(test_x, 0.0, L, lmax) 
        test_y = test_y.astype(np.int64)
        test_y = torch.LongTensor(test_y).type(dtypeLong)
        test_y = Variable( test_y , requires_grad=False) 
        acc_test = net.evaluation(y,test_y.data)
        running_accuray_test += acc_test
        running_total_test += 1
    t_stop_test = time.time() - t_start_test
    print('  accuracy(test) = %.3f %%, time= %.3f' % (running_accuray_test / running_total_test, t_stop_test))  
    


