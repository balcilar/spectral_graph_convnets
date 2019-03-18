import numpy as np
from lib.grid_graph import grid_graph
from lib.coarsening import coarsen
from lib.coarsening import lmax_L
from lib.coarsening import perm_data
from lib.coarsening import rescale_L
from scipy import sparse
import os
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import pdb #pdb.set_trace()
import collections
import time
import numpy as np
import tensorflow as tf
import pickle

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


def bspline_basis(K, x, degree=3):
    """
    Return the B-spline basis.

    K: number of control points.
    x: evaluation points
       or number of evenly distributed evaluation points.
    degree: degree of the spline. Cubic spline by default.
    """
    if np.isscalar(x):
        x = np.linspace(0, 1, x)

    # Evenly distributed knot vectors.
    kv1 = x.min() * np.ones(degree)
    kv2 = np.linspace(x.min(), x.max(), K-degree+1)
    kv3 = x.max() * np.ones(degree)
    kv = np.concatenate((kv1, kv2, kv3))

    # Cox - DeBoor recursive function to compute one spline over x.
    def cox_deboor(k, d):
        # Test for end conditions, the rectangular degree zero spline.
        if (d == 0):
            return ((x - kv[k] >= 0) & (x - kv[k + 1] < 0)).astype(int)

        denom1 = kv[k + d] - kv[k]
        term1 = 0
        if denom1 > 0:
            term1 = ((x - kv[k]) / denom1) * cox_deboor(k, d - 1)

        denom2 = kv[k + d + 1] - kv[k + 1]
        term2 = 0
        if denom2 > 0:
            term2 = ((-(x - kv[k + d + 1]) / denom2) * cox_deboor(k + 1, d - 1))

        return term1 + term2

    # Compute basis for each point
    basis = np.column_stack([cox_deboor(k, degree) for k in range(K)])
    basis[-1,-1] = 1
    return basis



class my_sparse_mm(torch.autograd.Function):
    """
    Implementation of a new autograd function for sparse variables, 
    called "my_sparse_mm", by subclassing torch.autograd.Function 
    and implementing the forward and backward passes.
    """
    
    def forward(self, W, x):  # W is SPARSE
        self.save_for_backward(W, x)
        y = torch.mm(W, x)
        return y
    
    def backward(self, grad_output):
        W, x = self.saved_tensors 
        grad_input = grad_output.clone()
        grad_input_dL_dW = torch.mm(grad_input, x.t()) 
        grad_input_dL_dx = torch.mm(W.t(), grad_input )
        return grad_input_dL_dW, grad_input_dL_dx
    
    
class Graph_ConvNet_LeNet5(nn.Module):
    
    def __init__(self, net_parameters):
        
        print('Graph ConvNet: LeNet5')
        
        super(Graph_ConvNet_LeNet5, self).__init__()
        
        # parameters
        D, DFeat,CL1_F, CL1_K, CL2_F, CL2_K, FC1_F, FC2_F = net_parameters
        FC1Fin = CL2_F*(D//16)
        
        # graph CL1
        self.cl1 = nn.Linear(CL1_K*DFeat, CL1_F) 
        #self.cl1 = nn.Linear(CL1_K, DFeat*CL1_F)

        Fin = CL1_K; Fout = CL1_F;
        scale = np.sqrt( 2.0/ (Fin+Fout) )
        self.cl1.weight.data.uniform_(-scale, scale)
        self.cl1.bias.data.fill_(0.0)
        self.CL1_K = CL1_K; self.CL1_F = CL1_F; 
        
        # graph CL2
        self.cl2 = nn.Linear(CL2_K*CL1_F, CL2_F) 
        #self.cl2 = nn.Linear(CL2_K, CL1_F*CL2_F) 

        Fin = CL2_K*CL1_F; Fout = CL2_F;
        scale = np.sqrt( 2.0/ (Fin+Fout) )
        self.cl2.weight.data.uniform_(-scale, scale)
        self.cl2.bias.data.fill_(0.0)
        self.CL2_K = CL2_K; self.CL2_F = CL2_F; 

        # FC1
        self.fc1 = nn.Linear(FC1Fin, FC1_F) 
        Fin = FC1Fin; Fout = FC1_F;
        scale = np.sqrt( 2.0/ (Fin+Fout) )
        self.fc1.weight.data.uniform_(-scale, scale)
        self.fc1.bias.data.fill_(0.0)
        self.FC1Fin = FC1Fin
        
        # FC2
        self.fc2 = nn.Linear(FC1_F, FC2_F)
        Fin = FC1_F; Fout = FC2_F;
        scale = np.sqrt( 2.0/ (Fin+Fout) )
        self.fc2.weight.data.uniform_(-scale, scale)
        self.fc2.bias.data.fill_(0.0)

        # nb of parameters
        nb_param = CL1_K* CL1_F + CL1_F          # CL1
        nb_param += CL2_K* CL1_F* CL2_F + CL2_F  # CL2
        nb_param += FC1Fin* FC1_F + FC1_F        # FC1
        nb_param += FC1_F* FC2_F + FC2_F         # FC2
        print('nb of parameters=',nb_param,'\n')
        
        
    def init_weights(self, W, Fin, Fout):

        scale = np.sqrt( 2.0/ (Fin+Fout) )
        W.uniform_(-scale, scale)

        return W

    def filter_in_fourier(self, x, L, Fout, K, U, W):
        # TODO: N x F x M would avoid the permutations
        N, M, Fin = x.size()
        N, M, Fin = int(N), int(M), int(Fin)
        #x = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x=x.permute(1,2,0).contiguous()

        # Transform to Fourier domain
        #x = tf.reshape(x, [M, Fin*N])  # M x Fin*N
        x = x.view([M, Fin*N])

        #x = tf.matmul(U, x)  # M x Fin*N
        x=torch.mm(torch.from_numpy(U).float().cuda(), x)

        #x = tf.reshape(x, [M, Fin, N])  # M x Fin x N
        x = x.view([M, Fin, N])
        # Filter
        #x = tf.matmul(W, x)  # for each feature 
        x=torch.bmm(W, x)

        x = x.permute(2,1,0).contiguous()  # N x Fout x M
        x = x.view([N*Fout, M])  # N*Fout x M
        # Transform back to graph domain
        x = torch.mm(x, torch.from_numpy(U).float().cuda())  # N*Fout x M
        x = x.view( [N, Fout, M])  # N x Fout x M
        return x.permute(0,2,1).contiguous()  # N x M x Fout

    def spline_new(self, x,cl, L,lmax, Fout, K):
        N, M, Fin = x.size()
        N, M, Fin = int(N), int(M), int(Fin)

        # Fourier basis
        lamb, U = np.linalg.eigh(L.toarray())        
        U=U.T
        # Spline basis
        B = bspline_basis(K, lamb, degree=3)  # M x K
        
        # Compose linearly Fin features to get Fout features        
        xx = cl(torch.from_numpy(B).float().cuda())                            
        xx = xx.view([M, Fout, Fin]) 

        xxx= self.filter_in_fourier(x[0,:,:].unsqueeze(0), L, Fout, K, U, xx)
        for i in range(1,N):
            xxx2= self.filter_in_fourier(x[i,:,:].unsqueeze(0), L, Fout, K, U, xx)
            xxx=torch.cat((xxx, xxx2), 0)

        #xxx= self.filter_in_fourier(x, L, Fout, K, U, xx)
        return xxx

    def spline(self, x,cl, L,lmax, Fout, K):
        N, M, Fin = x.size()
        N, M, Fin = int(N), int(M), int(Fin)
        # Fourier basis
        lamb, U = np.linalg.eigh(L.toarray())        
        U=U.T
        # Spline basis
        B = bspline_basis(K, lamb, degree=3)  # M x K
        
        # Compose linearly Fin features to get Fout features        
        xx = cl(torch.from_numpy(B).float().cuda())                            
        xx = xx.view([M, Fout, Fin])             

        xxx= self.filter_in_fourier(x, L, Fout, K, U, xx)
        return xxx

    def graph_conv_cheby(self, x, cl, L, lmax, Fout, K):

        # parameters
        # B = batch size
        # V = nb vertices
        # Fin = nb input features
        # Fout = nb output features
        # K = Chebyshev order & support size
        B, V, Fin = x.size(); B, V, Fin = int(B), int(V), int(Fin) 

        # rescale Laplacian
        #lmax = lmax_L(L)
        #L = rescale_L(L, lmax) 
        
        # convert scipy sparse matric L to pytorch
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col)).T 
        indices = indices.astype(np.int64)
        indices = torch.from_numpy(indices)
        indices = indices.type(torch.LongTensor)
        L_data = L.data.astype(np.float32)
        L_data = torch.from_numpy(L_data) 
        L_data = L_data.type(torch.FloatTensor)
        L = torch.sparse.FloatTensor(indices, L_data, torch.Size(L.shape))
        L = Variable( L , requires_grad=False)
        if torch.cuda.is_available():
            L = L.cuda()
        
        # transform to Chebyshev basis
        x0 = x.permute(1,2,0).contiguous()  # V x Fin x B
        x0 = x0.view([V, Fin*B])            # V x Fin*B
        x = x0.unsqueeze(0)                 # 1 x V x Fin*B
        
        def concat(x, x_):
            x_ = x_.unsqueeze(0)            # 1 x V x Fin*B
            return torch.cat((x, x_), 0)    # K x V x Fin*B  
             
        if K > 1: 
            x1 = my_sparse_mm()(L,x0)              # V x Fin*B
            x = torch.cat((x, x1.unsqueeze(0)),0)  # 2 x V x Fin*B
        for k in range(2, K):
            x2 = 2 * my_sparse_mm()(L,x1) - x0  
            x = torch.cat((x, x2.unsqueeze(0)),0)  # M x Fin*B
            x0, x1 = x1, x2  
        
        x = x.view([K, V, Fin, B])           # K x V x Fin x B     
        x = x.permute(3,1,2,0).contiguous()  # B x V x Fin x K       
        x = x.view([B*V, Fin*K])             # B*V x Fin*K
        
        # Compose linearly Fin features to get Fout features
        x = cl(x)                            # B*V x Fout  
        x = x.view([B, V, Fout])             # B x V x Fout
        
        return x

            
    # Max pooling of size p. Must be a power of 2.
    def graph_max_pool(self, x, p): 
        if p > 1: 
            x = x.permute(0,2,1).contiguous()  # x = B x F x V
            x = nn.MaxPool1d(p)(x)             # B x F x V/p          
            x = x.permute(0,2,1).contiguous()  # x = B x V/p x F
            return x  
        else:
            return x    
        
        
    def forward(self, x, d, L, lmax):
        
        # graph CL1
        #x = x.unsqueeze(2) # B x V x Fin=1  
        x = self.graph_conv_cheby(x, self.cl1, L[0], lmax[0], self.CL1_F, self.CL1_K)
        #x = self.spline(x, self.cl1, L[0], lmax[0], self.CL1_F, self.CL1_K)
        x = F.relu(x)
        x = self.graph_max_pool(x, 4)
        
        # graph CL2
        x = self.graph_conv_cheby(x, self.cl2, L[2], lmax[2], self.CL2_F, self.CL2_K)
        #x = self.spline(x, self.cl2, L[2], lmax[2], self.CL2_F, self.CL2_K)
        x = F.relu(x)
        x = self.graph_max_pool(x, 4)
        
        # FC1
        x = x.view(-1, self.FC1Fin)
        #x = x.view(-1, x.shape[0]*x.shape[1]*x.shape[2])
        x = self.fc1(x)
        x = F.relu(x)
        x  = nn.Dropout(d)(x)
        
        # FC2
        x = self.fc2(x)
        #x=self.fc2(torch.sum(x,0))   
        return x
        
        
    def loss(self, y, y_target, l2_regularization):
    
        loss = nn.CrossEntropyLoss()(y,y_target)

        l2_loss = 0.0
        for param in self.parameters():
            data = param* param
            l2_loss += data.sum()
           
        loss += 0.5* l2_regularization* l2_loss
            
        return loss
    
    
    def update(self, lr):
                
        update = torch.optim.SGD( self.parameters(), lr=lr, momentum=0.9 )
        
        return update
        
        
    def update_learning_rate(self, optimizer, lr):
   
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return optimizer

    
    def evaluation(self, y_predicted, test_l):
    
        _, class_predicted = torch.max(y_predicted.data, 1)
        return 100.0* (class_predicted == test_l).sum()/ y_predicted.size(0)
    


def readpah(dirname='datasets/PAH',ftype='train',mxnode=30):

    FF=[];GG=[];YY=[]

    for i in range(0,10):
        f=open(dirname+'/'+ftype+'set_'+str(i)+'.ds')
        line = f.readline()
        cnt = 0
        F=[];G=[];Y=[]
        while line:
            z=line.split()
            g=open(dirname+'/'+z[0] )
            tmp = g.readline()
            tmp = g.readline().split()
            nnode=int(tmp[0])
            nedge=int(tmp[1])
            X=np.zeros((nnode,2))
            A=np.zeros((nnode,nnode))
            # X=np.zeros((mxnode,2))
            # A=0.000*np.eye(mxnode)
            for j in range(nnode):
                tmp = g.readline().split()
                X[j,0]=float(tmp[0])
                X[j,1]=float(tmp[1])
            for j in range(nedge):
                tmp = g.readline().split()
                i1=int(tmp[0])-1
                j1=int(tmp[1])-1
                p1=int(tmp[2])
                p2=int(tmp[3])
                A[i1,j1]=p1
                A[j1,i1]=p2
            g.close()
            F.append(X)
            G.append(A)
            Y.append(int(z[1]))
            cnt+=1
            line = f.readline()

        FF.append(F)
        GG.append(G)
        YY.append(Y)
    return FF,GG,YY

if os.path.exists('pahdata.p'):
    f=open( "pahdata.p", "rb" )
    trainX=pickle.load(f)
    trainG=pickle.load(f)
    trainY=pickle.load(f)
    testX=pickle.load(f)
    testG=pickle.load(f)
    testY=pickle.load(f)
    f.close() 
else:

    trainX,trainG,trainY=readpah(dirname='datasets/PAH',ftype='train')
    testX,testG,testY=readpah(dirname='datasets/PAH',ftype='test')
    coarsening_levels = 4
    for i in range(0,len(trainX)):
        print(i)
        for j in range(0,len(trainX[i])):
            trainG[i][j], perm = coarsen(sparse.csr_matrix(trainG[i][j]), coarsening_levels)
            trainX[i][j] = perm_data(trainX[i][j].T, perm).T
    for i in range(0,len(testX)):
        for j in range(0,len(testX[i])):
            testG[i][j], perm = coarsen(sparse.csr_matrix(testG[i][j]), coarsening_levels)
            testX[i][j] = perm_data(testX[i][j].T, perm).T

    f=open( "pahdata.p", "wb" )
    pickle.dump(trainX,f)
    pickle.dump(trainG,f)
    pickle.dump(trainY,f)
    pickle.dump(testX,f)
    pickle.dump(testG,f)
    pickle.dump(testY,f)
    f.close()


kfold=0

train_data=np.zeros((0,32,2))
train_labels=np.zeros((0),dtype=np.int)
train_lap=[]
for i in range(len(trainX[kfold])):
    if trainX[kfold][i].shape[0]==32:
        train_data=np.vstack((train_data,np.expand_dims(trainX[kfold][i],0)))
        train_labels=np.hstack((train_labels,np.expand_dims(trainY[kfold][i],0)))
        train_lap.append(trainG[kfold][i])

test_data=np.zeros((0,32,2))
test_labels=np.zeros((0))
test_lap=[]
for i in range(len(testX[kfold])):
    if testX[kfold][i].shape[0]==32:
        test_data=np.vstack((test_data,np.expand_dims(testX[kfold][i],0)))
        test_labels=np.hstack((test_labels,np.expand_dims(testY[kfold][i],0)))
        test_lap.append(testG[kfold][i])

krr=np.sum(train_labels==1)==np.sum(train_labels==-1)

while not krr:
    for i in range(len(trainX[kfold])):
        if trainX[kfold][i].shape[0]==32 and trainY[kfold][i]==-1:
            train_data=np.vstack((train_data,np.expand_dims(trainX[kfold][i],0)))
            train_labels=np.hstack((train_labels,np.expand_dims(trainY[kfold][i],0)))
            train_lap.append(trainG[kfold][i])
            krr=np.sum(train_labels==1)==np.sum(train_labels==-1)
            if krr:
                break
        if krr:
            break

#train_labels
train_labels= (train_labels+1)/2
test_labels= (test_labels+1)/2




# network parameters
D = train_data.shape[1]
DFeat=train_data.shape[2]
CL1_F = 8
CL1_K = 25
CL2_F = 16
CL2_K = 25
FC1_F = 8
FC2_F = 2
net_parameters = [D,DFeat, CL1_F, CL1_K, CL2_F, CL2_K, FC1_F, FC2_F]


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
l2_regularization = 5e-3 
batch_size = 1
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
        batch_idx = [indices.popleft() for i in range(batch_size)]
        train_x, train_y = train_data[batch_idx,:], train_labels[batch_idx]
        train_x = Variable( torch.FloatTensor(train_x).type(dtypeFloat) , requires_grad=False) 
        train_y = train_y.astype(np.int64)
        train_y = torch.LongTensor(train_y).type(dtypeLong)
        train_y = Variable( train_y , requires_grad=False) 
            
        # Forward 
        y = net.forward(train_x, dropout_value, train_lap[batch_idx[0]],[1,1,1,1])  #
        #print(y)
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
        y = net.forward(test_x, 0.0, test_lap[batch_idx_test[0]], [1,1,1,1]) 
        test_y = test_y.astype(np.int64)
        test_y = torch.LongTensor(test_y).type(dtypeLong)
        test_y = Variable( test_y , requires_grad=False) 
        acc_test = net.evaluation(y,test_y.data)
        running_accuray_test += acc_test
        running_total_test += 1
    t_stop_test = time.time() - t_start_test
    print('  accuracy(test) = %.3f %%, time= %.3f' % (running_accuray_test / running_total_test, t_stop_test))  
    

        