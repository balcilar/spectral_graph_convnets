clear all
clc


[imgDataTrain, labelsTrain, imgDataTest, labelsTest] = prepareData;

A=createGridGraph(28);
%A=gridgrqph(28,8);
level=4;

[G, parents]= Coarsen(A,level);

% calculate basis
for i=1:level+1    
%     d = sum(G{i},2);
%     L{i} = diag(d)-G{i};

    D = diag(1./sqrt(sum(G{i},1)));    
    L{i}= eye(size(D,1)) - D * G{i} * D;    
    [u, v]=eig(full(L{i}));
    v=diag(v);
    lmax(i)=max(v);
    
    % normalized laplacian make max eigenvalue 1
    L{i}= 2*L{i}/lmax(i) - eye(size(D,1));
    [u1, v1]=eig(full(L{i}));
    v1=diag(v1);     
end



% take first 100 element as batch
bsize=1;
K=25;
nfo=32;

x=double(imgDataTrain(:,:,1,1:bsize));
x=reshape(x,size(x,1)*size(x,2),size(x,3),size(x,4));
% x(:,2,:)=255-x(:,1,:);
% x(:,3,:)=0;

% input consist of n node, f feature, b batch size (chunk of data)
[n, f, b]=size(x);

W=rand(K*f,nfo)-0;5;
B=rand(1,nfo)-0;5;

y=specCNNLayer(x,L{1},W,B);


nfo=64;
[n, f, b]=size(y);
W2=rand(K*f,nfo)-0;5;
B2=rand(1,nfo)-0;5;


y2=specCNNLayer(y,L{1},W2,B2);

            







