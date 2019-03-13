function y=specCNNLayer(x,L,W,B)

[n, f, b]=size(x);

nfo=size(W,2);
K=size(W,1)/f;

yx=reshape(x,[n f*b]);
x0=yx;

x1=L*x0;
yx(:,:,2)=x1;
for i=3:K
    x2 = 2 * (L*x1) - x0;  
    yx(:,:,i)=x2;
    x0=x1;
    x1=x2;
end

yx=reshape(yx,n,f,b,K);
yx=permute(yx,[1 3 2 4]);
yx=reshape(yx,n,b, f*K);
yx=reshape(yx,n*b, f*K);


y=yx*W+B;
y=reshape(y,[n b nfo]);
y=permute(y,[1 3 2]);