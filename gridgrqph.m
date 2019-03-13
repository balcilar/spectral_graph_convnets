function A=gridgrqph(nv,ne)
nn=linspace(0,1,nv);
[X, Y]=meshgrid(nn);
X=X(:);
Y=Y(:);
XY=[X Y];
dist=zeros(nv*nv);
for i=1:nv*nv-1
    for j=i+1:nv*nv
        dist(i,j)=norm(XY(i,:)-XY(j,:));
        dist(j,i)=dist(i,j);
    end
end
A=zeros(nv*nv);

for i=1:nv*nv
    tmp=[dist(:,i) [1:nv*nv]'];
    tmp=sortrows(tmp,1);
    A(tmp(1:ne+1,2),i)=dist(tmp(1:ne+1,2),i);
    dst(i)=tmp(ne+1,1);
end

sigma2 = mean(dst)^2;
A = exp(- A.^2 / sigma2);
A(A==1)=0;

A=sparse((A+A')/2);

    
    





