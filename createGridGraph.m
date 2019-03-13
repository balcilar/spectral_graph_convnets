function W=createGridGraph(n)

W=sparse(zeros(n*n,n*n));
for i=1:n
  for j=1:n
    p=sub2ind([n n],i,j);
    try
      p1=sub2ind([n n],i+1,j);
      W(p,p1)=1;
      W(p1,p)=1;
    catch
    end
    try
      p2=sub2ind([n n],i-1,j);
      W(p,p2)=1;
      W(p2,p)=1;
    catch
    end
    try
      p3=sub2ind([n n],i,j+1);
      W(p,p3)=1;
      W(p3,p)=1;
    catch
    end
    try
      p4=sub2ind([n n],i,j-1);
      W(p,p4)=1;
      W(p4,p)=1;
    catch
    end
  end
end 
% connect left column to right column top rows to bottom row
for i=1:n
  p1=sub2ind([n n],1,i);
  p2=sub2ind([n n],n,i);
  W(p1,p2)=1;
  W(p2,p1)=1;
  p1=sub2ind([n n],i,1);
  p2=sub2ind([n n],i,n);
  W(p1,p2)=1;
  W(p2,p1)=1;
end