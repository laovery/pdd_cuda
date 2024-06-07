function [V]=matrix_initiation(m,n)
    V=zeros(m,n);
    for k=1:m
        for s=1:n
            a=rand(1)+1i*rand(1);
            V(k,s)=a/abs(a);
        end
    end
end