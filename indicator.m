function [output]=indicator(input)
    m=size(input,1);n=size(input,2);
    x1=reshape(input',1,m*n);
    x2=zeros(1,m*n);
    x2(1)=0;
    for k=2:m*n
        if (x1(k)>x1(k-1))
            x2(k)=1;
        else
        x2(k)=0;
        end
    end
    output=reshape(x2,n,m)';
end