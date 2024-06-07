function [V2]=block_2(V1,Q,P)
    %infinite phase shifters
    %optimization problem======================================================
    %min  Tr(V_RF'*V_RF*Q)-2RE{Tr(V_RF'*P)}
    %s.t. |V_RF(m,n)|=1
    %==========================================================================
    %f0=trace(V1*Q*V1')-2*real(trace(V1'*P));
    m=size(V1,1);n=size(V1,2);%p=1e-3;
    x=zeros(m,n);
    %f1=zeros(m,n);
    %while(p>=1e-3)
    for k=1:m
        for s=1:n
            x(k,s)=-V1(k,:)*Q(:,s)+P(k,s)+V1(k,s)*Q(s,s);
            V1(k,s)=x(k,s)/abs(x(k,s));
%           if real(x(k,s))>0
%             V1(k,s)=1;
%           else
%             V1(k,s)=-1;
%           end
          %f1(k,s)=trace(V1*(A*A')*V1')-2*real(trace(V1'*(B*A')));
        end
    end
%    f2=trace(V1*Q*V1')-2*real(trace(V1'*P));
    %   p=abs(f1-f0)/abs(f0);
    %   f0=f1;
    % end
    V2=V1;
end
