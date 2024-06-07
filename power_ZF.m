function [p]=power_ZF(dl_user,ul_user,H_d,V,H1,H2,p_BS,p_u,I)
    
    b=zeros(1,dl_user);
    for k=1:dl_user
        for n=1:ul_user
            b(k)=b(k)+p_u(n)*abs(H1(k,n)^2); %power received by DL user
        end
    end
    
    cvx_begin quiet
    variable p(dl_user) 
    
    f1=0;
    for k=1:dl_user
        f1=f1+log(1+p(k)*(real(H_d(k,:)*V(:,k)))^2+b(k));
        %f2=f2+quad_form(V(:,k),h'*h/p2);
        %f3=f3+quad_form(V(:,k),eye(size(A,1)));
    end  
    maximize f1;
    subject to
        sum(p)<=p_BS;
        abs(H2*V).^2*p/I<=1;
        p>=0;
%       y1: f3<=p1;
%       y2: f2<=1;
    cvx_end
     
end