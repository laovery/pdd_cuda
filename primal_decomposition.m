function [V,p,I_th]=primal_decomposition(A,A1,B,B1,H,H1,p0,p1,p2,ul_user)
    %=========================optimization problem=========================
    %min  f0(V)+f1(p)
    %s.t. Tr(VV')<=p0, p(i)<=p1
    %     |H0*V|^2+|sqrt(p)*H1|^2<=p2 (C1)
    %note:
    % f0(V)=trace{V'*A*V}-2Re{V'*B}
    % f1(p)=p*A1*p'-2Re{p*B1}
    %======================================================================
   
    p=zeros(1,ul_user);
    threshold=1e-2;
    delta=threshold;
    loop=1;%number of iterations
    
    
    I_th=p2/(ul_user*2)*[ul_user,ones(1,ul_user)];%initial interference allocation
    
    y=zeros(1,ul_user+1);%subgradient
    f=zeros(1,ul_user+1);%function value;
    
    while (delta>=threshold)
               
        %update each subproblem
        [V,y(1),f(1)]=block_1(A,B,H,p0,I_th(1));
        for k=1:ul_user
            if (B1(k)>=0)
                p(k)=real(min([p1,abs((B1(k)/A1(k)))^(2),I_th(k+1)/(H1(k)*H1(k)')]));
            else 
                p(k)=0;
            end
            if (p(k) == real(I_th(k+1)/(H1(k)*H1(k)')))
                %wrong
                y(k+1)=(real(B1(k))-p(k)*A1(k))/((H1(k)*H1(k)'*p(k)));
            else
                y(k+1)=0;
            end
            f(k+1)=p(k)*A1(k)-2*real(B1(k))*sqrt(p(k));
        end
        
        I_th=real(0.9*I_th+0.1*p2*y/sum(y));
        F(loop)=sum(f);
        if (loop>=2)
            delta=abs((F(loop)-F(loop-1))/F(loop-1));
        end
         loop=loop+1;
    end
        
%     I_th=0.99*I_th+0.01*p2/(ul_user+1)*ones(1,ul_user+1);
%     V=block_1(A,B,H,p0,I_th(1));
%     for k=1:ul_user
%         if (B1(k)>=0)
%             p(k)=min([p1,abs((B1(k)/A1(k)))^(2),I_th(k+1)/(H1(k)*H1(k)')]);
%         else 
%             p(k)=0.01*min([p1,I_th/(H1(k)*H1(k)')]);
%         end
%         
%     end

end