function [V]=block_4(A,B,p1)
    %optimization problem======================================================
    %min  Tr(V'*A*V)-2Re{Tr(V'*B)}
    %s.t. Tr(V'*V)<=d
    %     Tr(V'*C*V)<=e 
    %==========================================================================

    m=size(A,1);
    n=size(B,2);
    f1=0;%f2=0;f3=0;
    cvx_begin quiet
    variable V(m,n) complex
    dual variable y1 
    dual variable y2
    for k=1:n
        f1=f1+quad_form(V(:,k),A);
        %f2=f2+quad_form(V(:,k),h'*h/p2);
        %f3=f3+quad_form(V(:,k),eye(size(A,1)));
    end  
    minimize (f1-2*real(trace(V'*B)));
    subject to
      y1: norm(V,'fro')<= sqrt(p1);
%       y1: f3<=p1;
%       y2: f2<=1;
    cvx_end
%     f0=trace(V'*A*V)-2*real(trace(V'*B));
%     y2=y2/p2;
end