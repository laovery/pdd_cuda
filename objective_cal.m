function [f]=objective_cal(p_u,V,V_RF,V_BB,lambda,sigma_u,sigma_d,W,Q,H_u,H_d,H_SI,H1,ul_user,dl_user,n_BS,N,p,I_W2B,I_W2U)
    e_u=zeros(1,ul_user);e_d=zeros(1,dl_user); %MSE
    A1=zeros(n_BS);
    for k=1:ul_user
        A1=A1+p_u(k)*H_u(:,k)*H_u(:,k)'+H_SI*V(:,k)*V(:,k)'*H_SI';
    end
    A1=N*eye(n_BS)+A1+I_W2B; a1=N*ones(1,dl_user)+diag(H_d*(V*V')*H_d')'+p_u*abs(H1').^(2)+I_W2U;
    B1=H_u*diag(sqrt(p_u)); b1=diag(H_d*V);
    for k=1:ul_user
        e_u(k)=W(:,k)'*A1*W(:,k)-2*real(W(:,k)'*B1(:,k))+1;
    end
    for k=1:dl_user
        e_d(k)=a1(k)*abs(Q(k))^(2)-2*real(Q(k)*b1(k))+1;
    end
    f=sigma_u*e_u'+sigma_d*e_d'-sum(log2(sigma_u))-sum(log2(sigma_d))+(1/(2*p))*norm(V-V_RF*V_BB+p*lambda,'fro')^2;
end