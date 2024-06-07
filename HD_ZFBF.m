function [SE,R_u,R_d] = HD_ZFBF(ul_user,dl_user,H_u,H_d,H2,H3,N,p_user,p_BS,n_BS,I,I_W2B,I_W2U)
    
    R_u=zeros(1,ul_user);
    R_d=zeros(1,dl_user);
    
    %uplink
    I1=I/ul_user;
    for k=1:ul_user
        p_u(k)=min(I1./abs(H3(k))^2,p_user);
    end
    
    %downlink 
    I2=I/dl_user;
    V=pinv(H_d);
    V=sqrt(p_BS)*V/norm(V,'fro'); %maximum power
    for k=1:dl_user
        V(:,k)=sqrt(I2/abs(H2*V(:,k))^2)*V(:,k);
    end    

    rec_u=zeros(n_BS);
    for n=1:ul_user
        rec_u=rec_u+p_u(n)*H_u(:,n)*H_u(:,n)';
    end
    rec_u=rec_u+N*eye(n_BS)+I_W2B; %covariance matrix of signal received by BS
    for k=1:ul_user
        sig_u=p_u(k)*H_u(:,k)*H_u(:,k)';
        R_u(k)=log2(det(eye(n_BS)+sig_u/(rec_u-sig_u)));
        R_u(k)=real(R_u(k));
    end

    for k=1:dl_user
        b3=0;
        
        for n=1:dl_user
            b3=b3+abs(H_d(k,:)*V(:,n))^2;
        end
        rec_d=b3+N+I_W2U(k);
        sig_d=abs(H_d(k,:)*V(:,k))^2;
        R_d(k)=log2(rec_d/(rec_d-sig_d));
    end
    SE=0.5*sum(R_u)+0.5*sum(R_d);
    
end