function [SE,R_u,R_d] = FD_ZFBF(ul_user,dl_user,H_SI,H_u,H_d,H1,H2,H3,N,p_user,p_BS,n_BS,I,I_W2B,I_W2U)

    %uplink
    I1=I/(dl_user+ul_user);
    for k=1:ul_user
        p_u(k)=min(I1./abs(H3(k))^2,p_user);
    end
    
    %downlink 
    V=pinv(H_d);
    for k=1:dl_user
        V(:,k)=V(:,k)/norm(V(:,k),'fro');
    end    
    [p_d]=power_ZF(dl_user,ul_user,H_d,V,H1,H2,p_BS,p_u,I);
    V=V*diag(sqrt(p_d));
    
    R_u=zeros(1,ul_user);
    R_d=zeros(1,dl_user);

    rec_u=zeros(n_BS);
    for n=1:dl_user
        SI=H_SI*(V(:,n)*V(:,n)')*H_SI';
    end
    for n=1:ul_user
        rec_u=rec_u+p_u(n)*(H_u(:,n)*H_u(:,n)');
    end
    rec_u=rec_u+SI+N*eye(n_BS)+I_W2B; %covariance matrix of signal received by BS
    for k=1:ul_user
        sig_u=p_u(k)*(H_u(:,k)*H_u(:,k)');
        R_u(k)=log2(det(eye(n_BS)+sig_u/(rec_u-sig_u)));
        R_u(k)=real(R_u(k));
    end

    for k=1:dl_user
        b3=0;b4=0;
        for n=1:dl_user
            b3=b3+abs(H_d(k,:)*V(:,n))^2;
        end
        for n=1:ul_user
            b4=b4+p_u(n)*abs(H1(k,n)^2); %power received by DL user
        end
        rec_d=b3+b4+N+I_W2U(k);
        sig_d=abs(H_d(k,:)*V(:,k))^2;
        R_d(k)=log2(rec_d/(rec_d-sig_d));
    end
    SE=sum(R_u)+sum(R_d);
    

end