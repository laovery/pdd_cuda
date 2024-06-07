function [sum_SE,R_u,R_d] = PDD_HD(ul_user,dl_user,H_u,H_d,H2,H3,N,I,p_user,p_BS,m_BS,n_BS,RF,n_user,I_W2B,I_W2U)
    %uplink power
    I1=I/ul_user;
    for k=1:ul_user
        p_u(k)=min(I1./abs(H3(k))^2,p_user);
    end
    
    V=zeros(m_BS,dl_user);V_RF=matrix_initiation(m_BS,RF);V_BB=zeros(RF,dl_user);%beam matirx at gNB
    Q=(1/sqrt(n_user))*ones(dl_user,n_user);%receiver
    p=10;lambda=zeros(m_BS,dl_user);%penalty factor and dual variable
    
    %MSE=============================================
    e_d=zeros(1,dl_user); %MSE
    a1=N*ones(1,dl_user)+diag(H_d*(V*V')*H_d')'+I_W2U;
    b1=diag(H_d*V);
    for k=1:dl_user
        e_d(k)=real(a1(k)*abs(Q(k))^(2))-2*real(Q(k)*b1(k))+1;
    end
   
    s1=0.8;%descent speed
    c1=1e-4;c2=1e-2;c3=1e-3;
    cv=c1;delta=c2;%constraint violation and augmented Lagrangian
    outer=1;
    
    while (cv>=c1)
        inner=1;f=[];
        while(delta>=c2 && inner < 50) 
            %update wighted factor=============================================
            sigma_d=(1/log(2))./e_d; %weighted factor
            %e1=objective_cal(p_u,V,V_RF,V_BB,lambda,sigma_u,sigma_d,W,Q,H_u,H_d,H_SI,H1,ul_user,dl_user,n_BS,N,p);

            
            %e2=objective_cal(p_u,V,V_RF,V_BB,lambda,sigma_u,sigma_d,W,Q,H_u,H_d,H_SI,H1,ul_user,dl_user,n_BS,N,p);

            %update V==================================================================
            h=0;
            for k=1:dl_user
                h=h+sigma_d(k)*H_d(k,:)'*(Q(k,:)'*Q(k,:))*H_d(k,:);
                Y(:,k)=sigma_d(k)*H_d(k,:)'*Q(k,:)';
            end
            X=h+(0.5/p)*eye(m_BS);
            Y=0.5/p*(V_RF*V_BB-p*lambda)+Y;
            V=block_1(X,Y,H2,p_BS,I);
            %e3=objective_cal(p_u,V,V_RF,V_BB,lambda,sigma_u,sigma_d,W,Q,H_u,H_d,H_SI,H1,ul_user,dl_user,n_BS,N,p);
            %[VV,pp]=primal_decomposition(X,a2,Y,b2,H2,H3,p_BS,p_user,I,ul_user,dl_user);
            %e7=objective_cal(p_u,V,V_RF,V_BB,lambda,sigma_u,sigma_d,W,Q,H_u,H_d,H_SI,H1,ul_user,dl_user,n_BS,N,p);
            
            %update V_BB===============================================================
            V_BB=pinv(V_RF)*(V+p*lambda);
            %e4=objective_cal(p_u,V,V_RF,V_BB,lambda,sigma_u,sigma_d,W,Q,H_u,H_d,H_SI,H1,ul_user,dl_user,n_BS,N,p);

            %update V_RF===============================================================
            A2=(V_BB*V_BB');B2=(V+p*lambda)*V_BB';
            V_RF=block_2(V_RF,A2,B2);
            %e5=objective_cal(p_u,V,V_RF,V_BB,lambda,sigma_u,sigma_d,W,Q,H_u,H_d,H_SI,H1,ul_user,dl_user,n_BS,N,p);

            %update Q============================================================
            e_d=zeros(1,dl_user); %MSE
            a1=N*ones(1,dl_user)+diag(H_d*(V*V')*H_d')'+I_W2U;
            b1=diag(H_d*V);
            for k=1:dl_user
                Q(k)=b1(k)/a1(k);
                e_d(k)=real(a1(k)*abs(Q(k))^(2))-2*real(Q(k)*b1(k))+1;
            end
            f(inner)=real(sigma_d*e_d'-sum(log2(sigma_d))+(0.5/p)*norm((V-V_RF*V_BB+p*lambda),'fro')^2);
            %E(inner,:)=[e1,e2,e3,e4,e5,f(inner)];
            if (inner>=2)
                delta=abs((f(inner)-f(inner-1))/f(inner));
            end
            inner=inner+1;
        end

        delta=c2;
        %c2=s1*c2;
        a3=zeros(1,dl_user);
        for k=1:dl_user
            a3(k)=norm(V(:,k)-V_RF*V_BB(:,k),inf);
        end
        cv=max(a3);
        
        if (cv>c3)
            p=s1*p;     
        else
            lambda=lambda+(1/p)*(V-V_RF*V_BB);
            c3=s1*cv;
        end

       %========calclate sum rate=========
        R_u=zeros(1,ul_user);
        R_d=zeros(1,dl_user);

        rec_u=zeros(n_BS);
        for n=1:ul_user
            rec_u=rec_u+p_u(n)*(H_u(:,n)*H_u(:,n)');
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
                b3=b3+abs(H_d(k,:)*V_RF*V_BB(:,n))^2;
            end
            rec_d=b3+N+I_W2U(k);
            sig_d=abs(H_d(k,:)*V_RF*V_BB(:,k))^2;
            R_d(k)=real(log2(rec_d/(rec_d-sig_d)));
        end

        CV(outer)=cv;
        SE(outer)=0.5*(sum(R_u)+sum(R_d));
        outer=outer+1;
    end
    
    sum_SE=real(SE(outer-1));
    %mean_SE=real(SE(outer-1)/user_num);
end