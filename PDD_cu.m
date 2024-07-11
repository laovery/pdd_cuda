function [sum_SE,r_u,r_d] = PDD_cu(ul_user,dl_user,H_SI,H_u,H_d,H1,H2,H3,N,I,p_user,p_BS,m_BS,n_BS,RF,n_user,I_W2B,I_W2U,alphe,temp)
    I_th=I/(2*ul_user);
    p_u=zeros(1,ul_user); %tx power single-antenna 
    V=zeros(m_BS,dl_user);V_RF=matrix_initiation(m_BS,RF);V_BB=zeros(RF,dl_user);%beam matirx at gNB
    %W=pinv(H_u)';
    W=sign(real(H_u));
    Q=(1/sqrt(n_user))*ones(dl_user,n_user);%receiver
    p=10;lambda=zeros(m_BS,dl_user);%penalty factor and dual variable
    
    %MSE=============================================
    e_u=zeros(1,ul_user);e_d=zeros(1,dl_user); %MSE
    A1=zeros(n_BS);
    for k=1:ul_user
        A1=A1+p_u(k)*(H_u(:,k)*H_u(:,k)')+H_SI*(V(:,k)*V(:,k)')*H_SI';
    end
    A1=N*eye(n_BS)+A1+I_W2B; 
    a1=N*ones(1,dl_user)+diag(H_d*(V*V')*H_d')'+p_u*abs(H1').^(2)+I_W2U;
    B1=H_u*diag(sqrt(p_u)); b1=diag(H_d*V);
    for k=1:ul_user
        e_u(k)=real(W(:,k)'*A1*W(:,k))-2*real(W(:,k)'*B1(:,k))+1;
    end
    for k=1:dl_user
        e_d(k)=real(a1(k)*abs(Q(k))^(2))-2*real(Q(k)*b1(k))+1;
    end
   
    s1=0.8;%descent speed
    c1=1e-4;c2=1e-2;c3=1e-3;
    cv=c1;delta=c2;%constraint violation and augmented Lagrangian
    outer=1;
    
    while (cv>=c1)
        inner=1;f=[];E=[];
        while(delta>=c2 && inner <30)   
            %if inner==1 && outer ==1
            %update wighted factor=============================================
            sigma_u=(1/log(2))./e_u;sigma_d=(1/log(2))./e_d; %weighted factor
            %e1=objective_cal(p_u,V,V_RF,V_BB,lambda,sigma_u,sigma_d,W,Q,H_u,H_d,H_SI,H1,ul_user,dl_user,n_BS,N,p,I_W2B,I_W2U);

            %update UL power===========================================================
            a2=zeros(1,ul_user);b2=zeros(1,ul_user);
            for k=1:ul_user
                a3=0;a4=0;
                for s=1:ul_user
                    a3=a3+sigma_u(s)*W(:,s)'*(H_u(:,k)*H_u(:,k)')*W(:,s);
                end
                for s=1:dl_user
                    a4=a4+sigma_d(s)*Q(s,:)*(H1(s,k)*H1(s,k)')*Q(s,:)';
                end
                a2(k)=a3+a4;
                b2(k)=real(sigma_u(k)*(W(:,k)'*H_u(:,k)));
                if (b2(k)>=0)
                    p_u(k)=min([p_user,abs((b2(k)/a2(k)))^(2),I_th/(H3(k)*H3(k)')]);
                else
                    p_u(k)=0;
                end
            end
            %e2=objective_cal(p_u,V,V_RF,V_BB,lambda,sigma_u,sigma_d,W,Q,H_u,H_d,H_SI,H1,ul_user,dl_user,n_BS,N,p,I_W2B,I_W2U);

            %update V==================================================================

            w=0;h=0;
            for k=1:ul_user
                w=w+W(:,k)*W(:,k)'*sigma_u(k);
            end 
            for k=1:dl_user
                h=h+sigma_d(k)*H_d(k,:)'*(Q(k,:)'*Q(k,:))*H_d(k,:);
                Y(:,k)=sigma_d(k)*H_d(k,:)'*Q(k,:)';
            end
            X=H_SI'*w*H_SI+h+(0.5/p)*eye(m_BS);
            Y=0.5/p*(V_RF*V_BB-p*lambda)+Y;

            V = inv(X+alphe*eye(32)) * Y;

            %V = inv(X + 0.1*eye(32)) * Y;
            %e3=objective_cal(p_u,V,V_RF,V_BB,lambda,sigma_u,sigma_d,W,Q,H_u,H_d,H_SI,H1,ul_user,dl_user,n_BS,N,p,I_W2B,I_W2U);
            %[VV,pp]=primal_decomposition(X,a2,Y,b2,H2,H3,p_BS,p_user,I,ul_user,dl_user);
            %e7=objective_cal(p_u,V,V_RF,V_BB,lambda,sigma_u,sigma_d,W,Q,H_u,H_d,H_SI,H1,ul_user,dl_user,n_BS,N,p);

            %update V_BB===============================================================
            V_BB=pinv(V_RF)*(V+p*lambda);
            %e4=objective_cal(p_u,V,V_RF,V_BB,lambda,sigma_u,sigma_d,W,Q,H_u,H_d,H_SI,H1,ul_user,dl_user,n_BS,N,p,I_W2B,I_W2U);

            %update V_RF===============================================================
            A2=(V_BB*V_BB');B2=(V+p*lambda)*V_BB';
            V_RF=block_2(V_RF,A2,B2);
            %e5=objective_cal(p_u,V,V_RF,V_BB,lambda,sigma_u,sigma_d,W,Q,H_u,H_d,H_SI,H1,ul_user,dl_user,n_BS,N,p,I_W2B,I_W2U);

            %update W and Q============================================================
            e_u=zeros(1,ul_user);e_d=zeros(1,dl_user); %MSE
            A1=zeros(n_BS);
            for k=1:ul_user
                A1=A1+p_u(k)*(H_u(:,k)*H_u(:,k)')+H_SI*(V(:,k)*V(:,k)')*H_SI';
            end
            A1=N*eye(n_BS)+A1+I_W2B; 
            a1=N*ones(1,dl_user)+diag(H_d*(V*V')*H_d')'+p_u*abs(H1').^(2)+I_W2U;
            B1=H_u*diag(sqrt(p_u)); b1=diag(H_d*V);
            for k=1:ul_user
                W(:,k)=pinv(A1)*B1(:,k);
                e_u(k)=real(W(:,k)'*A1*W(:,k))-2*real(W(:,k)'*B1(:,k))+1;
            end
            for k=1:dl_user
                Q(k)=b1(k)/a1(k);
                e_d(k)=real(a1(k)*abs(Q(k))^(2))-2*real(Q(k)*b1(k))+1;
            end
        %else
            [p_u,V,V_RF,V_BB,e_u,e_d,W,Q] = cu_mex(ul_user,RF,H_SI,H_u,H_d,H1,H2,H3,N,I,p_user,p_BS,m_BS,n_BS,I_W2B,I_W2U,p_u,V,V_RF,V_BB,W,Q,p,lambda,e_u,e_d);
            
            [r_u,r_d] = cal_e(ul_user,dl_user,n_BS,N,p_u,V,V_RF,V_BB,H_SI,H_u,H_d,H1,I_W2B,I_W2U);
            disp(["Sum Rate: ",r_u+r_d]);
        %end
            
            f(inner)=real(sigma_u*e_u'-sum(log2(sigma_u))+sigma_d*e_d'-sum(log2(sigma_d))+(0.5/p)*norm((V-V_RF*V_BB+p*lambda),'fro')^2);
            %E(inner,:)=[e1,e2,e3,e4,e5,f(inner)];
            if (inner>=2)
                delta=abs((f(inner)-f(inner-1))/f(inner));
            end
            inner=inner+1;
            
            break
        end

%         delta=c2;
%         
%         a5=zeros(1,dl_user);
%         for k=1:dl_user
%             a5(k)=norm(V(:,k)-V_RF*V_BB(:,k),inf);
%         end
%         cv=max(a5);
%         
%         I1=0; %interference to WiGig
%         for k=1:dl_user
%             I1=I1+H2*(V_RF*V_BB(:,k)*V_BB(:,k)'*V_RF')*H2';
%         end
%         I1=real(I1+p_u*abs(H3').^(2));
%         
%         if (cv>c3)
%             p=s1*p;     
%         else
%             lambda=lambda+(1/p)*(V-V_RF*V_BB);
%             c3=s1*cv;
%         end
% 
%        %========calclate sum rate=========
%         R_u=zeros(1,ul_user);
%         R_d=zeros(1,dl_user);
% 
%         rec_u=zeros(n_BS);
%         for n=1:dl_user
%             SI=H_SI*(V(:,n)*V(:,n)')*H_SI';
%         end
%         for n=1:ul_user
%             rec_u=rec_u++p_u(n)*(H_u(:,n)*H_u(:,n)');
%         end
%         rec_u=rec_u+SI+N*eye(n_BS)+I_W2B; %covariance matrix of signal received by BS
%         for k=1:ul_user
%             sig_u=p_u(k)*H_u(:,k)*H_u(:,k)';
%             R_u(k)=log2(det(eye(n_BS)+sig_u/(rec_u-sig_u)));
%             R_u(k)=real(R_u(k));
%         end
% 
%         for k=1:dl_user
%             b3=0;b4=0;
%             for n=1:dl_user
%                 b3=b3+abs(H_d(k,:)*V_RF*V_BB(:,n))^2;
%             end
%             for n=1:ul_user
%                 b4=b4+p_u(n)*abs(H1(k,n)^2); %power received by DL user
%             end
%             rec_d=b3+b4+N+I_W2U(k);
%             sig_d=abs(H_d(k,:)*V_RF*V_BB(:,k))^2;
%             R_d(k)=real(log2(rec_d/(rec_d-sig_d)));
%         end
%         
%         r_u=sum(R_u);
%         r_d=sum(R_d);
%         %[r_u,r_d] = cal_e(ul_user,dl_user,n_BS,N,p_u,V,V_RF,V_BB,H_SI,H_u,H_d,H1,I_W2B,I_W2U);
%         CV(outer)=cv;
%         SE(outer)=r_u+r_d;
%         outer=outer+1;
        break
    end
    
    
    sum_SE=r_u+r_d;

    
    %mean_SE=real(SE(outer-1)/user_num);
end