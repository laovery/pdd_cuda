function [SE,R_u,R_d]=FD_NRU1(user_num,n_BS,RF,beta)    
   
    %% parameter setup===========================================================
    %clc;
    %clear;
    %global H_SI H_u H_d H1 H2 H3 N I p_max p0 R
    %deployment area
    R=50;
    %carrier frequency
    fc=60*1e9;
    c=3e8;
    %WiGig's receiver
    wifi_num=1;
    ag_w=unifrnd(0,2*pi,1,wifi_num);%angle of wigig
    wifi_d=unifrnd(R/2,R,1,wifi_num);
    x_wifi=wifi_d*cos(ag_w);  
    y_wifi=wifi_d*sin(ag_w);
    Wifi=[x_wifi;y_wifi];
    %WiGig's transmitter
    Wifi_tx=[x_wifi-8;y_wifi];
    %gNB's location and antennas
    x_BS=0;
    y_BS=0;
    m_BS=n_BS;
    %m_BS=16;n_BS=16;RF=8;
    %UE's location and antennas
    ul_user=ceil(user_num/2);
    dl_user=user_num-ul_user;
    %ul_user=4;dl_user=4;
    m_user=1;n_user=1;
    %user distribution
    %beamwidth=60;
    user1=unifrnd(-R,R,[2,ul_user]);
    user2=unifrnd(-R,R,[2,dl_user]);
    %user1=[-20,-20;20,0];user2=[20,20;20,-20];
    u1_x=user1(1,:);u1_y=user1(2,:);
    u2_x=user2(1,:);u2_y=user2(2,:);
    
    %topology depiction
%     figure(1)
%     for k=1:ul_user
%         axis([-R,R,-R,R]); 
%         plot(u1_x(k),u1_y(k),'ko'); 
%         text(u1_x(k)+1,u1_y(k)+1,num2str(k));
%         hold on;
%     end
%     for k=1:dl_user
%         plot(u2_x(k),u2_y(k),'bo'); 
%         text(u2_x(k)+1,u2_y(k)+1,num2str(k));
%         hold on;
%     end
%     plot(u1_x,u1_y,'ko','MarkerSize',8);
%     plot(u2_x,u2_y,'bo','MarkerSize',8);
%     plot(x_BS,y_BS,'r^','MarkerSize',10);
%     plot(x_wifi,y_wifi,'gx','MarkerSize',10);
%     %legend('UL user','DL user','gNB','WiGig');  
%     grid on;
%     hold off;
  

    %% channel matrix============================================================
    %self-interference matrix
    %beta=0; %self-interference cancellation
    %beta=80;
    Kdb=50; %Recian factor in dB form
    H_SI0=rice_matrix(Kdb,n_BS,m_BS);
    H_SI=sqrt(10^(-beta/10))*H_SI0;
    H_SI1=sqrt(10^(-(beta+15)/10))*H_SI0;
    H_SI2=sqrt(10^(-(beta+30)/10))*H_SI0;
%     H_SI3=sqrt(10^(-(beta+30)/10))*H_SI0;
%     H_SI4=sqrt(10^(-(beta+40)/10))*H_SI0;
    
    %uplink\downlink channel
    H_u=zeros(n_BS,m_user*ul_user);H_d=zeros(n_user*dl_user,m_BS);%channel matrix
    dis_u=sqrt(u1_x.^2 + u1_y.^2);
    dis_d=sqrt(u2_x.^2 + u2_y.^2);
    pl_u=10*log10((c/(4*pi*fc))^(2)./(dis_u.*dis_u));
    pl_d=10*log10((c/(4*pi*fc))^(2)./(dis_d.*dis_d));
    %pl=22.7+36.7*log10(d_user)+26*log10(fc/1e9);
    for k = 1:ul_user 
        H_u(:,(k-1)*m_user+1:k*m_user)=sqrt(10^(pl_u(k)/10))*mmWave_matrix(n_BS,m_user);
    end 
    for k = 1:dl_user 
        H_d((k-1)*n_user+1:k*n_user,:)=sqrt(10^(pl_d(k)/10))*mmWave_matrix(n_user,m_BS);
    end
    
    %inter_user channel from UL user s to DL user k
    H1=zeros(dl_user,ul_user);pl1=zeros(dl_user,ul_user);
    for k=1:dl_user
        for s=1:ul_user
                pl1(k,s)=10*log10((c/(4*pi*fc))^(2)/((u1_x(s)-u2_x(k))^(2)+(u1_y(s)-u2_y(k))^(2)));
                H1(k,s)=sqrt(10^(pl1(k,s)/10))*mmWave_matrix(n_user,m_user);
        end
    end

    %cellular-to-wifi
    %BS-to-wifi
    pl2=10*log10((c/(4*pi*fc))^(2)./(wifi_d.^(2)));
    H2=sqrt(10^(pl2/10))*mmWave_matrix(1,m_BS);

    %user-to-wifi 
    H3=zeros(m_user,ul_user);
    pl3=zeros(wifi_num,ul_user);
    for k=1:ul_user
        pl3(k)=10*log10((c/(4*pi*fc))^(2)/((x_wifi-u1_x(k))^(2)+(y_wifi-u1_y(k))^(2)));
        H3(k)=sqrt(10^(pl3(k)/10))*mmWave_matrix(m_user,1);
    end
    
    %wifi transmitter to cellular
    H5=zeros(1,dl_user);pl5=zeros(1,dl_user);
    pl4=10*log10((c/(4*pi*fc))^(2)/(Wifi_tx'*Wifi_tx))-20;
    H4=sqrt(10^(pl4/10))*mmWave_matrix(n_BS,1);
    for k=1:dl_user
        pl5(k)=10*log10((c/(4*pi*fc))^(2)/((x_wifi-8-u2_x(k))^(2)+(y_wifi-u2_y(k))^(2)));
        H5(k)=sqrt(10^(pl5(k)/10))*mmWave_matrix(n_user,1);
    end
    
    %% PDD algorithm=============================================================
    %initialization
    p_wifi=0.5;p_user=1;p_BS=10; %max tx power
    I_W2B=p_wifi*(H4*H4'); 
    I_W2U=p_wifi*abs((H5.*H5));
    I=1e-11;%interference to wifi
    N=1e-12;
    [SE1,~] = PDD_original(ul_user,dl_user,H_SI,H_u,H_d,H1,H2,H3,N,I,p_user,p_BS,m_BS,n_BS,RF,n_user,I_W2B,I_W2U);
    [SE2,~] = PDD_original(ul_user,dl_user,H_SI1,H_u,H_d,H1,H2,H3,N,I,p_user,p_BS,m_BS,n_BS,RF,n_user,I_W2B,I_W2U);
    [SE3,~] = PDD_original(ul_user,dl_user,H_SI2,H_u,H_d,H1,H2,H3,N,I,p_user,p_BS,m_BS,n_BS,RF,n_user,I_W2B,I_W2U);
%     [SE4,~] = PDD_original(ul_user,dl_user,H_SI3,H_u,H_d,H1,H2,H3,N,I,p_user,p_BS,m_BS,n_BS,RF,n_user,I_W2B,I_W2U);
%     [SE5,~] = PDD_original(ul_user,dl_user,H_SI4,H_u,H_d,H1,H2,H3,N,I,p_user,p_BS,m_BS,n_BS,RF,n_user,I_W2B,I_W2U);
    %[SE2,R_u2,R_d2] = PDD_original(ul_user,dl_user,H_SI1,H_u,H_d,H1,H2,H3,N,I,p_user,p_BS,m_BS,n_BS,RF,n_user,I_W2B,I_W2U);
%     [SE3,R_u3,R_d3,I3] = PDD_original(ul_user,dl_user,H_SI1,H_u,H_d,H1,H2,H3,N,I,p_user,p_BS,m_BS,n_BS,RF,n_user,I_W2B,I_W2U);
%     [SE4,R_u4,R_d4,I4] = PDD_original(ul_user,dl_user,H_SI2,H_u,H_d,H1,H2,H3,N,I,p_user,p_BS,m_BS,n_BS,RF,n_user,I_W2B,I_W2U);
    %[SE5,R_u5,R_d5,I5,I_th] = PDD_pro(ul_user,dl_user,H_SI,H_u,H_d,H1,H2,H3,N,I,p_user,p_BS,m_BS,n_BS,RF,n_user);
%     SE=SE1;I=I1;
    SE4=0;SE5=0;
    SE=[SE1,SE2,SE3,SE4,SE5];
    %R_u=sum(R_u1);R_d=sum(R_d1);
%    I=[I1,I2,I3,I4];
%     R_u=[R_u1;R_u2:R_u3;R_u4];
%     R_d=[R_d1;R_d2;R_d3;R_d4];
    %% stimulation result
%     figure(2)
%     x=1:length(SE);
%     plot(x,real(SE),'ko-');
%     hold on;
%     x=1:length(SE1);
%     plot(x,real(SE1),'go-');
%     x=1:length(SE2);
%     plot(x,real(SE2),'ro-');
%     legend('p=10','p=6','p=3');
%     title('convergence analysis');
%     xlabel('outer loop iterations');
%     ylabel('spectral efficiency');
%     
% 
%     figure(3)
%     x=1:length(CV);
%     semilogy(x,real(CV),'ko-');
%     x=1:length(CV1);
%     hold on;
%     semilogy(x,real(CV1),'go-');
%     x=1:length(CV2);
%     semilogy(x,real(CV2),'ro-');
%     legend('p=10','p=6','p=3');
%     title('convergence analysis');
%     xlabel('outer loop iterations');
%     ylabel('constraint violation');
%     hold on
end