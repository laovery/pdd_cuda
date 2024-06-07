
    clear;
    %global H_SI H_u H_d H1 H2 H3 N I p_max p0 R
    %deployment area
    R=40;
    %carrier frequency
    fc=60*1e9;
    c=3e8;
    %WiGig's location
    wifi_num=1;
    ag_w=unifrnd(0,2*pi,1,wifi_num);%angle of wigig
    wifi_d=unifrnd(R/2,R,1,wifi_num);
    x_wifi=wifi_d*cos(ag_w);  
    y_wifi=wifi_d*sin(ag_w);
    Wifi=[x_wifi;y_wifi];
    x_wifi=[x_wifi,x_wifi+8]; 
    y_wifi=[y_wifi,y_wifi];
    %gNB's location and antennas
    x_BS=0;
    y_BS=0;
    %m_BS=n_BS;
    %m_BS=16;n_BS=16;RF=8;
    %UE's location and antennas
    %dl_user=ul_user;
    ul_user=15;dl_user=15;
    m_user=1;n_user=1;
    %user distribution
    beamwidth=60;
    user1=random_user(Wifi,beamwidth,ul_user,R);
    user2=random_user(Wifi,beamwidth,dl_user,R);
    %user1=[-20,-20;20,0];user2=[20,20;20,-20];
    u1_x=user1(1,:);u1_y=user1(2,:);
    u2_x=user2(1,:);u2_y=user2(2,:);
    
    %topology depiction
    figure(1)
    for k=1:ul_user
        axis([-R,R,-R,R]); 
        plot(u1_x(k),u1_y(k),'k*','MarkerSize',8); 
        hold on;
    end
    text(u1_x(k)+3,u1_y(k)+3,'上行用户','fontsize',10,'Interpreter','latex');
    for k=1:dl_user
        plot(u2_x(k),u2_y(k),'b*','MarkerSize',8); 
        hold on;
    end
    text(u2_x(k)+3,u2_y(k)+3,'下行用户','fontsize',10,'Interpreter','latex');
    plot(u1_x,u1_y,'k*','MarkerSize',8);
    plot(u2_x,u2_y,'b*','MarkerSize',8);
    plot(x_BS,y_BS,'r^','MarkerSize',10);
    text(x_BS+2,y_BS+2,'BS','fontsize',10,'Interpreter','latex');
    plot(x_wifi,y_wifi,'gx-','MarkerSize',8);
    text(min(x_wifi)-2,sum(y_wifi)/2-5,'WiGig link','fontsize',10,'Interpreter','latex');
    text(x_wifi(1)-6,y_wifi(1)+1,'发端','fontsize',10,'Interpreter','latex');
    text(x_wifi(2)+1,y_wifi(2)+1,'收端','fontsize',10,'Interpreter','latex');
    xlabel('Meters');
    ylabel('Meters');
    set(gca,'XTick',(-40:20:40));
    set(gca,'YTick',(-40:20:40));
    %legend('UL user','DL user','gNB','WiGig');  
    grid on;
    hold off;