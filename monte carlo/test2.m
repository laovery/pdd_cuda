clc;
iteration=100;
%%  sum rate VS SI
y3=zeros(5,8);
for n=1:8
    SE2=zeros(iteration,5);
    for k=1:iteration
        SE2(k,:)=FD_NRU1(2*n,32,8,15); 
        disp([num2str(2*n),' users',',',num2str(k),'-th iterations']);
    end
    y3(:,n)=sum(SE2)/iteration;
end
figure (4)
x=(2:2:16);
plot(x,real(y3(1,:)),'k>-');
set(gca,'XTick',(2:2:16));
set(gca,'YTick',(0:5:80));
hold on;
plot(x,real(y3(2,:)),'g^-');
plot(x,real(y3(3,:)),'bs-');
plot(x,real(y3(4,:)),'rp-');
plot(x,real(y3(5,:)),'m<-');
legend('10dB','20dB','30dB','40dB','50dB');
% xlabel('SI cancallation ');
% ylabel('spectral efficiency: bit/ Hz');
grid on;