clc;
iteration=200; 
%%  sum rate VS resolution of phase shifters
y2=zeros(3,7);
for n=2:8
    SE2=zeros(iteration,3);
    for k=1:iteration
        [SE2(k,:),~]=FD_NRU2(n,16,8,60); 
    end
    y2(:,n-1)=sum(SE2)/iteration;
end
figure (3)
x=(2:8);
plot(x,real(y2(1,:)),'kx-');
set(gca,'XTick',(2:1:8));
set(gca,'YTick',(0:5:30));
hold on;
plot(x,real(y2(2,:)),'g^-');
plot(x,real(y2(3,:)),'rp-');
legend('HD-ZF','1-bit','infinite bit');
xlabel('user pair numbers');
ylabel('spectral efficiency: bit/ Hz');
grid on;