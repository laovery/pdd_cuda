clc;
iteration=100; 

%% user number VS sum rate
y1=zeros(4,8);h1=zeros(4,8);
%y1=y2;h1=h2;
for n=1:8
    SE=zeros(iteration,4);
    for k=1:iteration 
        SE(k,:)=FD_NRU(n*2,32,8,60); 
        disp(['2*',num2str(n),' users',',',num2str(k),'-th iterations']);
    end
    y1(:,n)=sum(SE)/iteration;
end
figure (1)
x=2:2:16;
plot(x,real(y1(1,:)),'kx-');
set(gca,'XTick',(2:2:16));
set(gca,'YTick',(0:20:100));
hold on;
plot(x,real(y1(2,:)),'g^-');
plot(x,real(y1(3,:)),'bs-');
plot(x,real(y1(4,:)),'rp-');
legend('FD+ZF','HD+PDD','FD+PDD','FD OPT');
xlabel('the number of users');
ylabel('Spectral Efficiency: bit/ Hz');
grid on;

% h1=10*log10(h1/(1e-3));
% figure (2)
% x=2:2:16;
% plot(x,real(h1(1,:)),'kx-');
% set(gca,'XTick',(2:2:16));
% set(gca,'YTick',(-100:10:-40));
% hold on;
% plot(x,real(h1(2,:)),'g^-');
% plot(x,real(h1(3,:)),'bs-');
% plot(x,real(h1(4,:)),'rp-');
% legend('HD-ZF','FD-ZF','PDD-DL','PDD-FD','baseline');
% xlabel('user pair numbers');
% ylabel('Iterference to WiGig link: dBm');
% grid on;

