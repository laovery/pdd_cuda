clc;
iteration=200; 

%% antenna number
y1=zeros(5,8);
%y1=y2;h1=h2;
for n=1:8
    res=zeros(1,5);
    res1 = zeros(1,5);
    for k=1:iteration 
        [res(1),res(2),res(3),res(4),res(5)] = FD_NRU(2*n,32,8,60);
        res1 = res1 + res; 
        disp([num2str(n),'-th point',',',num2str(k),'-th iterations']);
    end
    y1(:,n)=res1/iteration;
end

figure (1)
y2=y1(1:3,:);
y2=[[0;0;0],y2];
plot((0:8)*2,real(y2(1,:)),'ro-');
set(gca,'XTick',(0:4)*4);
set(gca,'YTick',(0:20:80));
hold on;
plot((0:8)*2,real(y2(2,:)),'b^-');
plot((0:8)*2,real(y2(3,:)),'gv-');
xlabel('Number of users');
ylabel('Sum Rate: bit/Hz');
legend('Sum Rate','UL','DL');
grid on;

figure(2)
%y3=y1(4:5,:);
plot((1:8)*2,real(y3(1,:)),'bv-');
axis([2,16,-100,-20]);
set(gca,'XTick',(0:4)*4);
set(gca,'YTick',(-100:40:-20));
hold on;
plot((1:8)*2,real(y3(2,:)),'g^-');
xlabel('Number of users');
ylabel('Power: dBm');
legend('SI','Inter-user');
grid on;



