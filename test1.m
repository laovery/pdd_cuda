clc;
iteration=200; 

%% SI ans user-interference
y1=zeros(2,13);
%y1=y2;h1=h2;
for n=1:13
    res=zeros(1,2);
    res1 = zeros(1,2);
    for k=1:iteration 
        [~,~,~,res(1),res(2)] = FD_NRU(8,32,8,35+5*n);
        res1 = res1 + res; 
        disp([num2str(n),'-th point',',',num2str(k),'-th iterations']);
    end
    y1(:,n)=res1/iteration;
end

figure (1)
plot((4:10)*10,real(y1(1,:)),'bv-');
set(gca,'XTick',(40:20:100));
set(gca,'YTick',(-120:40:0));
axis([40,100,-120,0]);
hold on;
plot((4:10)*10,real(y1(2,:)),'g^-');
xlabel('Self-interference cancellation: dB');
ylabel('Power: dBm');
legend('SI','inter-user');
grid on;



