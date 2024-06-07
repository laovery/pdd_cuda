figure (1)
x=2:2:16;
plot(x,tu,'ro-');
set(gca,'XTick',(2:2:16));
set(gca,'YTick',(0:20:100));
hold on;
plot(x,tu_,'g^-');
legend('PDD+CVX','PDD');
xlabel('the number of users');
ylabel('Spectral Efficiency: bit/ Hz');
grid on;