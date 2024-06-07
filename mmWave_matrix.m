function [ H ] = mmWave_matrix(Rx,Tx)
    %Rx-number of receive antennas
    %Tx-number of transmit antennas
    %channel parameter
    cluster=randi([1,4]);
%     ray=randi([5,10]);
     AoA=unifrnd(0,pi,1,cluster);
     AoD=unifrnd(0,pi,1,cluster);
%     A=zeros(cluster,ray);
%     B=zeros(cluster,ray);
%     Std=0.2;
%     for k=1:cluster
%          A(k,:)=Laplace_distribution(AoA(k),Std,ray);
%          B(k,:)=Laplace_distribution(AoD(k),Std,ray);
%     end
    
    %chaneel matrix generation
    H=zeros(Rx,Tx);
     %G=rand(cluster,ray);
    for k=1:cluster
%        for s=1:ray
            Rx_vector=response_vector(Rx,AoA(k));
            Tx_vector=response_vector(Tx,AoD(k));
            gain=sqrt(1/2)*(randn(1)+rand(1)*1i);        %complex gain
            H=H+gain*Rx_vector*Tx_vector';
%         end
    end
    H=sqrt(Tx*Rx/(cluster)).*H;
end  

function [v]=response_vector(Len,Ang)
    v=zeros(Len,1);
    for i=1:Len
        v(i)=exp(1i*pi*(i-1)*sin(Ang));
    end
    v=1/sqrt(Len).*v;
end

% function [ random_number ] = Laplace_distribution(Avg,Std,Num)
%     lambda=Std/sqrt(2);          %根据标准差求相应的参数lambda
%     a=rand(1,Num)-0.5;    %生成(-0.5,0.5)区间内均匀分布的随机数列 (一万个数的行向量);
%     random_number=Avg-lambda*sign(a).*log(1-2*abs(a)); %生成符合拉普拉斯分布的随机数列
% end

