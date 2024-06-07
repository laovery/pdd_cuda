function [ H ] = rice_matrix(Kdb,Rx,Tx)
    % H = a*H_los + b*H_nlos
    % a^2 + b^2 = 1
    % K is the rician factor, denoting as the ratio of the LOS amplitude to the
    % rayleigh component
    % When K = 0, then rice matrix is reduced to rayleigh matrix.
    K=10^(Kdb/10);
    H_los=ones(Rx,Tx);%LOS
    H_nlos=(randn(Rx,Tx)+1i*randn(Rx,Tx))/sqrt(2);%rayleigh
    H=sqrt(K/(K+1))*H_los+sqrt(1/(1+K))*H_nlos;
end  