function [user] = random_user(a,b,N,R)
%a wigig location
%b beamwidth
%n number of user
    %global R
    a1=angle_calculate(a);
    n=1;user=zeros(2,N);
    while(n<=N)
        A=unifrnd(0,2*pi,1);
        A1=A*360/(2*pi);
        if abs(A1-a1)<=b/2 || abs(A1-a1)>=360-b/2
          
        else
            d=unifrnd(R/2,R,1);
            x=d*cos(A);  
            y=d*sin(A);
            user(:,n)=[x;y];
            n=n+1;
        end
    end
    
end