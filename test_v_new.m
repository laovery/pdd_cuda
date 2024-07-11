clc;
a = 0;
tu = zeros(8,2);
tu_ = zeros(8,2);
c = 1;
for use = 1:8 
    a=[0,0];
    %b=[0,0];
    for k = 1:10
        a = a + FD_NRU(use*2,32,8,60,0.08,1);
        %pause(5);

        %b = b + FD_NRU(use*2,32,8,60,0.08,1);
    end
    
    tu(c,:) = a/10;
    %tu_(c) = b/10;
    c = c+1;
end 

disp(c);


