clc;
a = 0;
tu = zeros(1,8);
tu_ = zeros(1,8);
c = 1;
for use = 1:8 
    a=0;
    b=0;
    for k = 1:10
        a = a + FD_NRU(use*2,32,8,60,0.08,0);
        b = b + FD_NRU(use*2,32,8,60,0.08,1);
    end
    
    tu(c) = a/10;
    tu_(c) = b/10;
    c = c+1;
end 

