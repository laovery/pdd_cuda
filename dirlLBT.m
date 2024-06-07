function [Group1,Group2] = dirlLBT(User,Wifi,beamwidth)
    
    A=angle_calculate(Wifi);B=angle_calculate(User);
    Group1=[];Group2=[];index=[];

    for k=1:length(B)
        for s=1:length(A)
            if abs(A(s)-B(k))<= beamwidth || abs(A(s)-B(k))>= 360-beamwidth
                index=[index,k];
                break
            end
        end
    end

    index1=setdiff(1:size(User,2),index);

    for k=1:length(index)
        p=index(k);
        Group1=[Group1,User(:,p)];
    end


    for k=1:length(index1)
        p=index1(k);
        Group2=[Group2,User(:,p)];
    end
end





      
            