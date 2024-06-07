function [angle] = angle_calculate(A)
    Xcoor=A(1,:);Ycoor=A(2,:);
    angle=zeros(1,length(Xcoor));
    for k=1:length(Xcoor)
        if Xcoor(k)<0 
          angle(k)=atan(Ycoor(k)/Xcoor(k))*180/pi+180;
        elseif Xcoor(k)>0 && Ycoor(k)>0
          angle(k)=atan(Ycoor(k)/Xcoor(k))*180/pi;
        elseif Xcoor(k)>0 && Ycoor(k)<0
          angle(k)=atan(Ycoor(k)/Xcoor(k))*180/pi+360;
        elseif Xcoor(k)==0 && Ycoor(k)>0
          angle(k)=90;
        elseif Xcoor(k)==0 && Ycoor(k)<0
          angle(k)=270; 
        end
    end
end