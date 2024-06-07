function [value]=bisection(A,B,c,lower,upper,threshold)
    % function: find the 'mid' value that satisfies
    % Tr{B'*(A+ans*I)^(-2)*B}=c
    % matrix A is Hermitian
    
    fmax=trace(B'*pinv(A+lower*eye(size(A,1)))'*pinv(A+lower*eye(size(A,1)))*B); 
    if fmax<=c
        value=lower;
        return;
    end
    
    while (upper-lower>=threshold)
            mid=(upper+lower)/2;
            fmid=trace(B'*pinv(A+mid*eye(size(A,1)))'*pinv(A+mid*eye(size(A,1)))*B);
            if (fmid<c)
                upper=mid;
            elseif (fmid>c)
                lower=mid;
            else 
                value=mid;
                return;
            end
    end
        
    value=(upper+lower)/2;  
end