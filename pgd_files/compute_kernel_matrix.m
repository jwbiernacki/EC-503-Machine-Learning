function K = compute_kernel_matrix(X1,X2,kernel,param)

[m1,d]=size(X1);
[m2,d]=size(X2);

if kernel==1 %linear
    K=X1*X2';
elseif kernel==2 % polynomial
    % This code is very slow, but it is easy to understand
    K=zeros(m1,m2);
    for i=1:m1
        for j=1:m2
            K(i,j)=(1+X1(i,:)*X2(j,:)')^param;
        end
    end
else %polynomial
    % This code is very slow, but it is easy to understand
    
    K=zeros(m1,m2);
    for i=1:m1
        for j=1:m2
            K(i,j)=exp(-param*norm(X1(i,:)-X2(j,:))^2);
        end
    end
end

end

