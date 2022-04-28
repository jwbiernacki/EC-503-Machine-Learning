function [W] = train_svm_proj(X,y,Delta,T,lambda)

k=size(Delta,1);
[d,m]=size(X);
W=zeros(d,k);

for i=1:T
    % sample a point at random.
    %
    % In alternative, one could shuffle and go over all the samples, then shuffle again. This works anyway, even if for different reasons than the approach above.
    idx=randi(m);
    Xi=X(:,idx);
    Yi=y(idx);
    pred=Xi'*W;
    aug_pred=pred+Delta(Yi,:);
    [~,mx_idx]=max(aug_pred);
    % shrinking due to the regularizer. Note that eta_i*lambda=1/i
    W=W*(1-1/i);
    % update two columns of W
    W(:,Yi)=W(:,Yi)+1/(lambda*i)*Xi;
    W(:,mx_idx)=W(:,mx_idx)-1/(lambda*i)*Xi;
end

end
