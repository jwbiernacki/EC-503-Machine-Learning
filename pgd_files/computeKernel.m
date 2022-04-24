%% Header
% Parker Dunn
% EC 503 Project
% Created 23 April 2022

function [K] = computeKernel(X1, X2, kernel, param)

%% Setup
[m1, d] = size(X1);
[m2, ~] = size(X2);

%% Kernel Calculation

switch kernel
    case 1 % linear
        K = X1 * X2';
        
    case 2 % polynomial
        K = (1 + (X1*X2')).^param;

    case 3 % gaussian
        squared_norm_vec = norm(repelem(X1, m2, 1)  -  repmat(X2, m1)).^2;            % <-- this is ||x - z|| for all sample combos
                % # of rows = m1 * m2     % # of rows = m2 * m1
                
        % reshaping len vector to a len matrix --> e.g. len_vec(3) = X1(1,:) - X2(3,:)
        % need to fill matrix horizontally
        squared_norm_mat = reshape(squared_norm_vec, m1, m2)';
        
        K = exp(-param .* squared_norm_mat);
        
        % Alternative code to this
%         K = zeros(m1, m2);
%         for i = 1:m1
%             for j = 1:m2
%                 K(i,j) = exp(-param*norm(X1(i,:)-X2(j,:))^2);
%             end
%         end

    otherwise
        error("No appropriate kernel selection was made. Please try again.");

end % End of switch statement


end % End of Function




%             text = "The kernel number you selected was not availble.\nPlease select one of the kernels below.";
%             list = ["Linear", "Polynomial", "Gaussian"];
%             [index, tf] = listdlg('ListString', list, 'PromptString', text, 'SelectionMode', 'single');
% 
%             if ~tf
%                 error("No appropriate kernel selection was made. Please try again.");
%             end
    