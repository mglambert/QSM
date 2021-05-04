function out = nDIL1(params)
% Non-regularized Dipole Inversion with streaking suppression via L1-norm optimization
%
% Input:
%   params - Structure with the following required fields:
%       params.input  -  Local Field Map
%       params.kernel  -  Dipole Kernel in the Frequency Space
%       params.maxOuterIter - Iterations  (Default = 50)
%       params.tol_update  -  Convergence Limit (Default = 0.1)
%       params.weight  -  Data Fidelity Spatially Variable Weight(recommended = magnitude_data). Not used if not specified
%       params.tau - Gradient Step Size
%       params.alpha  - Percentage of rejection (Default = 0.05) 
%
% Output:
%   out - Structure with the following fields:
%       out.x  -  Susceptibily Map
%       out.iter -  Number of Iterations 
%       out.time  -  Total Elapsed Time
% 
% Example:
%   params = [];
%   params.kernel = kernel;
%   params.weight = mask.*(mag_use/max(mag_use(:)));
%   params.input = mask.*phase_use/phase_scale;
%   out = nDIL1(params);
%

tic
% Required parameters
kernel = params.kernel;
phase = params.input;

N = size(params.input);

if isfield(params,'maxOuterIter')
    num_iter = params.maxOuterIter;
else
    num_iter = 50;
end

if isfield(params,'weight')
   W  = params.weight;
else
   W = ones(N);
end

if isfield(params,'tau')
   tau  = params.tau;
else
   tau = 1;
end
    
if isfield(params, 'alpha')
   alpha  = 1 - params.lambda;
else
   alpha = 0.95;
end

if isfield(params, 'tol_update')
   tol_update  = params.tol_update;
else
   tol_update = 0.1;
end


% Variable initialization
z = zeros(N, 'single') ;
s = zeros(N, 'single');

x = W .* phase;


for t = 1:num_iter
    % update x : susceptibility estimate
    aux =  (susc2field(kernel, x) - phase); 
    x_prev = x;
    x = x_prev - tau * susc2field(conj(kernel), (aux - z + s));
    
    x_update = 100 * norm(x(:)-x_prev(:)) / norm(x(:));
    
    fprintf('Iter: %4d  -  Solution update: %12f\n', t, x_update);

    % Proximal
    aux = (susc2field(kernel, x) - phase) + s; 
    temp = aux(W>0.0);
    temp = sort(unique(round(temp, 4)));
    lambda = temp(max(round(length(temp)*alpha), 1));
    z = max(abs(aux) - W*lambda, 0) .* sign(aux); 
    
    if x_update < tol_update
        break
    end

end

% Extract output values
out.time = toc;toc
out.x = x;
out.iter = t;

end

function [phi] = susc2field(D,X)
%Susceptibility to Field calculation

phi = real(ifftn( D.* fftn( X)));
end


