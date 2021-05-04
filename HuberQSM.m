function out = HuberQSM(params)
% Improving Quantitative Susceptibility Mapping reconstructions via non-linear Huber loss data fidelity term (Huber-QSM)
%
% This method solve:  min_{x} huber(w*(exp(i*F^H*D*F *x) - exp(i*phi) + alpha * TV(x)
%
% Input:
%   params - Structure with the following required fields:
%       params.input  -  Local Field Map
%       params.alpha  - Regularization Weight 
%       params.kernel  -  Dipole Kernel in the Frequency Space
%       params.mu1  -  Gradient Consistency Weight (Default = 100 * alpha)
%       params.mu2 - Fidelity Consistency Weight (Default = 1.0)
%       params.mu3 - Fidelity Consistency Weight (Default = 1.0)
%       params.maxOuterIter - Iterations  (Default = 100)
%       params.tol_update  -  Convergence Limit (Default = 0.1)
%       params.regweight  -  Regularization Spatially Variable Weight. Not used if not specified
%       params.weight  -  Data Fidelity Spatially Variable Weight(recommended = magnitude_data). Not used if not specified
%       params.delta_tol  -  Convergence Tolerance, for the Newton-Raphson solver (Default = 1e-6)
%       params.deltahu - Huber loss parameter (Default = 0.001)
%
% Output:
%   out - Structure with the following fields:
%       out.x  -  Susceptibily Map
%       out.iter -  Number of Iterations 
%       out.time  -  Total Elapsed Time
%       out.params  -  Input Params
%       out.updates = Convergence Update Solution
% 
% Example:
%   params = [];
%   params.kernel = kernel;
%   params.weight = mask.*(mag_use/max(mag_use(:)));
%   params.input = mask.*phase_use/phase_scale;
%   params.alpha = 10^-4.785;
%   out = HuberQSM(params);
%
% Based on the code by Carlos Milovic at https://gitlab.com/cmilovic/FANSI-toolbox


tic

% Required parameters
alpha = params.alpha;
kernel = params.kernel;
phase = params.input;



% Optional parameters
if isfield(params,'mu1')
     mu = params.mu1;
else
    mu = 100*alpha;
end

if isfield(params,'mu2')
     mu2 = params.mu2;
else
    mu2 = 1.0;
end

if isfield(params,'mu3')
     mu3 = params.mu3;
else
    mu3 = 1.0;
end

N = size(params.input);

if isfield(params,'maxOuterIter')
    num_iter = params.maxOuterIter;
else
    num_iter = 50;
end

if isfield(params,'tol_update')
   tol_update  = params.tol_update;
else
   tol_update = 0.1;
end

if isfield(params,'regweight')
    regweight = params.regweight;
    if length(size(regweight)) == 3
        regweight = repmat(regweight,[1,1,1,3]);
    end
else
    regweight = ones([N 3]);
end

if ~isfield(params,'delta_tol')
    delta_tol = 1e-6;
else
    delta_tol = params.delta_tol;
end

if ~isfield(params,'deltahu')
    deltahu = 0.001;
else
    deltahu = params.deltahu;
end

if isfield(params,'weight')
   W  = params.weight;
else
   W = ones(N);
end


% Variable initialization
z_dx = zeros(N, 'single');
z_dy = zeros(N, 'single');
z_dz = zeros(N, 'single');

s_dx = zeros(N, 'single');
s_dy = zeros(N, 'single');
s_dz = zeros(N, 'single');

x = zeros(N, 'single');

z2 =  W.*phase;
s2 = zeros(N,'single'); 

z3 = zeros(N,'single');
s3 = zeros(N,'single');


    
% Define the operators
IS = exp(1i*phase);

[k1, k2, k3] = ndgrid(0:N(1)-1,0:N(2)-1,0:N(3)-1);

E1 = 1 - exp(2i .* pi .* k1 / N(1));
E2 = 1 - exp(2i .* pi .* k2 / N(2));
E3 = 1 - exp(2i .* pi .* k3 / N(3));

E1t = conj(E1);
E2t = conj(E2);
E3t = conj(E3);

EE2 = E1t .* E1 + E2t .* E2 + E3t .* E3;
K2 = abs(kernel).^2;

ll = alpha/mu;

updates = [];
    
for t = 1:num_iter
    % update x : susceptibility estimate
    tx = E1t .* fftn(z_dx - s_dx);
    ty = E2t .* fftn(z_dy - s_dy);
    tz = E3t .* fftn(z_dz - s_dz);
    
    x_prev = x;
    Dt_kspace = conj(kernel) .* fftn(z2-s2);
    x = real(ifftn( (mu * (tx + ty + tz) + mu2*Dt_kspace) ./ (eps + mu2*K2 + mu * EE2) ));
   
    x_update = 100 * norm(x(:)-x_prev(:)) / norm(x(:));
    updates(t) = x_update;
    
    fprintf('Iter: %4d  -  Solution update: %12f\n', t, x_update);

    if x_update < tol_update || isnan(x_update)
        break
    end
    
    if t < num_iter
        % update z : gradient varible
        Fx = fftn(x);
        x_dx = real(ifftn(E1 .* Fx));
        x_dy = real(ifftn(E2 .* Fx));
        x_dz = real(ifftn(E3 .* Fx));
        
        z_dx = max(abs(x_dx + s_dx) - regweight(:,:,:,1)*ll, 0) .* sign(x_dx + s_dx);
        z_dy = max(abs(x_dy + s_dy) - regweight(:,:,:,2)*ll, 0) .* sign(x_dy + s_dy);
        z_dz = max(abs(x_dz + s_dz) - regweight(:,:,:,3)*ll, 0) .* sign(x_dz + s_dz);
    
        % update s : Lagrange multiplier
        s_dx = s_dx + x_dx - z_dx;
        s_dy = s_dy + x_dy - z_dy;            
        s_dz = s_dz + x_dz - z_dz;  
        
        
        Y3 = exp(1i*z2)-IS+s3; % aux variable
        z3 = (deltahu*mu3*Y3 + (W.^2) .* max(abs(Y3) - (W.^2 + deltahu*mu3)./(mu3*W + eps), 0) .* sign(Y3)) ./ (W.^2 + deltahu*mu3); 
        jiji = abs(Y3) - (W.^2 + deltahu*mu3)./(mu3*W + eps);
        disp(sum(jiji(:)>0));
        rhs_z2 = mu2*real(ifftn(kernel.*Fx)+s2  );
        z2 =  rhs_z2 ./ mu2 ;

        % Newton-Raphson method
        delta = inf;
        inn = 0;
        yphase = angle( IS+z3-s3 );
        ym = abs(IS+z3-s3);
        while (delta > delta_tol && inn < 10)
            inn = inn + 1;
            norm_old = norm(z2(:));
            
            update = ( mu3 .* sin(z2 - yphase-1i*log(ym)) + mu2*z2 - rhs_z2 )./( mu3 .* cos(z2 - yphase-1i*log(ym)) + mu2 +eps);   
        
            z2 = (z2 - update);     
            delta = norm(update(:)) / norm_old;
        end        
        
        s2 = s2 + real(ifftn(kernel.*Fx)) - z2;
        s3 = exp(1i*z2)-IS+s3 - z3;
    end
    
    
end
% Extract output values
out.time = toc;toc
out.x = x;
out.iter = t;
out.updates = updates;
out.params = params;

end
