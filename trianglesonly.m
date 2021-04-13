clear; close all
path(path,genpath(pwd));

% global G Gt
% [G,Gt] = defGGt; % discrete gradient


% ==== Generate ground truth image ==== %
ticks = (-1:0.01:1);   % Distance tags of each line measurment (mm)
ndiscs = 8;            % Number of discs
disc_radius = 0.1;    % Disc radius (mm)
length = 0.15;         % triangle side length

trianglefunc = @(x,y) (y > -length/(2*sqrt(3))) & (sqrt(3)*x/2 + y/2 < length/(2*sqrt(3))) & (-sqrt(3)*x/2 + y/2 < length/(2*sqrt(3))); 
trianglefunc = @(x,y) trianglefunc( x * cos(pi/3) + y * sin(pi/3), y * cos(pi/3) - x * sin(pi/3) );

D = DictProfile(ticks, trianglefunc);

% Generate random map X_true
X_SparseMap = SparseMap(ticks, 'random-location', disc_radius, ndiscs);

% Generate simulated triangles image Y
Y = D * X_SparseMap;

Y = Y.image;           % True triangles image
X_true = X_SparseMap.image; % True sparse map

% blob ground truth
load e_2d;

figure;
subplot(221); imagesc(Y);          title('True triangles');
subplot(222); imagesc(X_true);     title('sparse map');
subplot(223); imagesc(e_2d);       title('blob');
subplot(224); imagesc(Y+e_2d);     title('True Image');




% ==== Given information ==== %
% probelm settings
[p, q] = size(Y); % image size (201*201)
ratio = 0.05; % sampling of measurements
n = p;
m = round(ratio*n^2);

% measurement matrix
A = rand(m,p*q)- 0.5;
%L = rand(m,p*q)- 0.4;

% generate simulated line scans R
x_true = X_true(:);                % true vectorized sparse map x_true
e_true = e_2d(:);                  % true vectorized blob image e_true 
[G, Gx, Gy] = create_GradientMatrix(e_2d);
v_true = G*e_true;                 % true auxiliary varible v_true (discrecte gradient)
R = A*x_true;           % Given observation (simulated line scans)

% Now observation R, measurement matrix A and L are given, we want to
% reconstruct sparse map x and blob (garbage bin) e.




% ==== Reconstruction ==== %
% variable initialization
x = rand(size(x_true));
e = rand(size(e_true));
v = rand(size(v_true));   

% penalty parameters
alpha = 100;
%beta = 1e-100;
gamma = 0;

% iteration
maxiter = 5000;
objvalue = zeros(1, maxiter);
for i = 1:maxiter
    % simplify 
    Ge = G * e;
    Ax = A * x;
%    Le = L * e;

    % compute gradient
    dfx = A'* (Ax - (R));
    %dfv = beta * (v - Ge);
    %dfe = L' * (L*e - (R - A*x)) - beta * G' * (v-Ge);

    % Lipschitz constants
    % Lip_x = norm(A'*A);             %%% it doesn't work, too slow!!!
    % Lip_v = beta;
    % Lip_e = norm(L'*L + beta*G'*G); %%% it doesn't work, too slow!!!

    % step size
    t_x = 1e-5;
    %t_v = 1/beta;
    t_e = 1e-6;
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     % compute tau first
%     if ~isempty(gp)
%         dg = g - gp;                        % dg: pq
%         dg2 = g2 - g2p;                     % dg2: pq
%         ss = uup'*uup;                      % ss: constant
%         sy = uup'*(dg2 + muDbeta*dg);       % sy: constant
%         % sy = uup'*((dg2 + g2) + muDbeta*(dg + g));
%         % compute BB step length
%         tau = abs(ss/max(sy,eps));               % tau: constant
%         
%         fst_itr = false;
%     else
%         % do Steepest Descent at the 1st ieration
%         %d = g2 + muDbeta*g - DtsAtd;         % d: pq
%         [dx,dy] = D(reshape(d,p,q));                    %dx, dy: p*q
%         dDd = norm(dx,'fro')^2 + norm(dy,'fro')^2;      % dDd: cosntant
%         Ad = A(d,1);                        %Ad: m
%         % compute Steepest Descent step length
%         tau = abs((d'*d)/(dDd + muDbeta*Ad'*Ad));
%         
%         % mark the first iteration 
%         fst_itr = true;
%     end 
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % step 1
    x_tmp = x - t_x * dfx;
    %v_tmp = Ge;
    %e_tmp = e - t_e * dfe;

    % step 2
    x = soft(x_tmp, alpha * t_x);     %t_x should be 1/Lip_x
    %v = shrink(v_tmp, gamma/beta);
    %e = e_tmp;
    
    % objective function
    objf = 1/2 * norm(Ax - R)^2; % convex and smooth
    objg = alpha * norm(x,1) + gamma * norm(v); % convex and non-smooth
    objfunc = objf + objg;
    
    fprintf('iteration: %i\n',i);
    fprintf('objective value: %e\n', objfunc);
    objvalue(i) = objfunc;
end

e = e_2d; %%% useless, just for visualization
% ==== Results ==== % 
X_rec = reshape(x, size(X_true));
X_RecSparseMap = X_SparseMap;
X_RecSparseMap.image = X_rec;

Y_rec = D * X_RecSparseMap;
Y_rec = Y_rec.image; 

e_2d_rec = reshape(e, size(e_2d));

figure;
plot(objvalue(10:end), 'LineWidth', 2); xlabel('Iteration Number'); ylabel('Function value');


figure;
subplot(221); imagesc(Y_rec);          title('Reconstruted triangles');
subplot(222); imagesc(X_rec);          title('Reconstructed sparse map');
subplot(223); imagesc(e_2d_rec);       title('Reconstructed blob');
subplot(224); imagesc(Y_rec+e_2d_rec); title('Reconstructed Image');