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
[rows, cols] = size(Y); % image size (201*201)
ratio = 0.05; % sampling of measurements
n = rows;
m = round(ratio*n^2);

% measurement matrix
A = rand(m,rows*cols)- 0.5;
L = rand(m,rows*cols)- 0.4;

% generate simulated line scans R
x_true = X_true(:);                % true vectorized sparse map x_true
e_true = e_2d(:);                  % true vectorized blob image e_true 
[G, Gx, Gy] = create_GradientMatrix(e_2d);
v_true = G*e_true;                 % true auxiliary varible v_true (discrecte gradient)
R = A*x_true + L*e_true;           % Given observation (simulated line scans)

% Now observation R, measurement matrix A and L are given, we want to
% reconstruct sparse map x and blob (garbage bin) e.


% ==== Reconstruction ==== %
% variable initialization
x = rand(size(x_true));
e = rand(size(e_true));
v = rand(size(v_true));   

p = x;
q = v;
r = e;
tk = 1;

% penalty parameters
alpha = 100;  % verified!!!
beta = 5000;
gamma = 2e+10;

%% iteration
maxiter = 1500;
objvalue = zeros(1, maxiter);
for i = 1:maxiter
    x_old = x;
    v_old = v;
    e_old = e;
    
    % compute gradient
    dfp = A'* (A*p - (R - L*e));
    dfq = beta * (q - G*e);
    dfr = L' * (L*r - (R - A*x)) - beta * G' * (v-G*r);

    % Lipschitz constants
    % Lip_x = norm(A'*A);             %%% it doesn't work, too slow!!!
    % Lip_v = beta;
    % Lip_e = norm(L'*L + beta*G'*G); %%% it doesn't work, too slow!!!

    % step size
    t_x = 1e-5;
    t_v = 1/beta;
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
    x_tmp = p - t_x * dfp;
    v_tmp = q - t_v * dfq;
    e_tmp = r - t_e * dfr;

    % step 2
    x = soft(x_tmp, alpha * t_x);     %t_x should be 1/Lip_x
    v = shrink(v_tmp, gamma * t_v);
    e = e_tmp;
    
    % step 3
    t_old = tk;
    tk = (1+sqrt(1+4*tk*tk))/2;
    bk = (t_old - 1)/tk;
    
    % step 4
    p = x + bk * (x - x_old);
    q = v + bk * (v - v_old);
    r = e + bk * (e - e_old);
    
    % objective function
    objf = 1/2 * norm(A*x + L*e - R)^2 + beta/2 * norm(v - G*e)^2; % convex and smooth
    objg = alpha * norm(x,1) + gamma * norm(v); % convex and non-smooth
    objfunc = objf + objg;
    
    fprintf('iteration: %i\n',i);
    fprintf('objective value: %e\n', objfunc);
    objvalue(i) = objfunc;
end

%% ==== Results ==== % 
X_rec = reshape(x, size(X_true));
X_RecSparseMap = X_SparseMap;
X_RecSparseMap.image = X_rec;

Y_rec = D * X_RecSparseMap;
Y_rec = Y_rec.image; 

e_2d_rec = reshape(e, size(e_2d));

figure;
%plot(objvalue(10:end), 'LineWidth', 2); xlabel('Iteration Number'); ylabel('Function value');
plot(objvalue, 'LineWidth', 2); xlabel('Iteration Number'); ylabel('Function value');
title('Convergence of APG');

figure;
subplot(221); imagesc(Y_rec);          title('Reconstruted triangles');
subplot(222); imagesc(X_rec);          title('Reconstructed sparse map');
subplot(223); imagesc(e_2d_rec);       title('Reconstructed blob');
subplot(224); imagesc(Y_rec+e_2d_rec); title('Reconstructed Image');