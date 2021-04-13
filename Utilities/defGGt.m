function [G,Gt] = defGGt
% Define the discrete gradient
G = @(U) ForwardD(U);
Gt = @(X,Y) Dive(X,Y);

function [Gux,Guy] = ForwardD(U)
% [ux,uy] = G u

Gux = [diff(U,1,2), U(:,1) - U(:,end)];
Guy = [diff(U,1,1); U(1,:) - U(end,:)];

function GtXY = Dive(X,Y)
% GtXY = G_1' X + G_2' Y

GtXY = [X(:,end) - X(:, 1), -diff(X,1,2)];
GtXY = GtXY + [Y(end,:) - Y(1, :); -diff(Y,1,1)];
GtXY = GtXY(:);