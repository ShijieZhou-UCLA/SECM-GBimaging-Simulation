function [mD, mDx, mDy]= create_GradientMatrix(mI)
% input: 
% mI is an image(matrix), size: n * n
% 
% output: 
% mDx is the first-order finite difference matrix in the x direction,
% size: n^2 * n^2
%
% mDy is the first-order finite difference matrix in the y direction, 
% size: n^2 * n^2
%
% mD is the finite difference (discrete gradient) matrix,
% size: 2n^2 * n^2

numRows     = size(mI, 1);
numCols     = size(mI, 2);
numPixels   = numRows * numCols;


mDx = sparse(numPixels, numPixels);
mDy = sparse(numPixels, numPixels);


% for x
for ii = 1:(numPixels - numCols)
    mDx(ii, ii) = -1;
    mDx(ii, ii + numCols) = 1;
end
for ii = (numPixels - numCols + 1):numPixels
    mDx(ii, ii) = -1;
    mDx(ii, ii + numCols - numPixels) = 1;
end


% for y
for ii = 1:(numPixels)
    mDy(ii, ii) = -1;
    if(mod(ii, numRows) == 0)
        mDy(ii, ii + 1 - numCols) = 1;
    else
        mDy(ii, ii + 1) = 1;
    end
    
end

mD = [mDx; mDy];

% mIx = reshape(mDx * mI(:), numRows, numCols);
% mIy = reshape(mDy * mI(:), numRows, numCols);

