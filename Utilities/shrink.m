function y = shrink(x,lda)
%shrink(x,lda) 2D shrinkage operator of vector x with threshold lda
if lda<0 
    error('lda should be non-negative'); 
else
    y = x/norm(x).*(max(norm(x)-lda,0));
end