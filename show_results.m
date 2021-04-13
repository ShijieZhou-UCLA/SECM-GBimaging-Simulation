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