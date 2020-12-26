%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Bilinear Interpolation
%    Input
%     - X : X image
%     - Y : Y image
%     - peak : signal peak value
%     - b : image border cut
%    Output
%     - psnr : peak signal noise ratio 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function res_psnr = impsnr(X, Y, b)

if( nargin < 3 )
 b = 0;
end

if( b > 0 )
 X = X(b:size(X,1)-b, b:size(X,2)-b,:);
 Y = Y(b:size(Y,1)-b, b:size(Y,2)-b,:);
end

%for i=1:size(X, 3)
res_psnr = psnr(X, Y);
%end

end