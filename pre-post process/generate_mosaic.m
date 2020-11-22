function [mosaic_r, mosaic_g, mosaic_b] = generate_mosaic(f)

shape = size(f);
if(length(shape)~=3)
    disp('Error! The input figure must have 3 channles');
    return;
end

new = uint8(zeros(shape));

% 获取填零的mosaic_rgb
% for j = 1:shape(1)
%     for i = 1:shape(2)
%         if(mod(i,2)==0 && mod(j,2)==0)
%             new(j,i,1) = f(j,i,1);
%         elseif(mod(i,2)==1 && mod(j,2)==1)
%             new(j,i,3) = f(j,i,3);
%         else
%             new(j,i,2) = f(j,i,2);
%         end
%     end
% end
% 
% mosaic_r = new(:,:,1);
% mosaic_g = new(:,:,2);
% mosaic_b = new(:,:,3);

% 获取不填零的mosaic_rgb
r = f(:,:,1);
g = f(:,:,2);
b = f(:,:,3);
% g
mosaic_g = uint8(zeros([shape(1), shape(2) / 2]));
for i = 1:shape(1)
    if (mod(i,2) == 0)
        mosaic_g(i,:) = g(i,1:2:shape(2));
    else
        mosaic_g(i,:) = g(i,2:2:shape(2));
    end
end

% b
mosaic_b = b(1:2:shape(1), 1:2:shape(2));

% r
mosaic_r = r(1:2:shape(1), 1:2:shape(2));


