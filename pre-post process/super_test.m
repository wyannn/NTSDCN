function [psnr_res]=super_test(ori_dir, res_dir)

filepaths_gnd = dir(fullfile(ori_dir,'*.tif'));
filepaths_res = dir(fullfile(res_dir,'*.bmp'));

for i = 1 : length(filepaths_res)
    im_gnd = imread(fullfile(ori_dir,filepaths_gnd(i).name));
    im_res = imread(fullfile(res_dir,filepaths_res(i).name));
    im_gnd = modcrop(im_gnd, 2);
    
    [h, w, ~] = size(im_gnd);
    %% g
%     im_res(1:2:h, 1:2:w) = im_gnd(1:2:h, 1:2:w);
%     im_res(2:2:h, 2:2:w) = im_gnd(2:2:h, 2:2:w);
    
    %%  b
%     im_res(2:2:h, 1:2:w) = im_gnd(2:2:h, 1:2:w);
    
    %%  r
%     im_res(1:2:h, 2:2:w) = im_gnd(1:2:h, 2:2:w);

    %% rgb
    im_res(1:2:h, 2:2:w, 1) = im_gnd(1:2:h, 2:2:w, 1);
    im_res(1:2:h, 1:2:w, 2) = im_gnd(1:2:h, 1:2:w, 2);
    im_res(2:2:h, 2:2:w, 2) = im_gnd(2:2:h, 2:2:w, 2);
    im_res(2:2:h, 1:2:w, 3) = im_gnd(2:2:h, 1:2:w, 3);
   
    psnr_res(i) = impsnr(im_gnd, im_res, 4);
end

psnr_res = [psnr_res mean(psnr_res)];
end
