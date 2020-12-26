clear all;

ori_dir = '../test_mcm/mcm/';
res_dir = '../res/';

filepaths_res = dir(fullfile(res_dir,'*-*'));
psnr_res = [];
for i = 1 : length(filepaths_res)
    psnr_res = [psnr_res; super_test(ori_dir, fullfile(res_dir ,filepaths_res(i).name))];
end

mean_psnr = psnr_res(:, 19);
index = find(mean_psnr == max(mean_psnr));
fprintf('index: %f \n', index);
fprintf('psnr: %f dB\n', max(mean_psnr));