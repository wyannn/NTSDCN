clear all;

ori_dir = 'test_kodak/kodak';
res_dir = 'res';

filepaths_res = dir(fullfile(res_dir,'*-*'));
psnr_res = [];
for i = 1 : length(filepaths_res)
    psnr_res = [psnr_res; super_test(ori_dir, fullfile('res/' ,filepaths_res(i).name))];
end

mean_psnr = psnr_res(:, 25);
index = find(mean_psnr == max(mean_psnr));
fprintf('index: %f \n', index);
fprintf('psnr: %f dB\n', max(mean_psnr));