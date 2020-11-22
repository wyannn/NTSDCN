clear;close all;
%% settings
folder = '91';
savepath = 'train_40_800_b64_st40_rgb.h5';
size_input = 40;
size_label = 40;
stride = 40;
chunksz = 64;

%% initialization
data_g = zeros(size_input, size_input, 1, 1);
data_r = zeros(size_input, size_input, 1, 1);
data_b = zeros(size_input, size_input, 1, 1);
label_g = zeros(size_label, size_label, 1, 1);
label_r = zeros(size_label, size_label, 1, 1);
label_b = zeros(size_label, size_label, 1, 1);
count = 0;

%% generate data
filepaths1 = dir(fullfile(folder,'*.bmp'));
filepaths2 = dir(fullfile(folder,'*.png'));
filepaths = vertcat(filepaths1,filepaths2);
% filepaths = dir(fullfile(folder,'*.png'));
    
for i = 1 : length(filepaths)
    
    image = imread(fullfile(folder,filepaths(i).name));
    image = modcrop(image, 2);
    image = modcrop(image, 2);
    if size(image,3)>1
        [mosaic_r, mosaic_g, mosaic_b] = generate_mosaic(image);
    end
   
    im_label_r = image(:, :, 1);
    im_label_g = image(:, :, 2);
    im_label_b = image(:, :, 3);
    [hei,wid] = size(im_label_g);

    for x = 1 : stride : hei-size_input+1
        for y = 1 :stride : wid-size_input+1
            
            subim_mosaic_g = mosaic_g(x : x+size_input-1, y : y+size_input-1);
            subim_mosaic_r = mosaic_r(x : x+size_input-1, y : y+size_input-1);
            subim_mosaic_b = mosaic_b(x : x+size_input-1, y : y+size_input-1);
            subim_label_g = im_label_g(x : x+size_label-1, y : y+size_label-1);
            subim_label_r = im_label_r(x : x+size_label-1, y : y+size_label-1);
            subim_label_b = im_label_b(x : x+size_label-1, y : y+size_label-1);

            count=count+1;
            data_g(:, :, 1, count) = subim_mosaic_g;
            data_r(:, :, 1, count) = subim_mosaic_r;
            data_b(:, :, 1, count) = subim_mosaic_b;
            label_g(:, :, 1, count) = subim_label_g;
            label_r(:, :, 1, count) = subim_label_r;
            label_b(:, :, 1, count) = subim_label_b;
        end
    end
end

order = randperm(count);
data_g = data_g(:, :, 1, order);
data_r = data_r(:, :, 1, order);
data_b = data_b(:, :, 1, order);
label_g = label_g(:, :, 1, order); 
label_r = label_r(:, :, 1, order); 
label_b = label_b(:, :, 1, order); 

%% writing to HDF5
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata_g = data_g(:,:,1,last_read+1:last_read+chunksz); 
    batchdata_r = data_r(:,:,1,last_read+1:last_read+chunksz); 
    batchdata_b = data_b(:,:,1,last_read+1:last_read+chunksz); 
    batchlabs_g = label_g(:,:,1,last_read+1:last_read+chunksz);
    batchlabs_r = label_r(:,:,1,last_read+1:last_read+chunksz);
    batchlabs_b = label_b(:,:,1,last_read+1:last_read+chunksz);

    startloc = struct('dat_g',[1,1,1,totalct+1], 'dat_r',[1,1,1,totalct+1], 'dat_b',[1,1,1,totalct+1], 'lab_g', [1,1,1,totalct+1], 'lab_r', [1,1,1,totalct+1], 'lab_b', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata_g, batchdata_r, batchdata_b, batchlabs_g, batchlabs_r, batchlabs_b, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);
