clc; clear;
% global dataset name
DATASET_NAME="/data";

% for data 11
ot_minmax_path_11 = '/Users/williamwei/Codes/disp-refine-by-DL/data/11/OT_minmax_0222_agg.mat';
ot_normal_path_11 = '/Users/williamwei/Codes/disp-refine-by-DL/data/11/OT_normal_0222_agg.mat';
scaled_agg_path_11 = '/Users/williamwei/Codes/disp-refine-by-DL/data/11/scaled_0222_agg.mat';
dl_path_11 = '/Users/williamwei/Codes/disp-refine-by-DL/data/11/output_0222_DL.mat';

minmax_11 = h5reader(ot_minmax_path_11, DATASET_NAME)';
normal_11 = h5reader(ot_normal_path_11, DATASET_NAME)';
scaled_11 = h5reader(scaled_agg_path_11, DATASET_NAME)';

minmax_diff_11 = minmax_11 - scaled_11;
normal_diff_11 = normal_11 - scaled_11;

disp(max(minmax_11, [], 'all'));
disp(min(minmax_11, [], 'all'));

figure;
imshow(minmax_diff_11, [0, 10]);
colormap gray;
colorbar;
