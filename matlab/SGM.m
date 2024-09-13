function [disp] = sgm(left_path, right_path, blk_size, max_d, uniq_thr)
% Summary of this function goes here
%   Detailed explanation goes here
limg = imread(left_path);
rimg = imread(right_path);
disp = disparity(im2gray(limg), im2gray(rimg), 'Blocksize', blk_size, 'DisparityRange', [0, max_d], 'UniquenessThreshold', uniq_thr );
disp(disp<0)=0;
end

