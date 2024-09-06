clc, clear;
% 
% root_dir = "/home/william/extdisk/data/middlebury/middlebury2014/Bicycle1-perfect";
% left_img = fullfile(root_dir, 'im0.png');
% right_img = fullfile(root_dir, 'im1.png');
left_path =  "/home/william/Codes/disp-refine-by-OT/data/11/left.png";
right_path = "/home/william/Codes/disp-refine-by-OT/data/11/right.png";

limg = imread(left_path);
rimg = imread(right_path);
% limg = imresize(limg, 0.5);
% rimg = imresize(rimg, 0.5);
disp = disparity(im2gray(limg), im2gray(rimg), 'Blocksize', 15, 'DisparityRange', [0, 256], 'UniquenessThreshold', 0 );
disp(disp<0)=0;
figure;
imshow(disp, []);
colormap parula;
colorbar;
