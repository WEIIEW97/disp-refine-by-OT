clc, clear;

test_dir = "D:\william\data\disp-refine";

% sgm_params = struct('blk_size', 15, 'max_d', 64, 'uniq_thr', 0);
blk_size = 15;
max_d = 64;
uniq_thr = 0;
left_pat = "left.*";
right_pat = "right.*";
save_pat = "matlab_out_ksz_15_max_d_64.png";

test_folders = retrive_folders(test_dir);
for i = 1:length(test_folders)
    test_folder = test_folders{i};
    full_path = fullfile(test_dir, test_folder);
    test_sub_folders = retrive_folders(full_path);
    if isempty(test_sub_folders)
        all_files = retrive_files(full_path);
        left_matches = cellfun(@(x) ~isempty(regexp(x, left_pat, 'once')), all_files);
        left_paths = all_files(left_matches); 
        left_path = left_paths{1}; % assume there is only one

        right_matches = cellfun(@(x) ~isempty(regexp(x, right_pat, 'once')), all_files);
        right_paths = all_files(right_matches);
        right_path = right_paths{1};
        disp = sgm(fullfile(full_path, left_path), fullfile(full_path, right_path), blk_size, max_d, uniq_thr);
        disp_u16 = uint16(disp * 64);
        imwrite(disp_u16, fullfile(full_path, save_pat), 'BitDepth', 16);
    end
    for j = 1:length(test_sub_folders)
        full_sub_path = fullfile(test_dir, test_folder, test_sub_folders{j});
        all_files = retrive_files(full_sub_path);
        left_matches = cellfun(@(x) ~isempty(regexp(x, left_pat, 'once')), all_files);
        left_paths = all_files(left_matches); 
        left_path = left_paths{1}; % assume there is only one

        right_matches = cellfun(@(x) ~isempty(regexp(x, right_pat, 'once')), all_files);
        right_paths = all_files(right_matches);
        right_path = right_paths{1};
        disp = sgm(fullfile(full_sub_path, left_path), fullfile(full_sub_path, right_path), blk_size, max_d, uniq_thr);
        disp_u16 = uint16(disp * 64);
        imwrite(disp_u16, fullfile(full_sub_path, save_pat), 'BitDepth', 16);
    end
end