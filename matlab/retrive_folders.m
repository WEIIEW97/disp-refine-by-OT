function [sub_folders_name] = retrive_folders(path)
%RETRIVE_FOLDERS Summary of this function goes here
%   Detailed explanation goes here
files = dir(path);
dir_flags = [files.isdir];
sub_folders = files(dir_flags);
sub_folders_name = {sub_folders(3:end).name}; % start at 3 to skip . and ..
end

