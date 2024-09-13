function [sub_files_names] = retrive_files(path)
%RETRIVE_FILES Summary of this function goes here
%   Detailed explanation goes here
files = dir(path);
file_flags = [files.isdir];
sub_files = files(~file_flags);
sub_files_names = {sub_files.name}; 
end

