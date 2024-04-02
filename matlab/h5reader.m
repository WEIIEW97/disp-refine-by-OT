function m = h5reader(data_path, dataset_name)
%H5READER Summary of this function goes here
%   Detailed explanation goes here
m = h5read(data_path, dataset_name);
end

