import numpy as np
from skimage.transform import resize
from skimage.morphology import area_closing

def image_resize(img, sz):
    return resize(img, sz)

def op_norm(arr:np.array):
    return (arr - arr.min()) / (arr.max() - arr.min())

# try to use closing technique for agg output to make it continuous first
def op_area_closing(arr, area_threshold, connectivity=1):
    return area_closing(arr, area_threshold, connectivity)


def distribution_normalize(arr:np.array):
    mu = np.mean(arr)
    sigma = np.std(arr)
    return (arr - mu) / sigma

def distribution_minmax(arr:np.array):
    return (arr - arr.min()) / (arr.max() - arr.min())  