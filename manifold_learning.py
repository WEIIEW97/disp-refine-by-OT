import matplotlib.pyplot as plt

# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401
from matplotlib import ticker

from sklearn import manifold

import numpy as np
from skimage.transform import resize
from skimage.morphology import area_closing

import ot
import matplotlib.pylab as pl
import matplotlib.animation as animation

import cv2

from disprefine import save_numpy_array_to_matlab, save_numpy_array


def image_resize(img, sz):
    return resize(img, sz)

def op_norm(arr:np.array):
    return (arr - arr.min()) / (arr.max() - arr.min())

# try to use closing technique for agg output to make it continuous first
def op_area_closing(arr, area_threshold, connectivity=1):
    arr_close = area_closing(arr, area_threshold, connectivity)
    return arr_close

def plot_3d(points, title):
    x, y, z = points.T

    fig, ax = plt.subplots(
        figsize=(6, 6),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = ax.scatter(x, y, z, c=z, s=50, alpha=0.8)
    # ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    plt.show()


def plot_2d(points, title):
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points[:,:,2])
    plt.show()


def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

def apply_colormap(z_vals, cmap_name='gray'):
    """
    Apply a colormap to a 2D numpy array and return the corresponding RGBA values.

    Parameters:
    - z_vals: 2D numpy array of scalar values.
    - cmap_name: String name of the colormap to use.

    Returns:
    - color_mapped: 3D numpy array with an RGBA color mapped from each z value.
    """
    # Normalize z_vals to the range [0, 1]
    norm = plt.Normalize(vmin=z_vals.min(), vmax=z_vals.max())
    
    # Get the colormap
    cmap = plt.cm.get_cmap(cmap_name)
    
    # Apply the colormap (including normalization)
    color_mapped = cmap(norm(z_vals))
    
    return color_mapped

def figure_show(fig, idx=None):
    if idx is None:
        plt.figure()
    else:
        plt.figure(idx)
    plt.imshow(fig)
    plt.show()

# normalize the distribution by mean and std
# do we have to exclude the val == 0 pixels?
def distribution_normalize(arr:np.array):
    mu = np.mean(arr)
    sigma = np.std(arr)
    return (arr - mu) / sigma

def distribution_minmax(arr:np.array):
    return (arr - arr.min()) / (arr.max() - arr.min())

def normal_dist_normalizer(arr):
    mu = np.mean(arr)
    sigma = np.std(arr)
    norm = distribution_normalize(arr)
    return norm, mu, sigma

def minmax_normalizer(arr):
    minv = np.min(arr)
    maxv = np.max(arr)
    norm = distribution_minmax(arr)
    return norm, minv, maxv

# divide the whole image into 4 pieces
def divide(arr:np.ndarray, num_of_pieces=4):
    if num_of_pieces % 2 != 0:
        raise ValueError("number of pieces must be even!")

    if len(arr.shape) == 3:
        h, w, c = arr.shape
    else:
        h, w = arr.shape

    pass
    
def ot_transport_laplace(Xs, Xt):
    ot_emd_laplace = ot.da.EMDLaplaceTransport(reg_lap=100, reg_src=1, similarity='gauss')
    ot_emd_laplace.fit(Xs=Xs, Xt=Xt)

    transp_Xs_laplace = ot_emd_laplace.transform(Xs=Xs)
    return transp_Xs_laplace

def ot_transport_emd(Xs, Xt):
    ot_emd = ot.da.EMDTransport()
    ot_emd.fit(Xs=Xs, Xt=Xt)

    transp_Xs_emd = ot_emd.transform(Xs=Xs)
    return transp_Xs_emd

def build_ot_cost_matrix(Xs, Xt, metric='sqeuclidean'):
    return ot.dist(Xs, Xt, metric=metric)

def restore_from_normal(arr, mu, sigma):
    return arr * sigma + mu

def restore_from_minmax(arr, minval, maxval):
    return arr * (maxval - minval) + minval

def build_pipeline(Xagg, Xdl, hollow_thr=10, method="minmax"):
    # now there are many hollow holes in agg result, do the closing operation
    # to remedy first
    # close_agg = op_area_closing(Xagg, 64, 1)
    close_agg = Xagg
    # find the invalid pixels, and apply on dl result
    insufficient_indices = np.where(Xagg <= hollow_thr)
    dl = np.copy(Xdl)
    dl[insufficient_indices] = 0
    close_agg[insufficient_indices] = 0
    trans_agg_restore = None
    # normalize the distribution for both agg and dl result
    # for the scale shift invariant
    if method == "normal":
        norm_agg, mu_agg, sigma_agg = normal_dist_normalizer(close_agg)
        norm_dl, _, _ = normal_dist_normalizer(dl)

        trans_agg = ot_transport_emd(norm_agg, norm_dl)
        trans_agg_restore = restore_from_normal(trans_agg, mu_agg, sigma_agg)
    elif method == "minmax":
        norm_agg, minv_agg, maxv_agg = minmax_normalizer(close_agg)
        norm_dl, _, _ = minmax_normalizer(dl)

        trans_agg = ot_transport_emd(norm_agg, norm_dl)
        trans_agg_restore = restore_from_minmax(trans_agg, minv_agg, maxv_agg)
    return trans_agg_restore

def build_guided_filter(Xs, Xt, radius=15, eps=0.1):
    assert Xs.shape == Xt.shape,"Images must have the same dimensions and channels"
    Xs = np.float32(Xs)
    Xt = np.float32(Xt)
    return cv2.ximgproc.guidedFilter(Xs, Xt, radius=radius, eps=eps)

def recheck(Xs:np.ndarray, Xt:np.ndarray, kernel_size=4, alpha=0.1, Ws=0.5, Wt=0.5):
    # make sure the width and height is divisible by kernel_size
    assert Xs.shape == Xt.shape,"Images must have the same dimensions and channels"
    assert Xs.shape[0] % kernel_size == 0, f"Image height {Xs.shape[0]} must be divisible by {kernel_size}."
    assert Xs.shape[1] % kernel_size == 0, f"Image width {Xs.shape[0]} must be divisible by {kernel_size}."
    
    Xsc = np.copy(Xs)

    for i in range(0, Xs.shape[0]-kernel_size, kernel_size):
        for j in range(0, Xs.shape[1]-kernel_size, kernel_size):
            # calculate the block mean and min max
            block_s = Xsc[i:i+kernel_size, j:j+kernel_size]
            block_t = Xt[i:i+kernel_size, j:j+kernel_size]
            mu_s = np.mean(block_s)
            mu_t = np.mean(block_t)
            minv_s = np.min(block_s)
            minv_t = np.min(block_t)
            maxv_s = np.max(block_s)
            maxv_t = np.max(block_t)

            cond = np.abs(mu_s-mu_t) <= alpha*mu_t and np.abs(minv_s-minv_t) <= alpha*minv_t and np.abs(maxv_s-maxv_t) <= alpha*maxv_t

            if not cond:
                Xsc[i:i+kernel_size, j:j+kernel_size] = block_t * Wt + block_s * Ws

    print("===> rechecking is completed!")
    return Xsc

def recheck_vectorized(Xs: np.ndarray, Xt: np.ndarray, kernel_size=4, alpha=0.1, Ws=0.5, Wt=0.5):
    # Ensure inputs meet assumptions
    assert Xs.shape == Xt.shape, "Images must have the same dimensions and channels"
    assert Xs.shape[0] % kernel_size == 0, f"Image height {Xs.shape[0]} must be divisible by {kernel_size}."
    assert Xs.shape[1] % kernel_size == 0, f"Image width {Xs.shape[1]} must be divisible by {kernel_size}."

    # Reshape images into (n_blocks_x, n_blocks_y, kernel_size, kernel_size, n_channels)
    # This groups each block's pixels together for block-wise operations
    K = kernel_size
    n_blocks_x, n_blocks_y = Xs.shape[0] // K, Xs.shape[1] // K
    Xs_blocks = Xs.reshape(n_blocks_x, K, n_blocks_y, K, -1).transpose(0, 2, 1, 3, 4)
    Xt_blocks = Xt.reshape(n_blocks_x, K, n_blocks_y, K, -1).transpose(0, 2, 1, 3, 4)

    # Compute mean, min, max for each block
    mu_s = Xs_blocks.mean(axis=(2, 3), keepdims=True)
    mu_t = Xt_blocks.mean(axis=(2, 3), keepdims=True)
    minv_s = Xs_blocks.min(axis=(2, 3), keepdims=True)
    minv_t = Xt_blocks.min(axis=(2, 3), keepdims=True)
    maxv_s = Xs_blocks.max(axis=(2, 3), keepdims=True)
    maxv_t = Xt_blocks.max(axis=(2, 3), keepdims=True)

    # Check conditions
    cond = (np.abs(mu_s - mu_t) <= alpha * mu_t) & \
           (np.abs(minv_s - minv_t) <= alpha * minv_t) & \
           (np.abs(maxv_s - maxv_t) <= alpha * maxv_t)

    # Apply updates
    updates = np.where(cond, (Xt_blocks * Wt + Xs_blocks * Ws), Xs_blocks)

    # Reshape updates back to original image shape
    Xsc = updates.transpose(0, 2, 1, 3, 4).reshape(Xs.shape)
    
    print("===> rechecking is completed!")
    return Xsc


if __name__ == "__main__":
    # for the full resolution, original data
    INVALID_THR = 10
    subfolders = [11, 12, 19]
    for sub in subfolders:
        agg_path = f'data/{sub}/output_0222_agg.npy'
        dl_path = f'data/{sub}/output_0222_DL.npy'
        ot_agg_normal_save_path = f'data/{sub}/OT_normal_0222_agg.mat'
        ot_agg_minmax_save_path = f'data/{sub}/OT_minmax_0222_agg.mat'
        scaled_agg_save_path = f'data/{sub}/scaled_0222_agg.mat'
        full_agg = np.load(agg_path)
        full_dl = np.load(dl_path)

        # resize the source and target image for computation efficiency
        scaled_agg = image_resize(full_agg, (full_agg.shape[0]//2, full_agg.shape[1]//2))
        scaled_dl = image_resize(full_dl, (full_dl.shape[0]//2, full_dl.shape[1]//2))

        print("Switching to normal method.")
        ret_norm = build_pipeline(Xagg=scaled_agg, Xdl=scaled_dl, method="normal")
        diff_ret_norm = ret_norm - scaled_agg
        ret_recheck_norm = recheck(ret_norm, scaled_agg, alpha=0.15)
        print(f"===> Tackling {agg_path} done.")
        print(f"===> The mean difference offset is {np.mean(diff_ret_norm)}")

        print("Switching to minmax method.")
        ret_minmax = build_pipeline(Xagg=scaled_agg, Xdl=scaled_dl, method="minmax")
        diff_ret_minmax = ret_minmax - scaled_agg
        ret_recheck_minmax = recheck(ret_minmax, scaled_agg, alpha=0.15)
        print(f"===> Tackling {agg_path} done.")
        print(f"===> The mean difference offset is {np.mean(diff_ret_minmax)}")

        save_numpy_array_to_matlab(scaled_agg, scaled_agg_save_path)
        save_numpy_array_to_matlab(ret_recheck_norm, ot_agg_normal_save_path)
        save_numpy_array_to_matlab(ret_recheck_minmax, ot_agg_minmax_save_path)

    print("===> mission completed!")

    
    
