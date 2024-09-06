import matplotlib.pyplot as plt
from matplotlib import ticker

import numpy as np


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

def figure_show(fig, idx=None, vmin=None, vmax=None):
    if idx is None:
        plt.figure()
    else:
        plt.figure(idx)
    
    # If vmin and vmax are not provided, use the min and max values of the image.
    if vmin is None:
        vmin = fig.min()
    if vmax is None:
        vmax = fig.max()
    # Display the image with the specified vmin and vmax
    img = plt.imshow(fig, vmin=vmin, vmax=vmax)
    # Add a colorbar with the correct scaling.
    plt.colorbar(img)
    plt.show()