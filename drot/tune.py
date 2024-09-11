import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm


def recheck(Xs: np.ndarray, Xt: np.ndarray, kernel_size=4, alpha=0.1, Ws=0.5, Wt=0.5):
    # make sure the width and height is divisible by kernel_size
    assert Xs.shape == Xt.shape, "Images must have the same dimensions and channels"
    assert (
        Xs.shape[0] % kernel_size == 0
    ), f"Image height {Xs.shape[0]} must be divisible by {kernel_size}."
    assert (
        Xs.shape[1] % kernel_size == 0
    ), f"Image width {Xs.shape[0]} must be divisible by {kernel_size}."

    Xsc = np.copy(Xs)

    for i in range(0, Xs.shape[0] - kernel_size, kernel_size):
        for j in range(0, Xs.shape[1] - kernel_size, kernel_size):
            # calculate the block mean and min max
            block_s = Xsc[i : i + kernel_size, j : j + kernel_size]
            block_t = Xt[i : i + kernel_size, j : j + kernel_size]
            mu_s = np.mean(block_s)
            mu_t = np.mean(block_t)
            minv_s = np.min(block_s)
            minv_t = np.min(block_t)
            maxv_s = np.max(block_s)
            maxv_t = np.max(block_t)

            cond = (
                np.abs(mu_s - mu_t) <= alpha * mu_t
                and np.abs(minv_s - minv_t) <= alpha * minv_t
                and np.abs(maxv_s - maxv_t) <= alpha * maxv_t
            )

            if not cond:
                Xsc[i : i + kernel_size, j : j + kernel_size] = (
                    Wt * block_t + Ws * block_s
                )

    print("===> rechecking is completed!")
    return Xsc


def recheck_vectorized(
    Xs: np.ndarray, Xt: np.ndarray, kernel_size=4, alpha=0.1, Wt=0.5, Ws=0.5
):
    # Ensure inputs meet assumptions
    assert Xs.shape == Xt.shape, "Images must have the same dimensions and channels"
    assert (
        Xs.shape[0] % kernel_size == 0
    ), f"Image height {Xs.shape[0]} must be divisible by {kernel_size}."
    assert (
        Xs.shape[1] % kernel_size == 0
    ), f"Image width {Xs.shape[1]} must be divisible by {kernel_size}."

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
    cond = (
        (np.abs(mu_s - mu_t) <= alpha * mu_t)
        & (np.abs(minv_s - minv_t) <= alpha * minv_t)
        & (np.abs(maxv_s - maxv_t) <= alpha * maxv_t)
    )

    # Apply updates
    updates = np.where(cond, Wt * Xt_blocks + Ws * Xs_blocks, Xs_blocks)

    # Reshape updates back to original image shape
    Xsc = updates.transpose(0, 2, 1, 3, 4).reshape(Xs.shape)

    print("===> rechecking is completed!")
    return Xsc


class KLTripletOptimizer:

    def __init__(self, xs, xt, xd, device='cuda', reg_weight=1e-8):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Running on: {self.device}")
        self.kl_loss = nn.KLDivLoss(reduction='batchmean').to(self.device)
        self.huber_loss = nn.HuberLoss(reduction='mean').to(self.device)
        self.xs = xs.to(self.device).detach().requires_grad_(True)
        self.xs_init = self.xs.clone().detach()
        self.xt = xt.to(self.device)
        self.xd = xd.to(self.device)

        self.optimizer = optim.Adam([self.xs], lr=0.01)
        self.kl_losses = []
        self.huber_losses = []
        self.total_losses = []

        self.kl_w = 1.0
        self.hu_w = 1.0
        self.reg_weight = reg_weight

    def weights_adjust(self, kl_w, hu_w):
        self.kl_w = kl_w
        self.hu_w = hu_w

    def compute_histogram(self, x:torch.Tensor):
        # b, c, h, w = x.size()
        x = x.detach().squeeze().flatten()
        hist = torch.histc(x, bins=256)
        hist = hist / hist.sum()
        return hist
    
    def train(self, num_epoches=3000):
        for _ in tqdm(range(num_epoches)):
            self.optimizer.zero_grad()
            xs_norm = torch.sigmoid(self.xs)
            xd_norm = torch.sigmoid(self.xd)
            hist_s = self.compute_histogram(xs_norm)
            hist_d = self.compute_histogram(xd_norm)

            loss_kl = self.kl_loss(torch.log(hist_s+1e-10), hist_d)
            loss_huber = self.huber_loss(self.xs, self.xt)
            reg_loss = self.reg_weight * torch.norm(self.xs - self.xs_init)**2  # add a very small l2 to avoid overfitting
        
            self.kl_w = 1 / (1 + loss_huber.detach().item())
            self.hu_w = 1 / (1 + loss_kl.detach().item())
            total_loss = self.kl_w*loss_kl + self.hu_w*loss_huber + reg_loss

            total_loss.backward()
            self.optimizer.step()


            self.kl_losses.append(loss_kl.item())
            self.huber_losses.append(loss_huber.item())
            self.total_losses.append(total_loss.item())
