import ot
import numpy as np

from .util import distribution_minmax, distribution_normalize

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

def ot_transport_mapping_linear(Xs, Xt, mu=1e0, eta=1e-8, bias=True, max_iter=20, verbose=True):
    ot_mapping_linear = ot.da.MappingTransport(
        kernel='linear',
        mu=mu,
        eta=eta,
        bias=bias,
        max_iter=max_iter,
        verbose=verbose
    )
    ot_mapping_linear.fit(Xs=Xs, Xt=Xt)

    transp_Xs_mapping_linear = ot_mapping_linear.transform(Xs=Xs)
    return transp_Xs_mapping_linear

def ot_transport_mapping_gaussian(Xs, Xt, eta=1e-5, mu=1e-1, bias=True, sigma=1,
    max_iter=10, verbose=True):
    ot_mapping_gaussian = ot.da.MappingTransport(
        kernel='gaussian',
        eta=eta,
        mu=mu,
        bias=bias,
        sigma=sigma,
        max_iter=max_iter,
        verbose=verbose
    )
    ot_mapping_gaussian.fit(Xs=Xs, Xt=Xt)

    transp_Xs_mapping_gaussian = ot_mapping_gaussian.transform(Xs=Xs)
    return transp_Xs_mapping_gaussian

def restore_from_normal(arr, mu, sigma):
    return arr * sigma + mu

def restore_from_minmax(arr, minval, maxval):
    return arr * (maxval - minval) + minval

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