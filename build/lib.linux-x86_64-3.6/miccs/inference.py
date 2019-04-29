import numpy as np
from scipy import ndimage

def h(pvals):
    return 2*(pvals > 0.5).astype(float)

def s(pvals):
    return 1-pvals

def g(pvals):
    return np.minimum(pvals, s(pvals))

def sinv(tdpvals):
    return 1-tdpvals

def sdot(pvals):
    return -1

def fdp_hat(pvals, mask):
    if np.sum(mask) == 0:
        return 0
    else:
        return np.sum(mask*h(pvals)) / np.sum(mask)
    
def score_smooth(pvals, mask, steps_em=5, sigma=1, mux_init=None):
    tdpvals_0 = np.where(mask, g(pvals), pvals)
    tdpvals_1 = sinv(tdpvals_0)
    if mux_init is None:
        mux_init = np.mean(-np.log(tdpvals_0))
    mux = np.full(pvals.shape, mux_init)

    for _ in range(steps_em):
        imputed_logpvals = ((tdpvals_0**(1/mux-1)*(-np.log(tdpvals_0)) +
                             tdpvals_1**(1/mux-1)*(-np.log(tdpvals_1))) /
                            (tdpvals_0**(1/mux-1)+tdpvals_1**(1/mux-1)/(-sdot(tdpvals_1))))

        mux = ndimage.gaussian_filter(imputed_logpvals, sigma)
        
    return mux

def score_blobs(pvals, mask, steps_em=5):
    tdpvals_0 = np.where(mask, g(pvals), pvals)
    tdpvals_1 = sinv(tdpvals_0)
    if mux_init is None:
        mux_init = np.mean(-np.log(tdpvals_0))
    mux = np.full(pvals.shape, mux_init)

    denom_mux = np.ones(pvals.shape)
    denom_mux[:-1,:] += 1; denom_mux[1:,:] += 1
    denom_mux[:,1:] += 1; denom_mux[:,:-1] += 1

    for _ in range(steps_em):
        imputed_logpvals = ((tdpvals_0**(1/mux-1)*(-np.log(tdpvals_0)) +
                             tdpvals_1**(1/mux-1)*(-np.log(tdpvals_1))) /
                            (tdpvals_0**(1/mux-1)+tdpvals_1**(1/mux-1)/(-sdot(tdpvals_1))))

        imputed_mux = np.copy(imputed_logpvals)
        imputed_mux[:-1,:] += imputed_logpvals[1:,:]
        imputed_mux[1:,:] += imputed_logpvals[:-1,:]
        imputed_mux[:,1:] += imputed_logpvals[:,:-1]
        imputed_mux[:,:-1] += imputed_logpvals[:,1:]
        mux = imputed_mux / denom_mux
        
    return mux

def STAR(pvals, alpha = 0.05, prop_carve = 0.2, steps_em = 5,
         score_fn = score_smooth, **kwargs):
    p = pvals.shape
    
    mask = np.full(p, True)    
    boundary = np.full(p, False)
    boundary[0,:] = True; boundary[-1,:] = True
    boundary[:,0] = True; boundary[:,-1] = True
    
    fdp = fdp_hat(pvals, mask)
    R = np.sum(mask)
    R_min = R * (1-prop_carve)
    
    score = score_fn(pvals, mask, steps_em, **kwargs)
    
    while fdp > alpha:
        min_ind = np.unravel_index(
            np.argmin(score + np.where(mask & boundary, 0, np.inf)), 
            mask.shape)
        mask[min_ind] = False
        if min_ind[0] > 0:
            boundary[min_ind[0]-1, min_ind[1]] = True
        if min_ind[0] < p[0]-1:
            boundary[min_ind[0]+1, min_ind[1]] = True
        if min_ind[1] > 0:
            boundary[min_ind[0], min_ind[1]-1] = True
        if min_ind[1] < p[1]-1:
            boundary[min_ind[0], min_ind[1]+1] = True

        fdp = fdp_hat(pvals, mask)
        R = np.sum(mask)

        if R <= R_min:
            score = score_fn(pvals, mask, steps_em, **kwargs)
            R_min = R * (1-prop_carve)

        # if 2 / (1+R) > alpha:
        #     break
            
    return R, fdp, score, mask