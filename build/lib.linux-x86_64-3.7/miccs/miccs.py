import time, sys
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from miccs_core import optimize_once

def fit(observations, full_graph, smooth_graph,
        lambda_glasso, lambda_ridge, lambda_smooth, 
        indices=np.zeros((1,)), ths=1e-4, max_iter=10000, 
        weights_init=None, verbose=True):

    if observations[0].ndim == 2:
        observation = np.concatenate(observations, axis=0)
        indices = np.cumsum(
            np.concatenate([[0],[obs.shape[0] for obs in observations]])).astype(int)
    
    elif observations[0].ndim == 1:
        assert(observations.shape[0] == indices[-1])
        observation = observations
        observations = [observation[ind_from:ind_to] for ind_from, ind_to
                        in zip(indices[:-1],indices[1:])]
        
    else:
        raise
    
    # initialization
    weights = []
    if weights_init is None:
        for obs in observations:
            weight = np.ones(np.shape(obs)[0])
            weights.append(weight
                           /np.sqrt(np.var(weight @ obs) 
                                    + lambda_ridge * weight @ weight))
    else:
        for weight, obs in zip(weights_init, observations):
            weights.append(weight
                           /np.sqrt(np.var(weight @ obs) 
                                    + lambda_ridge * weight @ weight))          
        
    weight = np.concatenate(weights, axis=0)
    latent = np.array([w @ obs for obs, w in zip(observations, weights)])
    correlation = (np.cov(latent) 
                   * (1 - np.eye(len(observations))) 
                   + np.eye(len(observations)))
    precision = np.array(np.linalg.inv(correlation))
          
    # initial setting
    start_prog = time.time()
    cost_old = np.inf
    console = open('/dev/stdout', 'w')
    
    # iteration
    converged = False
    for n_iter in range(max_iter):
        start_iter = time.time()

        cost = optimize_once(precision, weight, latent, correlation,
                             observation, indices, full_graph, smooth_graph,
                             lambda_glasso, lambda_ridge, lambda_smooth)

        lapse = time.time() - start_iter
        change_cost = cost_old - cost
        
        if verbose:
            clear_output(wait=True)
            display('Iteration %d, cost: %f, change of cost: %f, lapse: %fs'
                    %(n_iter, cost, change_cost, lapse))
            
        if np.isnan(cost):
            break

        if change_cost < ths:
            converged = True
            if verbose:
                console.write("Converged, total lapse: %fs\n"
                                     %(time.time() - start_prog))
            break
        cost_old = cost

    if not converged:
        if verbose:
            console.write("Did not converge, total lapse: %fs\n"
                                 %(time.time() - start_prog))
    
    weights = [weight[ind_from:ind_to] for ind_from, ind_to
               in zip(indices[:-1],indices[1:])]
    
    return converged, precision, weights, latent, correlation

def imshow(image, vmin=None, vmax=None, cmap='RdBu'):
    assert(image.ndim == 2)
    assert(image.shape[0] == image.shape[1])
        
    # get vmin, vmax
    if vmin is None:
        vmin = -np.max(np.abs(image))
    if vmax is None:
        vmax = np.max(np.abs(image))
    
    # get figure    
    plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)