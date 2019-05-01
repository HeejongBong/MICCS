import time
import numpy as np
from miccs.optimize import coord_desc

def fit(observation, dims_pop, lambda_ridge, lambda_glasso,
        ths=1e-4, max_iter=10000, verbose=False, verb_period=10):
    
    # check on observation and dims_pop
    if observation[0].ndim == 2:
        observation = np.concatenate(observation, axis=0)
        dims_pop = np.array([obs.shape[0] for obs in observations]).astype(int)
    elif observation[0].ndim == 1:
        assert(observation.shape[0] == np.sum(dims_pop))
        dims_pop = np.array(dims_pop).astype(int)
    else:
        raise
        
    # check on lambda_ridge
    if isinstance(lambda_ridge, (int, float)):
        lambda_ridge = np.full(np.shape(dims_pop), lambda_ridge)
    else:
        assert(np.shape(lambda_ridge) == dims_pop.shape)  
    
    # check on lambda_glasso                   
    assert(lambda_glasso.shape == (dims_pop.shape[0], dims_pop.shape[0]))
    
    # initial setting
    start_time = time.time()
    console = open('/dev/stdout', 'w')
    
    # run
    converged, precision, correlation, latent, weight = \
        coord_desc(observation, dims_pop, lambda_ridge, lambda_glasso,
                   ths, max_iter, verbose, verb_period)       
    if converged:
        console.write("Converged, total lapse: %fs\n"
                                 %(time.time() - start_time))
    else:
        console.write("Did not converge, total lapse: %fs\n"
                                 %(time.time() - start_time))
    
    weights = [weight[np.sum(dims_pop[:i]):np.sum(dims_pop[:i+1])] for i, _
               in enumerate(dims_pop)]
    
    return converged, precision, correlation, latent, weights 

def imshow(image, vmin=None, vmax=None, cmap='RdBu'):
    assert(image.ndim == 2)
    assert(image.shape[0] == image.shape[1])

    # get vmin, vmax
    if vmin is None:
        vmin = -np.max(np.abs(image))
    if vmax is None:
        vmax = np.max(np.abs(image))
    
    # get figure    
    import matplotlib.pyplot as plt
    plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)