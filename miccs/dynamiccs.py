import numpy as np
import miccs.miccs as mx

def fit(populations, lambda_ridge, lambda_glasso_cross, offset_cross,
        lambda_glasso_auto=0, offset_auto=1, 
        ths=1e-4, max_iter=10000, verbose=False, verb_period=10,
        data_mode='spike', bin_width=40, adj_sign=True):
    
    # get observation
    if data_mode == 'spike':
        bin_num = int(populations[0].shape[0]/bin_width)
        observation = np.concatenate(
            [np.sum(pop[:bin_num*bin_width].reshape((bin_num, -1)+pop.shape[1:]),
                    axis=1).reshape((-1, pop.shape[2]))
             for pop in populations], axis=0)
        
    if data_mode == 'count':
        bin_num = populations[0].shape[0]
        observation = np.concatenate([pop.reshape((-1,pop.shape[2]))
                                      for pop in populations], axis=0)
    
    # get indices
    dims_pop = np.concatenate([np.repeat(pop.shape[1], bin_num) 
                               for pop in populations]).astype(int)
    
    # get full_graph
    lambda_glasso_auto = _generate_lambda_glasso(bin_num, lambda_glasso_auto, 
                                                 offset_auto, auto=True)
    lambda_glasso_cross = _generate_lambda_glasso(bin_num, lambda_glasso_cross,
                                                  offset_cross)
    
    lambda_glasso = np.array(np.block(
        [[lambda_glasso_auto if j==i else lambda_glasso_cross
          for j, _ in enumerate(populations)]
         for i, _ in enumerate(populations)]))
        
    # run
    converged, precision, correlation, latent, weights =\
        mx.fit(observation, dims_pop, lambda_ridge, lambda_glasso,
               ths, max_iter, verbose, verb_period)
    
    # adjust sign
    if converged and adj_sign:
        adj_sign = np.cumprod(np.concatenate([
            np.ones((1)),
            np.sign(np.sign(correlation[np.arange(0, bin_num-1),
                                        np.arange(1,bin_num)])+0.5),
            np.ones((1)),
            np.sign(np.sign(correlation[np.arange(bin_num, 2*bin_num-1),
                                        np.arange(bin_num+1, 2*bin_num)])+0.5)
        ]))

        correlation = correlation *\
            adj_sign.reshape((2*bin_num, 1)) * adj_sign.reshape((1, 2*bin_num))
        precision = precision *\
            adj_sign.reshape((2*bin_num, 1)) * adj_sign.reshape((1, 2*bin_num))
        latent = latent * adj_sign.reshape((2*bin_num, 1))
        weights = [w*sgn for w, sgn in zip(weights, adj_sign)]

        adj_sign = np.cumprod(np.concatenate([
            np.ones((bin_num)),
            np.sign(np.sign(np.sum(np.sign(
                correlation[np.arange(0,bin_num-1), np.arange(bin_num, 2*bin_num-1)]
            )))+0.5).reshape((1)),
            np.ones((bin_num-1))
        ]))

        correlation = correlation *\
            adj_sign.reshape((2*bin_num, 1)) * adj_sign.reshape((1, 2*bin_num))
        precision = precision *\
            adj_sign.reshape((2*bin_num, 1)) * adj_sign.reshape((1, 2*bin_num))
        latent = latent * adj_sign.reshape((2*bin_num, 1))
        weights = [w*sgn for w, sgn in zip(weights, adj_sign)]

    return converged, precision, correlation, latent, weights

def imshow(image, vmin=None, vmax=None, cmap='RdBu', time=None, time_stim=0):
    assert(image.ndim == 2)
    assert(image.shape[0] == image.shape[1])
    
    # get vmin, vmax
    if vmin is None:
        vmin = -np.max(np.abs(image))
    if vmax is None:
        vmax = np.max(np.abs(image))
            
    # get figure   
    mx.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    
    if time:
        assert(len(time) == 2)
        
        import matplotlib.pyplot as plt
        bin_num = image.shape[0]
        plt.plot([-0.5,bin_num-0.5],[-0.5,bin_num-0.5], linewidth = 0.3, color='black')
        plt.gca().set_xticks(np.linspace(-0.5, bin_num-0.5,
                                         int((time[1]-time[0])/.5+1)))
        plt.gca().set_xticklabels(np.linspace(time[0], time[1],
                                              int((time[1]-time[0])/.5+1)))
        plt.gca().set_yticks(np.linspace(-0.5, bin_num-0.5,
                                         int((time[1]-time[0])/.5+1)))
        plt.gca().set_yticklabels(np.linspace(time[0], time[1],
                                              int((time[1]-time[0])/.5+1)))
        if time_stim is not None:
            plt.plot([-0.5,bin_num-0.5],
                     [((time_stim-time[0])/(time[1]-time[0]))*bin_num-0.5]*2, 
                     linewidth=0.3, color='red')
            plt.plot([((time_stim-time[0])/(time[1]-time[0]))*bin_num-0.5]*2,
                     [-0.5,bin_num-0.5], 
                     linewidth=0.3, color='red')
            
def _generate_lambda_glasso(bin_num, lambda_glasso, offset, auto=False):
    lambda_glasso_out = np.full((bin_num, bin_num), -1) + (1+lambda_glasso) * \
           (np.abs(np.arange(bin_num) - np.arange(bin_num)[:,np.newaxis]) <= offset)
    if auto:
        lambda_glasso_out[np.arange(bin_num), np.arange(bin_num)] = 0
    return lambda_glasso_out