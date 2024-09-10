import torch
import math
import numpy as np
from time import time
from scipy.optimize import minimize 
from scipy.optimize import fmin_l_bfgs_b

#############
#Stein Variational Gradient Descent

def svgd(x0, score, step, max_iter=1000, bw=-1, tol=1e-5, verbose=False,
         store=False, kernel = 'Laplace', ada = False): 
    
    backend='torch'
    
    x_type = type(x0)
    width = bw
    
    if backend == 'torch':
        x = x0.detach().clone()
    n_samples, n_features = x.shape
    if store:
        storage = []
        t0 = time()
        timer = []
    alpha = 0.9
    fudge_factor = 1e-6
    historical_grad = 0
    for i in range(max_iter):
        if store:
            storage.append(x.clone())
            timer.append(time() - t0)
        d = (x[:, None, :] - x[None, :, :])
        dists = (d ** 2).sum(axis=-1) #dists: ||x_i-x_j||^2 matrix
        if width == -1:
            h = math.sqrt(dists.mean())*n_samples**(-1/(n_features+4))
            if kernel == 'Laplace':
                bw = h
            elif kernel == 'Gaussian':
                bw == h**2
            else:
                raise TypeError('Wrong kernel')
                
        if kernel == 'Laplace':
            nx = torch.sqrt(dists)
            k = torch.exp(- nx / bw) # Laplace kernel matrix     
            k_der = (d / (nx+ 1e15*torch.eye(n_samples))[:,:,None]) * k[:, :, None] / bw # -Laplace kernel gradient, or directly set diagonal entries to be 0
        elif kernel == 'Gaussian':
            k = torch.exp(- dists / bw / 2) #k: \eta(x_i-x_j) = e^{-||x_i-x_j||^2/(2*bw)}
            k_der = d * k[:, :, None] / bw # -+\nabla \eta(x_i-x_j)       
            
        scores_x = score(x) #-x
        ks = k.mm(scores_x) #ks: [ -\sum \eta(x_i-x_j) \nabla U(x_j) ]
        kd = k_der.sum(axis=0)
        direction = (ks - kd) / n_samples
        
        if ada:
            if i == 0:
                historical_grad = historical_grad + direction ** 2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (direction ** 2)
            adj_grad = np.divide(direction, fudge_factor+np.sqrt(historical_grad))
            x += step * adj_grad
        else:
            x += step * direction
    if store:
        return x, storage, timer
    return x



def nsvgd(x0, score, step, max_iter=1000, bw=-1, tol=1e-5, alpha = 0.5, h = 1, verbose=False,
          store=False, kernel = 'Laplace', ada = False):
    x_type = type(x0)
    width = bw
    backend='torch'
    if backend == 'torch':
        x = x0.detach().clone()
        
    n_samples, n_features = x.shape    
    if store:
        storage = []
        timer = []
        t0 = time()
    #alpha = 0.9
    fudge_factor = 1e-6
    historical_grad = 0
    for i in range(max_iter):
        if store:
            if backend == 'torch':
                storage.append(x.clone())
        d = x[:, None, :] - x[None, :, :] # x_i - x_j
        dists = (d ** 2).sum(axis=-1) #dists: ||x_i-x_j||^2 matrix
        h = 1.059*math.sqrt(dists.mean())*n_samples**(-1/(n_features+4))
        
        if width == -1:
            if kernel == 'Laplace':
                bw = h
            elif kernel =='Gaussian':
                bw = h**2
            else:
                raise TypeError('Wrong kernel')

        if kernel == 'Laplace':
            nx = torch.sqrt(dists)
            k = torch.exp(- nx / bw) # Laplace kernel matrix   
            kh = torch.exp(- nx / h)  # Laplace KDE kernel matrix
            k_der = (d / (nx+ 1e15*torch.eye(n_samples))[:,:,None]) * k[:, :, None] / bw # -Laplace kernel gradient, or set diagonal entries to be 0
            kh_der = (d / (nx+ 1e15*torch.eye(n_samples))[:,:,None]) * kh[:, :, None] / h # -Laplace KDE kernel gradient
        elif kernel =='Gaussian':
            k = torch.exp(- dists / bw / 2) #k: \eta(x_i-x_j) = e^{-||x_i-x_j||^2/(2*bw)}
            kh = torch.exp(- dists /(h**2)/ 2)  # Gaussian KDE kernel matrix   
            k_der = d * k[:, :, None] / bw # -+\nabla \eta(x_i-x_j)        
            kh_der = d * kh[:, :, None] / (h**2) 

        scores_x = score(x)
        rho_eta = kh.sum(axis=0)/ n_samples #/ (h**n_features) 
        rho_mtx = (rho_eta[:,None].mm(rho_eta[None,:])).pow(alpha).reciprocal()
        term1 = (rho_mtx[:,:,None] * k_der).sum(axis = 0)
        term2 = (rho_mtx * k).mm(alpha * (kh_der.sum(axis = 0)/ n_samples) / rho_eta[:,None])
        term3 = -(rho_mtx * k).mm(scores_x)
        direction = -(term1 + term2 + term3) / n_samples 
        
        if ada:
            if i == 0:
                historical_grad = historical_grad + direction ** 2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (direction ** 2)
            adj_grad = np.divide(direction, fudge_factor+np.sqrt(historical_grad))
            x += step * adj_grad
        else:
            x += step * direction
    if store:
        return x, storage, timer
    return x

#############
#MMD discrepency with lbfgs algo


def gaussian_kernel(x, y, sigma):
    d = (x[:, None, :] - y[None, :, :])
    dists = (d ** 2).sum(axis=-1)
    return torch.exp(- dists / sigma / 2)

def mmd_lbfgs(x0, target_samples, bw=1, max_iter=10000, tol=1e-5,
              store=False,verbose=False):
    '''Sampling by optimization of the MMD
    This uses target samples from a base distribution and
    returns new samples by minimizing the maximum mean discrepancy.
    Parameters
    ----------
    x0 : torch.tensor, size n_samples x n_features
        initial positions
    target_samples : torch.tensor, size n_samples x n_features
        Samples from the target distribution
    bw : float
        bandwidth of the stein kernel
    max_iter : int
        max numer of iters
    tol : float
        tolerance for L-BFGS
    Returns
    -------
    x: torch.tensor
        The final positions
    References
    ----------
    M.Arbel, A.Korba, A.Salim, A.Gretton. Maximum mean discrepancy
    gradient flow, Neurips, 2020.
    '''
    x = x0.clone().detach().numpy()
    n_samples, p = x.shape
    k_yy = gaussian_kernel(target_samples, target_samples, bw).sum().item()
    if store:
        class callback_store():
            def __init__(self):
                self.t0 = time()
                self.mem = []
                self.timer = []

            def __call__(self, x):
                self.mem.append(np.copy(x))
                self.timer.append(time() - self.t0)

            def get_output(self):
                storage = [torch.tensor(x.reshape(n_samples, p),
                                        dtype=torch.float32)
                           for x in self.mem]
                return storage, self.timer
        callback = callback_store()
    else:
        callback = None

    def loss_and_grad(x_numpy):
        x_numpy = x_numpy.reshape(n_samples, p)
        x = torch.tensor(x_numpy, dtype=torch.float32)
        x.requires_grad = True
        k_xx = gaussian_kernel(x, x, bw).sum()
        k_xy = gaussian_kernel(x, target_samples, bw).sum()
        loss = k_xx - 2 * k_xy + k_yy
        loss.backward()
        grad = x.grad
        return loss.item(), np.float64(grad.numpy().ravel())

    t0 = time()
    x, f, d = fmin_l_bfgs_b(loss_and_grad, x.ravel(), maxiter=max_iter,
                            factr=tol, epsilon=1e-12, pgtol=1e-10,
                            callback=callback)
    if verbose:
        print(verbose)               
        print('Took %.2f sec, %d iterations, loss = %.2e' %(time() - t0, d['nit'], f))
    output = torch.tensor(x.reshape(n_samples, p), dtype=torch.float32)
    if store:
        storage, timer = callback.get_output()
        return output, storage, timer
    return output
    
    
    
##########
# Kernel Stein Discrepency with lbfgs algo
def gaussian_stein_kernel_single(x, score_x, sigma, return_kernel=False):
    """Compute the Gaussian Stein kernel between x and x
    Parameters
    ----------
    x : torch.tensor, shape (n, p)
        Input particles
    score_x : torch.tensor, shape (n, p)
        The score of x
    sigma : float
        Bandwidth
    return_kernel : bool
        whether the original kernel k(xi, xj) should also be returned
    Return
    ------
    stein_kernel : torch.tensor, shape (n, n)
        The linear Stein kernel
    kernel : torch.tensor, shape (n, n)
        The base kernel, only returned id return_kernel is True
    """
    _, p = x.shape
    # Gaussian kernel:
    norms = (x ** 2).sum(-1)
    dists = -2 * x @ x.t() + norms[:, None] + norms[None, :]
    k = (-dists / 2 / sigma).exp()

    # Dot products:
    diffs = (x * score_x).sum(-1, keepdim=True) - (x @ score_x.t())
    diffs = diffs + diffs.t()
    scalars = score_x.mm(score_x.t())
    der2 = p - dists / sigma
    stein_kernel = k * (scalars + diffs / sigma + der2 / sigma)
    if return_kernel:
        return stein_kernel, k
    return stein_kernel


def ksdd_lbfgs(x0, score, kernel='gaussian', bw=1.,
               max_iter=10000, tol=1e-12, beta=.5,
               store=False, verbose=False):
    '''Kernel Stein Discrepancy descent with L-BFGS
    Perform Kernel Stein Discrepancy descent with L-BFGS.
    L-BFGS is a fast and robust algorithm, that has no
    critical hyper-parameter.
    Parameters
    ----------
    x0 : torch.tensor, size n_samples x n_features
        initial positions
    score : callable
        function that computes the score
    kernl : 'gaussian' or 'imq'
        which kernel to choose
    max_iter : int
        max numer of iters
    bw : float
        bandwidth of the stein kernel
    tol : float
        stopping criterion for L-BFGS
    store : bool
        whether to stores the iterates
    verbose: bool
        wether to print the current loss
    Returns
    -------
    x: torch.tensor, size n_samples x n_features
        The final positions
    References
    ----------
    A.Korba, P-C. Aubin-Frankowski, S.Majewski, P.Ablin.
    Kernel Stein Discrepancy Descent
    International Conference on Machine Learning, 2021.
    '''
    x = x0.clone().detach().numpy()
    n_samples, p = x.shape
    if store:
        class callback_store():
            def __init__(self):
                self.t0 = time()
                self.mem = []
                self.timer = []

            def __call__(self, x):
                self.mem.append(np.copy(x))
                self.timer.append(time() - self.t0)

            def get_output(self):
                storage = [torch.tensor(x.reshape(n_samples, p),
                                        dtype=torch.float32)
                           for x in self.mem]
                return storage, self.timer
        callback = callback_store()
    else:
        callback = None

    def loss_and_grad(x_numpy):
        x_numpy = x_numpy.reshape(n_samples, p)
        x = torch.tensor(x_numpy, dtype=torch.float32)
        x.requires_grad = True
        scores_x = score(x)
        if kernel == 'gaussian':
            stein_kernel = gaussian_stein_kernel_single(x, scores_x, bw)
        elif kernel == 'imq':
            stein_kernel = imq_kernel(x, x, scores_x, scores_x, bw, beta=beta)
        else:
            stein_kernel = linear_stein_kernel(x, x, scores_x, scores_x)
        loss = stein_kernel.sum()
        loss.backward()
        grad = x.grad
        return loss.item(), np.float64(grad.numpy().ravel())

    t0 = time()
    x, f, d = fmin_l_bfgs_b(loss_and_grad, x.ravel(), maxiter=max_iter,
                            factr=tol, epsilon=1e-12, pgtol=1e-10,
                            callback=callback)
    if verbose:
        print('Took %.2f sec, %d iterations, loss = %.2e' %
              (time() - t0, d['nit'], f))
    output = torch.tensor(x.reshape(n_samples, p), dtype=torch.float32)
    if store:
        storage, timer = callback.get_output()
        return output, storage, timer
    return output
