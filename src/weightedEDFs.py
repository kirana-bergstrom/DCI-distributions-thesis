import numpy as np # Heavily used
from scipy.integrate import quad as squad # For numerical integration of exact target CDF (note: quadrature can fail)
import cvxopt as cvx
from cvxopt.solvers import qp # For quadratic optimization program
import collections # for checking if CDF function expressions are passed
from scipy.stats import gaussian_kde as GKDE # for checking against RN weights

'''
This module constructs the basic elements required to formulate the constrained quadratic optimization
problem presented in

  Optimal $L_2$-norm empirical importance weights for the change of probability measure,
  S. Amaral, D. Allaire, K. Willcox, Stat. Comput., 27, 625-643 (2017)
  
  URL: https://link.springer.com/article/10.1007/s11222-016-9644-3
  
The goal is to use this approach to solve Data-Consistent Inversion (DCI) problems as presented in

  Combining Push-Forward Measures and Bayes' Rule to Construct Consistent Solutions to Stochastic Inverse Problems, 
  T. Butler, J. Jakeman, T. Wildey, SIAM J. Sci. Comput., 40(2), A984–A1011 (2018)
  
  URL: https://epubs.siam.org/doi/abs/10.1137/16M1087229

See the accompanying notebook: weighted-CDF-DCI-ideas.ipynb for more details
'''

def compute_H(sample_set, w_init=None, bbox=None): 
    '''
    This function computes the H matrix on a given set of samples.
    
    :param sample_set: a numpy array of dimension n-by-d (n=# samples, d=dimension of samples)
    :param w_init: [optional] a numpy array of dimension n-by-1 defining initial weights for the samples
    :param bbox: [optional] a numpy array of dimension d-by-2 defining the lower and upper bounds of
                 integration (bbox[i,0] and bbox[i,1] are the $i+1$th lower and upper 
                 bounds, respectively)
                 
    
    :rtype: numpy array
    :returns: n-by-n array H
    '''

    n = np.size(sample_set[:,0])   # n is number of samples
    d = np.size(sample_set[0,:])   # d is dimension of samples
    H = np.zeros((n,n))

    if w_init is None:

        w_init = np.ones((n,1))

    if bbox is None: # Well, I guess we better make one then
    
        bbox = np.zeros([d,2]) # This will be the bounding box
        
        for i in range(0,d):
            
            bbox[i,0] = np.min(sample_set[:,i])
            bbox[i,1] = np.max(sample_set[:,i])          

    for i in range(0,n):
                    
        H[i,:] = w_init[i,0]*w_init[:,0]

        for k in range(0,d):
            
            temp = (bbox[k,1] - np.maximum(sample_set[i,k],sample_set[:,k])) / (bbox[k,1] - bbox[k,0])

            H[i,:] *= temp

    H *= (1/n**2)

    return H
   


def compute_b(sample_set_1, sample_set_2=None, w_1=None, w_2=None, targ_CDF=None, bbox=None): 
    '''
    This function computes the b vector involving the difference of two CDFs defined by an empirical
    CDF and a "target" CDF either described by an empirical CDF (so sample_set_2) or "exactly" (so no
    sample_set_2 but using targ_CDF).
    
    :param sample_set_1: a numpy array of dimension n-by-d (n=# samples, d=dimension of samples)
    :param sample_set_2: [optional] a numpy array of dimension m-by-d (m=# samples,
                        d=dimension of samples) for the target distribution
    :param w_1: [optional] a numpy array of dimension n-by-1 matching sample_set_1 that gives weights
                of this initial distribution of samples when they are not i.i.d.
    :param w_2: [optional] a numpy array of dimension m-by-1 matching sample_set_2 that gives weights
                of these target distribution samples when they are not i.i.d.
    :param targ_CDF: [optional] a CDF function like those given in scipy that we can integrate
                     using some sort of quadrature
    :param bbox: a numpy array of dimension d-by-2 defining the lower and upper bounds of
                 integration (bbox[i,0] and bbox[i,1] are the $i+1$th lower and upper
                 bounds, respectively)
                 
    :rtype: numpy array
    :returns: n-by-1 array b
    '''

    exact_target = True

    n = np.size(sample_set_1[:,0]) # n is the number of samples in set 1
    d = np.size(sample_set_1[0,:]) # d is the dimension of samples in set 1
    b = np.zeros((n,1))
    
    if w_1 is None:
        
        w_1 = np.ones((n,1))
    

    # Check if sample set 2 is given
    if sample_set_2 is not None:
        
        exact_target = False
        m = np.size(sample_set_2[:,0]) # m is the number of samples in set 2
        
        if w_2 is None:
            w_2 = np.ones((m,1))
    
    if bbox is None: # Need to construct this if not given 
        
        bbox = np.zeros([d,2])
        
        if exact_target == False:
            
            for i in range(0,d):

                # not doing this efficiently, just for clarity
                temp1 =  np.min(sample_set_1[:,i])
                temp2 =  np.min(sample_set_2[:,i])
                bbox[i,0] = np.min([temp1,temp2])

                temp1 =  np.max(sample_set_1[:,i])
                temp2 =  np.max(sample_set_2[:,i])
                bbox[i,1] = np.max([temp1,temp2])  
        
        else:
            for i in range(0,d):
                bbox[i,0] = np.min(sample_set_1[:,i])
                bbox[i,1] = np.max(sample_set_1[:,i])
    # Uses exact target CDF if no sample_set_2 is not given
    if exact_target:

        for i in range(n):
            b[i] = w_1[i]

            for k in range(d):
                (temp,_) = squad(targ_CDF,sample_set_1[i,k],bbox[k,1]) 
                b[i] *= temp / (bbox[k,1]-bbox[k,0])

        b *= 1/n

    else:
        
        for i in range(n):
            temp = w_2[:,0]

            for k in range(d):
                temp = temp * (bbox[k,1] - np.maximum(sample_set_1[i,k], sample_set_2[:,k])) / (bbox[k,1] - bbox[k,0])
                
            b[i] += np.sum(temp)
            b[i] *= w_1[i]
        
        b *= (1./n)*(1./m)

    return b
    

def compute_optimal_w(H, b, w_initial=None, bins=None, iws=None):
    # This function computes and returns the optimal weights corresponding to matrix H and vector b
    # Assuming w_initial is given as a numpy array of dimension (n,1)
    # cons indicates which kind of constraint we implement:
    #     - b: both the w_initial * w = n and 1 * w = n constraints,
    #     - j: just the w_initial * w = n,
    #     - r: just the 1 * w = n constraint
    
    n = np.size(H[:,0])
    
    if w_initial is None:
        
        A = cvx.matrix(np.ones((1,n)))
        b_constraint = cvx.matrix(float(n))
        
        if bins is not None:
            
            unique = np.unique(bins)
            
            n_clusters = len(unique)
            # print(n_clusters)
            A = np.zeros((6000-n_clusters+1, 6000))
            A[0,:] = np.ones((6000,))
            row = 1
            inds = np.array(range(0,6000))
            
            for c in range(n_clusters):

                labs = bins[bins == c]
                ilabs = inds[bins == c]
                nsamp = len(labs)
                start_cluster = ilabs[0]
                for nn in range(1,nsamp):
                    A[row,start_cluster] = 1
                    A[row,ilabs[nn]] = -1
                    row = row+1
                    
            A = cvx.matrix(A)
            b_constraint = np.zeros(np.shape(A)[0])
            b_constraint[0] = 6000
            b_constraint = cvx.matrix(b_constraint)
        
    else:
        
#         if cons == 'b':
#             A = cvx.matrix(np.vstack((np.ones((1,n)),w_initial.T)))
#             b_constraint = cvx.matrix(n*np.ones((2,1)))
#         elif cons == 'j':
#             A = cvx.matrix(w_initial.T)
#             b_constraint = cvx.matrix(n*np.ones((1,1)))
#         else:
#             A = cvx.matrix(np.ones((1,n)))
#             b_constraint = cvx.matrix(n*np.ones((1,1)))
        A = cvx.matrix(np.vstack((np.ones((1,n)),w_initial.T)))
        b_constraint = cvx.matrix(n*np.ones((2,1)))
    
    H = cvx.matrix(H)
    b = cvx.matrix(b)

    G = cvx.matrix(0.0, (n,n))
    G[::n+1] = -1.0
    h = cvx.matrix(0.0, (n,1))

    cvx.solvers.options['show_progress'] = False
    opt_soln = qp(H, -b, G, h, A, b_constraint)
#     opt_soln = qp(H, -b, G, h, A, b_constraint, initvals=cvx.matrix(iws))
    #opt_soln = qp(H, -b, G, h)
    
    w_opt = np.asarray(opt_soln['x'])
    
    return w_opt

def error(sample_set_1, sample_set_2, w1 = None , w2 = None, bbox=None):
    #This function computes the $L_2$-error between two sample sets with associated weight vectors
    
    n = np.size(sample_set_1[:,0])
    m = np.size(sample_set_2[:,0])
    d = np.size(sample_set_1[0,:])
    
    if w1 is None: # no weights for sample_set_1?
        
        w1 = np.ones((n,1)) # then make all the same
        
    if w2 is None: # no weights for sample_set_2?
        
        w2 = np.ones((m,1)) # then make all the same

    if bbox is None:
        
        bbox = np.zeros([d,2])
        
        for i in range(0,d):

            temp1 =  np.min(sample_set_1[:,i])
            temp2 =  np.min(sample_set_2[:,i])
            bbox[i,0] = np.min([temp1,temp2])

            temp1 =  np.max(sample_set_1[:,i])
            temp2 =  np.max(sample_set_2[:,i])
            bbox[i,1] = np.max([temp1,temp2])  
    
    b = compute_b(sample_set_1, sample_set_2, bbox=bbox)       
    
    Hprop = compute_H(sample_set_1, bbox=bbox)        
    
    Htarg = compute_H(sample_set_2, bbox=bbox)  
    
    hw = np.dot(Hprop,w1)  
    
    w2h = np.dot(w2.T,Htarg)
    
    e = np.dot(w1.T, hw) - 2*np.dot(w1.T,b) + np.dot( w2h ,w2)
    
    return np.sqrt(e, bbox) # Take the square root to get the $L_2$ error

def compute_RN(sample_set_1, sample_set_2):
    
    # estimate proposal density on output
    q_prop_kde = GKDE(sample_set_1.T)

    # estimate target density on output
    q_targ_kde = GKDE(sample_set_2.T)

    # get RN weights on the sample space
    rn_w = q_targ_kde(sample_set_1.T) / q_prop_kde(sample_set_1.T)

    return rn_w
