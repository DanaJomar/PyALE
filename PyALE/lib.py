import  numpy as np

def cmds(D, k=2):
    """
    https://en.wikipedia.org/wiki/Multidimensional_scaling#Classical_multidimensional_scaling
    http://www.nervouscomputer.com/hfs/cmdscale-in-python/
    """
    # Number of points                                                                        
    n = len(D)
    if k > (n-1):
        raise Exception('k should be an integer <= len(D) - 1' )
    # (1) Set up the squared proximity matrix
    D_double = D**2
    # (2) Apply double centering: using the centering matrix
    # centering matrix
    center_mat = np.eye(n) - np.ones((n, n))/n
    # apply the centering
    B = -(1/2) * center_mat.dot(D_double).dot(center_mat)
    # (3) Determine the m largest eigenvalues 
    # (where m is the number of dimensions desired for the output)
    # extract the eigenvalues
    eigenvals, eigenvecs = np.linalg.eigh(B)
    # sort descending
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    # (4) Now, X=eigenvecs.dot(eigen_sqrt_diag), where eigen_sqrt_diag = diag(sqrt(eigenvals))
    eigen_sqrt_diag = np.diag(np.sqrt(eigenvals[0:k]))
    ret = eigenvecs[:,0:k].dot(eigen_sqrt_diag)
    return(ret)
