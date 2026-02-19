import numpy as np
from sklearn.neighbors import KernelDensity


def get_KDE(yval,xval,weights=None,bandwidth='silverman',kernel='gaussian'):
    """
    KDE density estimation function. The function is written to account for weight from biased simulation as well.

    yval: An array (M, 1) containing samples of an observable. M is number of observations
    xval: An array (N, 1) containing the grid values for which KDE is estimated. N is the number of grid points.
    weights: An array (M,1) containing the weights for corresponding observation.
    bandwidth: Bandwidth used to fit kernels. It can be a Float point variable or 
               a strung containg the method of estimating bandwidths {'silverman','scott'}
    kernel: Type of kernel being deposited to estimate density. 
            It can contain {'gaussian','tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'}.
    """
    xval_size = xval.shape
    if len(xval_size) == 1:
        xval_checked = np.reshape(xval,(xval.shape[0],1))
    elif len(xval_size) == 2:
        if xval_size[0] == 1:
            xval_checked = xval.T
        elif xval_size[1] == 1:
            xval_checked = xval
        else:
            raise ValueError("The xval should be an array with dimension (N,1)")
    else:
        raise ValueError("The xval should be an array with dimension (N,1)")
    
    yval_size = yval.shape
    if len(yval_size) == 1:
        yval_checked = np.reshape(yval,(yval.shape[0],1))
    elif len(yval_size) == 2:
        if yval_size[0] == 1:
            yval_checked = yval.T
        elif yval_size[1] == 1:
            yval_checked = yval
        else:
            raise ValueError("The yval should be an array with dimension (M,1)")
    else:
        raise ValueError("The yval should be an array with dimension (M,1)") 

    KDEmodel=KernelDensity(kernel=kernel, bandwidth=bandwidth)
    if weights is not None:
        KDEmodel.fit(yval_checked,sample_weight=weights) # To Do: Add a check for weights and make a separate function to check.
    else:
        KDEmodel.fit(yval_checked)
    kde_val=np.exp(KDEmodel.score_samples(xval_checked))
    return kde_val

def get_conv(y1,y2,xv,te): # This was a naive fucntion to convolve
    newyval=np.convolve(y1,y2)
    newxv=xv.tolist()+xv.tolist()
    newxv=np.array(newxv)
    newxv[te:]+=newxv[te-1]
    newnewxv=np.zeros(newyval.shape)
    newnewxv[:te]=newxv[:te]
    newnewxv[te:]=newxv[te+1:]
    return newyval, newnewxv

def get_conv2(y1,y2,x1,x2): # This is a better version for the convolvution function
    newyval=np.convolve(y1,y2) # To Do: To absorb the other function and use only this function.
    te=x1.shape[0]
    newxv=np.concatenate((x1,x2))
    newxv[te:]+=newxv[te-1]
    newnewxv=np.concatenate((newxv[:te],newxv[te+1:]))
    return newyval, newnewxv

def integ(y,x): # To Do: Go back to simpsons method.
    integ_v=np.trapz(y,x)
    return integ_v

def get_impulse(xval,value):
    indx=np.where(np.abs(xval-value)==np.min(np.abs(xval-value)))[0]
    impulse=np.zeros(xval.shape)
    impulse[indx]=1
    return impulse


def get_probab(px,xv,inds):
    tot=integ(px,xv)
    pval=integ(px[inds],xv[inds])
    pval=pval/tot

    return pval


def get_Xinterested(xvals,upper,lower):
    mask = (xvals > lower) & (xvals < upper)
    inds = np.flatnonzero(mask)
    x_interested = xvals[inds]
    return x_interested, inds


def get_regions_probabilities(prob_avg, region_split, xvals):
    """Region specific probability calculation.
    """
    probab = []
    if len(xvals.shape) > 1:
        raise ValueError("xvals must be a 1D array.")

    edges = list(region_split)
    _, line1, line2, max_val = edges

    if max_val - line2 <= 0:
        edges = edges[:-1]
    elif max_val - line2 > 0:
        if np.abs(line2 - max_val) < (np.abs(line2 - line1))/2:
            edges = edges[:-1]

    for i in range(len(edges) - 1):
        low, up = edges[i], edges[i + 1]
        _, indices = get_Xinterested(xvals, up, low)
        if indices.size == 0:
            probab.append(0.0)
        else:
            probab.append(float(get_probab(prob_avg, xvals, indices)))
    return probab


def get_estimators(probab):
    """
    The Phi value calculator
    """
    if len(probab) == 3:
        pA, pS, pO = probab
        dG2 = -np.log(pS / pO)
        dG1 = -np.log(pS / pA)
        phi = dG2 + dG1
    elif len(probab) == 2:
        pA, pS = probab
        dG2 = None
        dG1 = -np.log(pS / pA)
        phi = dG1
    else:
        raise ValueError("Expected 2 or 3 region probabilities.")
    return dG1, dG2, phi


def get_estimators_delta(probab, probab_delta):
    """The Phi value error calculator based ont the manuscript.
    """
    if len(probab) == 3:
        pA, pS, pO = probab
        pAd, pSd, pOd = probab_delta

        term2 = 2 * (pSd / pS) ** 2 + (pOd / pO) ** 2 if pS != 0 else (pOd / pO) ** 2
        term1 = 2 * (pSd / pS) ** 2 + (pAd / pA) ** 2 if pS != 0 else (pAd / pA) ** 2

        dG2_err = ((pSd / pS) ** 2 + (pOd / pO) ** 2) ** 0.5 if pS != 0 else abs(pOd / pO)
        dG1_err = ((pSd / pS) ** 2 + (pAd / pA) ** 2) ** 0.5 if pS != 0 else abs(pAd / pA)
        phi_err = (term1 + term2) ** 0.5
    elif len(probab) == 2:
        pA, pS = probab
        pAd, pSd = probab_delta
        dG2_err = None
        dG1_err = ((pSd / pS) ** 2 + (pAd / pA) ** 2) ** 0.5 if pS != 0 else abs(pAd / pA)
        phi_err = dG1_err
    else:
        raise ValueError("Expected 2 or 3 region probabilities.")
    return dG1_err, dG2_err, phi_err
