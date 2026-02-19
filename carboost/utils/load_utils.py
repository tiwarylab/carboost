from carboost.synapse import get_KDE
import numpy as np
from tqdm import tqdm
import pickle
from importlib import resources as importlib_resources
from pathlib import Path

def _resolve_resource_dir(path_resources, subdir):
    if path_resources is None:
        return importlib_resources.files("carboost").joinpath("resources", "CD8alpha", subdir)
    return Path(path_resources)


def load_rMSA_AF2_KDEs(hinge_sequence_lengths,bandwidth,path_resources=None,chemical_bias=True):
    
    """
    A function to load the rMSA AF2 based distributions of end-to-end distances along z axis.
    """
    probab_cars = {}
    max_zvals = {}

    rMSA_data = [3,8,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,51,56,61,76]
    if not np.asarray([hlen in rMSA_data for hlen in np.unique(np.asarray(hinge_sequence_lengths))]).all():
        raise ValueError(f"rMSA AF2 data is only present for CD8alpha derived hinges with the following sequence length {[str(hh) for hh in rMSA_data]}")
 

    xx=np.linspace(0,25,num=500)
    x_vals=np.reshape(xx,(xx.shape[0],1))

    resource_dir = _resolve_resource_dir(path_resources, "rMSA_AF2")

    for hinge_len in tqdm(hinge_sequence_lengths):
        probab_cars[hinge_len]={}
        
        if hinge_len <= 19:
            dist = np.load(resource_dir / f"ze2e_8H{hinge_len}_wt.npy")
            dist = np.reshape(dist,(dist.shape[0],1))
        else:
            if chemical_bias:
                dist = np.load(resource_dir / f"ze2e_8H{hinge_len}_mt.npy")
                dist = np.reshape(dist,(dist.shape[0],1))
            else:
                dist = np.load(resource_dir / f"ze2e_8H{hinge_len}_wt.npy")
                dist = np.reshape(dist,(dist.shape[0],1))
        
        probab_cars[hinge_len]['kdes']= get_KDE(yval=dist,xval=x_vals,bandwidth=bandwidth).reshape(1,-1)
        probab_cars[hinge_len]['xval']= x_vals
        max_zvals[hinge_len]=np.ceil(np.max(dist))
    
    print(f"Please note that these distribution were calculated from reduce MSA approach on AlphaFold2.\nThese KDEs are not thermodynamically weighted \nKindly, use this data for an estimate.")
    print(f"For hinge sequence length less than 40 amino acids, rMSA AF2 with chemical bias can give a reasonable estimate.")

    return probab_cars, max_zvals

def load_af2rave_KDEs(hinge_sequence_lengths,path_resources=None):
    """
    A function to load the af2rave based distributions of end-to-end distances along z axis.
    """
    if not np.asarray([hlen in [15,27,33,40,47,51,56,61,76,91] for hlen in np.unique(np.asarray(hinge_sequence_lengths))]).all():
        raise ValueError("af2rave data is only present for CD8alpha derived hinges with the following sequence length [15,27,33,40,47,51,56,61,76,91]")

    
    xx=np.linspace(0,25,num=500)
    x_vals=np.reshape(xx,(xx.shape[0],1))
    
    probab_cars={}
    max_zvals={}

    resource_dir = _resolve_resource_dir(path_resources, "af2rave")

    with open(resource_dir / "max_hinge_values.pkl","rb") as f:
        max_hinge_zvalues = pickle.load(f)

    for hinge_len in tqdm(hinge_sequence_lengths):
        probab_cars[hinge_len]={}
        prb = np.load(resource_dir / f"probab_8H{hinge_len}.npy")
        probab_cars[hinge_len]['kdes'] = prb
        probab_cars[hinge_len]['xval'] = x_vals
        max_zvals[hinge_len] = max_hinge_zvalues[hinge_len]
    
    print(f"Please note that these distribution were calculated from long biased well-tempered metadynamics trajectory.\nConvergence may not have achieved. Further, KDEs were built for certain bandwidth.\nKindly, use this data for an estimate and cross check with rMSA AF2 data.")
    print(f"For hinge sequence length less than 40 amino acids, rMSA AF2 with chemical bias can give a reasonable estimate.")

    return probab_cars, max_zvals
