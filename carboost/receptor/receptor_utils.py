import re
import numpy as np

def get_coords(file):
    """ Function to parse AF2 predicted pdb files to 
        get a python dictionary that contains `atom` and `coordinate` information.

        file: PDB file from AF2
    """
    res_data = []
    resno = []
    pre_reno = 0
    flg = 0

    with open(file) as fids:
        for linee in fids.readlines():
            if (
                linee[:4] == "ATOM"
                and not re.search(r"^H", linee[12:16].strip())
                and linee[12:16].strip() != "OXT"
            ):
                reno = int(linee[22:26])

                if reno not in resno:
                    if flg > 0:
                        res_data.append(resd)
                    resno.append(reno)
                    resd = {}
                    flg += 1
                    pre_reno = reno

                if reno == pre_reno:
                    atyp = linee[12:16].strip()
                    resd[atyp] = [
                        float(linee[30:38]),
                        float(linee[38:46]),
                        float(linee[46:54]),
                    ]
                    pre_reno = reno

    res_data.append(resd)
    return res_data, resno

def get_hinge_like_region_indices(ss_info, max_gap=2, max_end_non_C=3):
    regions = []
    n=len(ss_info)
    i=0
    while i<n:
        if ss_info[i]!="C":
            i+=1
            continue

        start=i
        j=i

        while j<n:
            if ss_info[j]=="C":
                j+=1
                continue

            k=j
            while k<n and ss_info[k]!="C":
                k += 1

            gap_len = k - j
            if k < n and gap_len <= max_gap:
                j = k                       # -> bridge small non 'C' island
            else:
                break

        regions.append((start, j - 1))
        i = j + 1

    if not regions:
        return None

    start, end=regions[-1]
    tail_len = (n - 1) - end
    if tail_len <= max_end_non_C:
        end = n - 1

    return (start, end)

def find_hinge_like_region(sequence, ss_info, max_gap=2, max_end_non_C=3):
    """ To check for coiled/disordered regions that could behave as a hinge-like region. 
        This function is to automate the identification of a potential hinge like region. 
        Please always check the predicted hinge-like region and manually adjust the sequence 
        according to any available prior knowledge.

        sequence: The primary sequence of target receptor's extracellular region.
        ss_info: The secondary structure information of the target receptor.
        max_gap: Maximum allowed gap between two coil islands.
        max_end_non_C: Maximum allowed non coil region in the C terminal attached to the transmembrane region. 
    """
    
    if not np.asarray([ss in ['C','E','H']for ss in np.unique(ss_info)]).all():
        raise ValueError("The secondary structure string should only contain {`C`,`H`,`E`}")

    range_hinge = get_hinge_like_region_indices(ss_info, max_gap=max_gap,
                                                max_end_non_C=max_end_non_C)
    
    if range_hinge is None:
        return ""
    
    hinge_indices = np.arange(range_hinge[0], range_hinge[1] + 1)
    hinge_sequence = ''.join(sequence[i] for i in hinge_indices)
    return hinge_sequence