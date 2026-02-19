import numpy as np
from .receptor_utils import get_coords

class FilterStructures:
    """
    Applies geometry based filters to remove structures that doesn't obey the geometry of membrane.
    
    Note: This only for a receptor transmembrane helix.

    tol: Tolerance for structure not obeying the membrane geometry.
    lres: The last residue from which the normal is calculated.
    """

    def __init__(self, tol=0, lres=None):
        self.tol = tol
        self.lres = lres


    def get_val(self, p1, coeff):
        return (
            coeff[0] * p1[0]
            + coeff[1] * p1[1]
            + coeff[2] * p1[2]
            + coeff[3]
        )

    def get_plane(self, p1, v1):
        a, b, c = v1
        d = -(p1[0]*a + p1[1]*b + p1[2]*c)
        return [a, b, c, d]

    def check_structure(self, resd, s1, coeff):
        atoms = ["CA", "C", "N"]
        c = 0

        for res in resd:
            for i in atoms:
                p = np.array(res[i])
                val = self.get_val(p, coeff)
                s = np.sign(val)
                if s1 * s >= 0:
                    c += 1

        return 1 if c <= self.tol else 0

    def check_file(self, file):
        resd, _ = get_coords(file)

        bres = self.lres - 4
        p1 = np.array(resd[bres]["CA"])
        p2 = np.array(resd[self.lres]["CA"])

        v1 = p2 - p1
        pln_coeff = self.get_plane(p1, v1)

        val1 = self.get_val(p2, pln_coeff)
        s1 = np.sign(val1)

        return self.check_structure(resd[0:bres-2], s1, pln_coeff)
    
    def apply_filter(self, samplefiles):
        rfiles = []
        for file in samplefiles:
            if self.check_file(file) == 1:
                rfiles.append(file)
        
        return rfiles

class EndToEndCalculator(FilterStructures):
    """
    Calculates end-to-end distance along the normal to the membrane plane.
    This contains a regular end-to-end distance calculator between any two points.
    This class can also filter out the structures if there is an transmembrane helix 
    to account for the geometry of the membrane.

    top: PDB file to serve as the topology.
    start: A residue to which the distance is calculated.
    stop: A residue from which the distance is calculated.

    Following is for applying membrane filters. Check the parent class `FilterStructures`.
    tol: Tolerance for structure not obeying the membrane geometry.
    last_res: The last residue from which the normal is calculated.
    last_residue_offset: An offset to use different last residue for filtering purpose and for calculating z_e2e distance.
                         This is an hyperparameter and can be changed accordingly.
    """

    def __init__(self, top, tol=0, last_res=45, start=0, stop=45, last_residue_offset=6):
        
        _, resno = get_coords(top)
        self.n_res = len(resno)

        if last_res < self.n_res:
            super().__init__(tol=tol, lres=last_res-last_residue_offset)
        else:
            raise ValueError(f"The parameter `last_res` should be less than number of residues in `top` file ({self.n_res})")

        
        if start < self.n_res and stop < self.n_res:
            if start < stop: 
                self.start = start
                self.stop = stop
                self.last_res = last_res
            else:
                raise ValueError(f"`stop` residues should be greater than `start` residue")
        else:
            raise ValueError(f"The parameters should be less than number of residues in `top` file ({self.n_res})")

    def calcdist(self, file, b, e):
        resd, _ = get_coords(file)

        coord1 = np.array(resd[b]["CA"])
        coord2 = np.array(resd[e]["CA"])
        diffvect = coord1 - coord2

        return np.sqrt(np.sum(diffvect * diffvect))

    def calc_cosdist(self, file, b, e):
        resd, _ = get_coords(file)

        bres2 = self.last_res - 4
        p1 = np.array(resd[bres2]["O"])
        p2 = np.array(resd[self.last_res]["N"])

        v1 = p2 - p1
        norm_v1 = np.linalg.norm(v1)
        if norm_v1 == 0:
            raise ValueError(f"Cannot compute membrane normal: identical points at indices {bres2} and {self.last_res} in {file}")
        v1_cap = v1/norm_v1

        coord1 = np.array(resd[b]["CA"])
        coord2 = np.array(resd[e]["CA"])
        diffvect = coord1 - coord2

        cos_dist = np.abs(np.sum(diffvect * v1_cap))
        return cos_dist

    def get_e2e(self, samplefiles, apply_membrane_filter=True):
        """
        The function to calculate end-to-end distances. 
        This function can also apply the membrane filter to remove structures that
        doesn't obey the membrane geometry.


        Check the parent classes `EndToEndCalculator` and `FilterStructures` for more information

        samplefiles: A list of pdb files for which the end-to-end distance is calcualted.
        apply_membrane_filter: Turn on when membrane filter needs to be activated.
        """
        dist = []
        cos_dist = []
        from tqdm import tqdm

        if apply_membrane_filter:
            filtered_files = self.apply_filter(samplefiles)
            print(f"Percentage of structure after geometry filter to account for membrane(%): {len(filtered_files)*100/len(samplefiles)}")
        else:
            filtered_files = samplefiles

        for file in tqdm(filtered_files):
            
                dist.append(self.calcdist(file, self.start, self.stop))
                cos_dist.append(self.calc_cosdist(file, self.start, self.stop))

        return filtered_files, dist, cos_dist

class EndToEndCalculatorCOM(FilterStructures):
    """
    Calculates end-to-end distance between center of masses along the normal to the membrane plane.
    This contains a regular end-to-end distance calculator between any two center of masses.
    This class is made specifically for a receptor. Thus, a discontinuos range of epitope residues can also be provided.
    This class can also filter out the structures if there is an transmembrane helix 
    to account for the geometry of the membrane.

    top: PDB file to serve as the topology.
    start: (List of lists) A range of residues to which the distance is calculated. It can be discontinuos as well
    stop: (List of lists) A range of residues from which the distance is calculated.

    For example:
        start=[[1,10],[20,25],[30,32]]
        stop=[[200,202]]
        These are all accepted types. In this case COM of atoms from 1 to 10, 20 to 25, and 30 to 32 is calculated.
        Similarly, COM of 200 to 202 is also calculated. Then the distance is calculated between these 2 COMs

    Following variables is for applying membrane filters. Check the parent class `FilterStructures`.
    tol: Tolerance for structure not obeying the membrane geometry.
    last_res: The last residue from which the normal is calculated.
    last_residue_offset: An offset to use different last residue for filtering purpose and for calculating z_e2e distance.
                         This is an hyperparameter and can be changed accordingly.
    """

    def __init__(self, top, tol=0, last_res=None, start=None, stop=None, last_residue_offset=6):
        
        _, resno = get_coords(top)
        self.n_res = len(resno)

        if last_res < self.n_res:
            super().__init__(tol=tol, lres=last_res-last_residue_offset)
        else:
            raise ValueError(f"The parameter `last_res` should be less than number of residues in `top` file ({self.n_res})")
        wrong_var_type = True
        if isinstance(start, list) and all(isinstance(i, list) for i in start):
            if isinstance(stop, list) and all(isinstance(i, list) for i in stop):
                wrong_var_type=False
                if all(val < self.n_res for row in start for val in row):
                    if all(val < self.n_res for row in stop for val in row):
                        self.start = start
                        self.stop = stop
                        self.last_res =last_res
                    else:
                        raise ValueError(f"`stop` residues should be less than number of residues in `top` file ({self.n_res})")
                else:
                    raise ValueError(f"`start` residues should be less than number of residues in `top` file ({self.n_res})")

        if wrong_var_type:
            raise TypeError(f"`start` and `stop` variable should be list of lists with each inner list containing the range residues.\n For instance, if start=[[123,140],[240,250]]), COM is calculated for residue 123-140 and 240-250.")

    def calcCOM_dist(self,file):
        
        import mdtraj as md

        def get_collective_atoms(c_atoms):
            collective_atoms=[]
            for c_as in c_atoms:
                collective_atoms += top.select(f"resid {' '.join([str(i) for i in range(c_as[0],c_as[1]+1)])}").tolist()
            return np.array(collective_atoms)


        pdb = md.load(file)
        top = pdb.topology
        
        atoms1 = get_collective_atoms(self.start)
        atoms2 = get_collective_atoms(self.stop)
        g1 = pdb.atom_slice(atoms1)
        g2 = pdb.atom_slice(atoms2)
        
        com1 = md.compute_center_of_mass(g1)[0]
        com2 = md.compute_center_of_mass(g2)[0]
        
        diffvect = com1-com2
        
        return np.sqrt(np.sum(diffvect*diffvect))

    def calcCOM_cos_dist(self,file):

        import mdtraj as md

        def get_collective_atoms(c_atoms):
            collective_atoms=[]
            for c_as in c_atoms:
                collective_atoms += top.select(f"resid {' '.join([str(i) for i in range(c_as[0],c_as[1]+1)])}").tolist()
            return np.array(collective_atoms)


        pdb = md.load(file)
        top = pdb.topology
        resd, _ = get_coords(file)

        begin_res = self.last_res-4
        p1 = np.array(resd[begin_res]['O'])
        p2 = np.array(resd[self.last_res]['N'])
        v1 = p2-p1
        mag_v1 = np.sqrt(np.sum(v1*v1))
        v1_cap = v1/mag_v1
        
        atoms1 = get_collective_atoms(self.start)
        atoms2 = get_collective_atoms(self.stop)
        g1 = pdb.atom_slice(atoms1)
        g2 = pdb.atom_slice(atoms2)
        
        com1 = md.compute_center_of_mass(g1)[0]
        com2 = md.compute_center_of_mass(g2)[0]
        
        diffvect = com1-com2
        cos_dist = np.abs(np.sum(diffvect*v1_cap))
        
        return cos_dist 

    def get_e2e(self, samplefiles, apply_membrane_filter=True):
        """
        The function to calculate end-to-end distances. 
        This function can also apply the membrane filter to remove structures that
        doesn't obey the membrane geometry. 

        Check the parent classes `EndToEndCalculatorCOM` and `FilterStructures` for more information

        samplefiles: A list of pdb files for which the end-to-end distance is calcualted.
        apply_membrane_filter: Turn on when membrane filter needs to be activated.
        """
        dist = []
        cos_dist = []
        from tqdm import tqdm

        if apply_membrane_filter:
            filtered_files = self.apply_filter(samplefiles)
            print(f"Percentage of structure after geometry filter to account for membrane(%): {len(filtered_files)*100/len(samplefiles)}")
        else:
            filtered_files = samplefiles

        for file in tqdm(filtered_files):
            
                dist.append(self.calcCOM_dist(file))
                cos_dist.append(self.calcCOM_cos_dist(file))

        return filtered_files, dist, cos_dist