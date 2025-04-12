from featurisers.ertl_algorithm import ErtlAlgorithm
from rdkit import Chem
import numpy as np
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.rdchem import Mol


class FGFingerprintGenerator:
    def __init__(self, fg_n_bits: int = 64, radius: int = 2):
        self.n_bits = fg_n_bits
        self.radius = radius
        self.fg_detector = ErtlAlgorithm()

    @staticmethod
    def smiles_to_fingerprint(mol: Mol, n_bits, radius=2):
        morgan_generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
        if mol is None:
            fingerprint = np.zeros(n_bits)  # Zero-vector for invalid SMILES
        else:
            fingerprint = morgan_generator.GetCountFingerprintAsNumPy(mol)
        return fingerprint

    def _aggregate_fg_fps(self, ifg_list):
        combined_fp = np.zeros(self.n_bits, dtype=int)

        for ifg in ifg_list:
            fg_mol = Chem.MolFromSmiles(ifg.type)
            fg_fp = self.smiles_to_fingerprint(
                fg_mol, n_bits=self.n_bits, radius=self.radius
            )
            combined_fp = np.maximum(combined_fp, fg_fp)
        return combined_fp

    def create_fg_encoding(self, mol: Mol):
        ifg_list = self.fg_detector.detect(mol)
        combined_fp = self._aggregate_fg_fps(ifg_list=ifg_list)
        return combined_fp


test = FGFingerprintGenerator()
print(test.create_fg_encoding(mol))
