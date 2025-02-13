from abc import ABC, abstractmethod
from rdkit import Chem  # type: ignore


# pylint: disable=no-member
class Preprocessor(ABC):

    @abstractmethod
    def process_smiles(self, smiles: str) -> str:
        pass


class NoPreprocessing(Preprocessor):

    def process_smiles(self, smiles: str) -> str:
        return smiles


class CanonicalPreprocessor(Preprocessor):

    def process_smiles(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return Chem.MolToSmiles(mol, canonical=True)


class PolymerizationPreprocessor(Preprocessor):

    def _identify_polymer_bond(self):
        for bond in self.mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE and not bond.IsInRing():
                atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()
                if atom1.GetAtomicNum() == 6 and atom2.GetAtomicNum() == 6:
                    return bond
        raise ValueError("No suitable polymerizable double bond found in the molecule.")

    def process_smiles(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        rw_mol = Chem.RWMol(mol)

        bond = self._identify_polymer_bond()
        atom1, atom2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

        dummy1, dummy2 = Chem.Atom(0), Chem.Atom(0)
        dummy1_idx, dummy2_idx = rw_mol.AddAtom(dummy1), rw_mol.AddAtom(dummy2)

        rw_mol.RemoveBond(atom1, atom2)
        rw_mol.AddBond(atom1, atom2, Chem.rdchem.BondType.SINGLE)
        rw_mol.AddBond(atom1, dummy1_idx, Chem.BondType.SINGLE)
        rw_mol.AddBond(atom2, dummy2_idx, Chem.BondType.SINGLE)

        return Chem.MolToSmiles(rw_mol, canonical=True)


class MoleculeProcessor:
    """
    Handles the processing of molecules using a specified preprocessor.
    """

    def __init__(self, preprocessor: Preprocessor = NoPreprocessing()):
        self.preprocessor = preprocessor

    def process(self, smiles: str) -> str:
        """Converts processed smiles to  Chem.mol"""
        processed_smiles = self.preprocessor().process_smiles(smiles)
        mol = Chem.MolFromSmiles(processed_smiles.strip())
        if mol is not None:
            mol = Chem.AddHs(mol)
        return mol


# add in conversion to mol?
