from abc import ABC, abstractmethod
from rdkit import Chem  # type: ignore


# pylint: disable=no-member
class SmilesTransformer(ABC):

    @abstractmethod
    def transform(self, smiles: str) -> str:
        pass


class NoSmilesTransform(SmilesTransformer):

    def transform(self, smiles: str) -> str:
        return smiles


class CanonicalSmilesTransform(SmilesTransformer):

    def transform(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return Chem.MolToSmiles(mol, canonical=True)


class PolymerisationSmilesTransform(SmilesTransformer):

    def _identify_polymer_bond(self, mol):
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE and not bond.IsInRing():
                atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()
                if atom1.GetAtomicNum() == 6 and atom2.GetAtomicNum() == 6:
                    return bond

        raise ValueError("No suitable polymerizable double bond found in the molecule.")

    def transform(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        rw_mol = Chem.RWMol(mol)

        bond = self._identify_polymer_bond(mol=mol)
        atom1, atom2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

        dummy1, dummy2 = Chem.Atom(0), Chem.Atom(0)
        dummy1_idx, dummy2_idx = rw_mol.AddAtom(dummy1), rw_mol.AddAtom(dummy2)

        rw_mol.RemoveBond(atom1, atom2)
        rw_mol.AddBond(atom1, atom2, Chem.rdchem.BondType.SINGLE)
        rw_mol.AddBond(atom1, dummy1_idx, Chem.BondType.SINGLE)
        rw_mol.AddBond(atom2, dummy2_idx, Chem.BondType.SINGLE)

        return Chem.MolToSmiles(rw_mol, canonical=True)


# add in conversion to mol?
