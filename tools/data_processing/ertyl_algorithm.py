from collections import namedtuple
from rdkit import Chem  # type: ignore

# pylint: disable=no-member
IFG = namedtuple("IFG", ["atomIds", "bonds", "envIds", "atoms", "type"])


class ErtlAlgorithm:
    patterns = [
        Chem.MolFromSmarts("A=,#[!#6]"),  # Double/triple bond to heteroatoms
        Chem.MolFromSmarts("C=,#C"),  # Non-aromatic C=C or C#C
        Chem.MolFromSmarts("[CX4](-[O,N,S])-[O,N,S]"),  # Acetal carbons
        Chem.MolFromSmarts("[O,N,S]1CC1"),  # Oxirane, aziridine, thiirane
    ]
    # Added "bonds" field to store bond indices.

    def __init__(self):
        pass

    def _mark_atoms(self, mol: Chem.Mol):
        """Marks functional groups while ignoring polymerization site dummy atoms."""
        marked_atoms = set()
        for atom in mol.GetAtoms():
            if atom.HasProp("polymer_site"):
                continue
            # We still detect FGs based on non-carbon, non-hydrogen atoms.
            if atom.GetAtomicNum() not in (6, 1):
                marked_atoms.add(atom.GetIdx())

        for pattern in self.patterns:
            if pattern is None:
                continue
            for match in mol.GetSubstructMatches(pattern):
                for idx in match:
                    if not mol.GetAtomWithIdx(idx).HasProp("polymer_site"):
                        marked_atoms.add(idx)
        return marked_atoms

    def _merge_groups(self, mol: Chem.Mol, marked, aset):

        bset = set()
        for idx in aset:
            atom = mol.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                jdx = neighbor.GetIdx()
                if jdx in marked:
                    marked.remove(jdx)
                    bset.add(jdx)
        if not bset:
            return
        self._merge_groups(mol=mol, marked=marked, aset=bset)
        aset.update(bset)

    def detect(self, mol: Chem.Mol):
        """Finds and labels functional groups in the molecule."""

        marked_atoms = self._mark_atoms(mol=mol)
        groups = []
        functional_groups = []
        # Merge adjacent marked atoms.
        while marked_atoms:
            grp = {marked_atoms.pop()}
            self._merge_groups(mol=mol, marked=marked_atoms, aset=grp)
            groups.append(grp)

        for g in groups:
            # Expand FG to include adjacent carbons and hydrogens.
            extra = set()
            for idx in g:
                for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                    if neighbor.HasProp("polymer_site"):
                        continue
                    # Include both carbons and hydrogens.
                    if neighbor.GetAtomicNum() in (6, 1):
                        extra.add(neighbor.GetIdx())
            full_group = g.union(extra)

            environment = extra - g

            # Get bonds where both endpoints are in the FG.
            bonds_in_group = set()
            for bond in mol.GetBonds():
                begin = bond.GetBeginAtomIdx()
                end = bond.GetEndAtomIdx()
                if begin in full_group and end in full_group:
                    bonds_in_group.add(bond.GetIdx())

            functional_groups.append(
                IFG(
                    atomIds=tuple(full_group),
                    bonds=tuple(bonds_in_group),
                    # The 'atoms' field shows the original marked atoms.
                    envIds=tuple(environment),
                    atoms=Chem.MolFragmentToSmiles(
                        mol, g, canonical=True, allHsExplicit=True
                    ),
                    # The 'type' field shows the FG plus the adjacent C/H atoms.
                    type=Chem.MolFragmentToSmiles(
                        mol, full_group, canonical=True, allHsExplicit=True
                    ),
                )
            )
        return functional_groups
