from tools.mol_to_molgraph import FGMembershipMol2MolGraph
from rdkit import Chem

smiles = "CC"
mol = Chem.MolFromSmiles(smiles)
fg = FGMembershipMol2MolGraph()

print(fg.convert(mol))
