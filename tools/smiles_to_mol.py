from rdkit import Chem
from tools.smiles_transformers import SmilesTransformer, NoSmilesTransform


class Smiles2Mol:
    """
    Handles the processing of molecules using a specified preprocessor.
    """

    def __init__(self, smiles_transformer: SmilesTransformer = NoSmilesTransform()):
        self.transformer = smiles_transformer

    def convert(self, smiles: str) -> str:
        """Converts processed smiles to  Chem.mol"""
        processed_smiles = self.transformer.transform(smiles)
        mol = Chem.MolFromSmiles(processed_smiles.strip())
        if mol is not None:
            mol = Chem.AddHs(mol)
        return mol
