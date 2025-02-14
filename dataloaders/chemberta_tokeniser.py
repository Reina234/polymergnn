import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


class ChemBERTaEmbedder:
    chemberta = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")

    def __init__(self):
        pass

    def embed(self, smiles: str):
        token = self.tokenizer(smiles, return_tensors="pt", truncation=True)
        output = self.chemberta(**token)
        embedding = torch.mean(output[0], 1)
        return embedding
