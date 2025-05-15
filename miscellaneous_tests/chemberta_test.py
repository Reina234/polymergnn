import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

chemberta = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")

test = tokenizer("CC", return_tensors="pt", truncation=True)
output = chemberta(**test)
print(output)
embeddings = output[0][:, 0, :]
print(embeddings)
embedding = torch.mean(output[0], 1)
print(embedding)
print(embedding.shape)

test = tokenizer("CCC", return_tensors="pt", truncation=True)
output = chemberta(**test)
print(output)
embeddings = output[0][:, 0, :]
print(embeddings)
embedding = torch.mean(output[0], 1)
print(embedding)
print(embedding.shape)
