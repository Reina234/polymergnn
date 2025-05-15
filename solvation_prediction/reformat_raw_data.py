import pandas as pd

input_file = "solvation_prediction/polymer_solvation_data_cleaned.csv"  # Update path
df = pd.read_csv(input_file, header=None)  

polymer_smiles = df.iloc[0, 2:].tolist()

solvent_smiles = df.iloc[1:, 0].tolist()  
densities = df.iloc[1:, 1].tolist() 

interaction_matrix = df.iloc[1:, 2:].reset_index(drop=True)

data_list = []
for solvent_idx, solvent_smiles_str in enumerate(solvent_smiles):
    for polymer_idx, polymer_smiles_str in enumerate(polymer_smiles):
        interaction_label = interaction_matrix.iloc[solvent_idx, polymer_idx]

        # Encode interaction label
        if interaction_label == "Y":
            encoded_label = 1
        elif interaction_label == "N":
            encoded_label = 0
        elif interaction_label == "INTERACTION":
            encoded_label = 2
        else:
            encoded_label = None  

        data_list.append(
            [
                solvent_smiles_str,  
                densities[solvent_idx],  
                polymer_smiles_str, 
                interaction_label,  
                encoded_label, 
            ]
        )


long_format_df = pd.DataFrame(
    data_list,
    columns=[
        "Solvent_SMILES",
        "Density",
        "Polymer_SMILES",
        "Interaction_Label",
        "Encoded_Label",
    ],
)


output_file = "solvent_polymer_interactions.csv"
long_format_df.to_csv(output_file, index=False)

print(f"Converted CSV saved as: {output_file}")
