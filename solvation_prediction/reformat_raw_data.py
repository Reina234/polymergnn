import pandas as pd

# Load the CSV file
input_file = "solvation_prediction/polymer_solvation_data_cleaned.csv"  # Update path
df = pd.read_csv(input_file, header=None)  # No headers

# Extract polymer (monomer) SMILES from Row 1 (index 0), Column 3 onwards
polymer_smiles = df.iloc[0, 2:].tolist()

# Extract solvent SMILES & density from Row 2 onward
solvent_smiles = df.iloc[1:, 0].tolist()  # Column 1 (index 0) = Solvent SMILES
densities = df.iloc[1:, 1].tolist()  # Column 2 (index 1) = Densities

# Extract interaction matrix (from Row 2 onward, Column 3 onward)
interaction_matrix = df.iloc[1:, 2:].reset_index(drop=True)

# Convert to long-format (tidy) dataset
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
            encoded_label = None  # Handle unexpected values

        data_list.append(
            [
                solvent_smiles_str,  # Solvent SMILES
                densities[solvent_idx],  # Solvent Density
                polymer_smiles_str,  # Polymer SMILES
                interaction_label,  # Original Interaction Label
                encoded_label,  # Encoded Interaction Label
            ]
        )

# Convert list to DataFrame
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

# Save to CSV
output_file = "solvent_polymer_interactions.csv"
long_format_df.to_csv(output_file, index=False)

print(f"Converted CSV saved as: {output_file}")
