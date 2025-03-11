import pandas as pd


def add_solvent_density(output_csv, solvent_csv, output_new_csv):
    # Load datasets
    output_df = pd.read_csv(output_csv)
    solvent_df = pd.read_csv(solvent_csv)

    # Extract necessary columns from solvent dataset
    solvent_dict = dict(
        zip(solvent_df["name"], solvent_df["density"] / 1000)
    )  # Convert density

    # Extract first column (solvent names) from output dataset
    output_df["Density (g/cm³)"] = output_df.iloc[:, 0].map(solvent_dict)

    # Reorder columns to place density at the beginning
    output_df = output_df[["Density (g/cm³)"] + list(output_df.columns[:-1])]

    # Save modified output dataset
    output_df.to_csv(output_new_csv, index=False)

    print(f"Updated CSV saved to: {output_new_csv}")


# Example Usage
output_csv_path = "data/output_3_11.csv"
solvent_csv_path = "data/solvent_data.csv"
output_new_csv_path = "data/output_3_11_with_density.csv"

add_solvent_density(output_csv_path, solvent_csv_path, output_new_csv_path)
