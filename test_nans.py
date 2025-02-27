import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv("tests/output_2_4.csv")

# Assuming X is your input data
print(df.notna().any())


def check_negative_values(df):
    """
    Check for negative values in numerical columns of a dataframe.

    Parameters:
    df (pd.DataFrame): The dataframe to check.
    """
    # Select only numeric columns (int and float)
    numeric_df = df.select_dtypes(include=["number"])

    # Find columns containing negative values
    negative_counts = (numeric_df < 0).sum()

    # Filter columns that have at least one negative value
    negative_columns = negative_counts[negative_counts > 0]

    if negative_columns.empty:
        print("No negative values found in numerical columns.")
        return

    print("Negative values detected in the following numerical columns:\n")
    for col, count in negative_columns.items():
        print(f"- Column '{col}': {count} negative values")

    print("\nDisplaying first few rows with negative values:")
    negative_rows = numeric_df[numeric_df.lt(0).any(axis=1)]
    print(negative_rows.head())  # Show first few rows with negative values


check_negative_values(df)
