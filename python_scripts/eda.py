import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set(style="whitegrid")

data_path = "data_combined.csv"

data = pd.read_csv(data_path)

# Display basic information about the dataset
print("Dataset Info:\n")
data.info()

# Display the first 5 rows
print("\nFirst 5 Rows of the Dataset:\n")
print(data.head())

data = data.drop(["Name", "Name.1", "Name.2", "Name.3"],axis=1)
print(data.head())

# --------------------------------------------------------------------------
# Check for missing values
missing_values = data.isnull().sum().sum()
print(f"\nTotal Missing Values in Dataset: {missing_values}")

# Basic statistics for all columns
print("\nBasic Statistics:\n")
print(data.describe().transpose())

# Correlation analysis
# Compute the correlation matrix for a subset of columns (e.g., first 50 columns for better visualization)
corr_subset = data.corr()

# Visualize the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr_subset, cmap="coolwarm", annot=False, cbar=True)
plt.title("Correlation Matrix of the First 50 Columns")
plt.savefig("correlation.png")


def find_strings_in_dataframe(dataframe):
    string_cells = dataframe.applymap(lambda x: isinstance(x, str))
    if string_cells.any().any():
        print("\nString values detected in the dataset.")
        string_locations = np.where(string_cells)
        for row, col in zip(*string_locations):
            print(f"String found at Row: {row}, Column: {dataframe.columns[col]} | Value: {dataframe.iloc[row, col]}")
    else:
        print("\nNo string values detected in the dataset.")

find_strings_in_dataframe(data)

# Identify highly correlated column pairs (correlation > 0.9)
print(data)
data.to_csv("tbd.csv")
corr_matrix = data.corr()
# ------------------------
# Select the upper triangle of the correlation matrix
upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)

# Identify columns with high correlation
to_drop = [
    column
    for column in corr_matrix.columns
    if any(corr_matrix[column][upper_triangle] > threshold)
]

# Drop the columns
reduced_df = data.drop(columns=to_drop)
reduced_df.to_csv("combined_wo_corr.csv")
# ------------------------:

high_corr_pairs = []
threshold = 0.9
for i in range(corr_matrix.shape[0]):
    for j in range(i + 1, corr_matrix.shape[1]):
        if corr_matrix.iloc[i, j] > threshold:
            high_corr_pairs.append((corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

# Display highly correlated pairs
print("\nHighly Correlated Column Pairs (Correlation > 0.9):\n")
if high_corr_pairs:
    for pair in high_corr_pairs:
        print(f"Columns: {pair[0]} and {pair[1]} | Correlation: {pair[2]:.2f}", end ="\t")
else:
    print("No highly correlated column pairs found.")

# Summary
print("\nEDA Completed.")
