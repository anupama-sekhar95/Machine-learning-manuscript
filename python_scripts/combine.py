import pandas as pd

fname = "../data/fp_MACCS.csv"
prefix = "MACCS"
data1 = pd.read_csv(fname)

fname = "../data/fp_Pubchem.csv"
prefix = "Pubchem"
data2 = pd.read_csv(fname)

fname = "../data/fp_EXT.csv"
prefix = "EXT"
data3 = pd.read_csv(fname)

# fname = "../data/fp_Sub_count.csv"
# prefix = "Sub_count"

fname = "../data/fp_Sub.csv"
prefix = "Sub"
data4 = pd.read_csv(fname)


print(data1.shape)
print(data2.shape)
print(data3.shape)
print(data4.shape)

data1 = data1.sort_values("Name").reset_index(drop=True)
print(data1.describe)

data2 = data2.sort_values("Name").reset_index(drop=True)
print(data2.describe)

data3 = data3.sort_values("Name").reset_index(drop=True)
print(data3.describe)

data4 = data4.sort_values("Name").reset_index(drop=True)
print(data4.describe)

# data = pd.concat([data1, data2, data3, data4],axis=1)
# print(data.describe)

# data = data.drop(["Name.1", "Name.2", "Name.3"],axis=1)
# print(data.head())
# data.to_csv("data_combined.csv", index=False)



# --------------

data = pd.concat([data1, data3],axis=1)
print(data.describe)

# data = data.drop(["Name.1"],axis=1)
print(data.head())
data.to_csv("data_combined_MP.csv", index=False)


# --------------

data = pd.concat([data1, data3],axis=1)
print(data.describe)

# data = data.drop(["Name.1"],axis=1)
print(data.head())
data.to_csv("data_combined_ME.csv", index=False)

# --------------

data = pd.concat([data1, data4],axis=1)
print(data.describe)

# data = data.drop(["Name.1"],axis=1)
print(data.head())
data.to_csv("data_combined_MS.csv", index=False)

# --------------

data = pd.concat([data2, data3],axis=1)
print(data.describe)

# data = data.drop(["Name.1"],axis=1)
print(data.head())
data.to_csv("data_combined_PE.csv", index=False)

# --------------

data = pd.concat([data2, data4],axis=1)
print(data.describe)

# data = data.drop(["Name.1"],axis=1)
print(data.head())
data.to_csv("data_combined_PS.csv", index=False)

# --------------

data = pd.concat([data3, data4],axis=1)
print(data.describe)

# data = data.drop(["Name.1"],axis=1)
print(data.head())
data.to_csv("data_combined_ES.csv", index=False)