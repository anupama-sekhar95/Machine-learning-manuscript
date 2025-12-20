import numpy as np
import pandas as pd

from sklearn.metrics import jaccard_score
from sklearn.feature_selection import VarianceThreshold


def corr_sel(data):
    corr_matrix = data.corr(method='spearman') 
    # Create mask for highly correlated features
    high_corr_features = set()
    threshold = 0.9

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:  
                colname = corr_matrix.columns[i]
                high_corr_features.add(colname)

    print(f"Highly correlated features to remove: {len(high_corr_features)}")
    data = data.drop(high_corr_features,axis = 1)
    return data

def near_constant(data):
    selector = VarianceThreshold(threshold=0.01)  # Remove near-constant features, i.e. all low-variance features
    data_reduced = selector.fit_transform(data)
    sel_feature = selector.get_feature_names_out(input_features=data.columns)
    data_reduced = pd.DataFrame(data_reduced, columns=sel_feature)
    return data_reduced

def jaccard(df):
    # The Jaccard coefficient can be a value between 0 and 1, with 0 indicating no overlap and 1 complete overlap between the sets.
    #try required as in case of data with fingerprint, it can not check twol cols with different type:ValueError: Classification metrics can't handle a mix of binary and continuous targets
    def jac(col1, col2):
        try: 
            return jaccard_score(df[col1], df[col2])
        except:return 0
        
    jaccard_matrix = pd.DataFrame(
    [[jac(df[col1], df[col2]) for col2 in df.columns] for col1 in df.columns],
    index=df.columns,
    columns=df.columns
    )  
    
    # Set a threshold (e.g., 0.9) to remove highly similar features
    threshold = 0.9
    high_jaccard_features = set()

    for i in range(len(jaccard_matrix.columns)):
        for j in range(i):
            if jaccard_matrix.iloc[i, j] > threshold:
                high_jaccard_features.add(jaccard_matrix.columns[i])    
                
    df = df.drop(high_jaccard_features,axis=1)
    return df


if __name__ == "__main__":
    
    data = pd.read_csv('../data/data_to_cluster_edited.csv')
    names = list(data.Name)
    print(data.columns)
    data = data.drop(['Plant family', 'Plant genus', 'Species', 'Plant part_Methodology',
       'Plant order', 'plant name'], axis=1)
    # data = data.drop("Name", axis=1)
    
    data = jaccard(data)
    print(data.shape)

    data = corr_sel(data)
    print(data.shape)

    data = near_constant(data)
    print(data.shape)

    data["Name"] = names
    data.to_csv("../data/data_to_cluster_edited_reduced.csv",index = False)