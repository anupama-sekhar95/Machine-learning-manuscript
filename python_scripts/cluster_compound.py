# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.manifold import TSNE

import seaborn as sns

data = pd.read_csv("data_to_cluster_edited.csv")
print(data.head())
print(data.shape)
Plant_order = data["Plant order"].values
Plant_family = data["Plant family"].values
Plant_genus = data["Plant genus"].values
print(data.nunique())


X = data.drop(["Plant order", "Plant family","Plant genus", "Plant part_Methodology","Species","plant name" ], axis=1).values
scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
X_scaled = X

# palette_38 = sns.color_palette("hsv", 38)   # 38 unique hues
palette_38 = sns.color_palette("pastel")+sns.color_palette("bright")+sns.color_palette("dark")+sns.color_palette("deep")

f = open(f"output1/cluster_performance.txt","a+")
f.write(f"fp_name, n_clusters_number ; sil_K sil_agg sil_dbscan sil_spectral \t dav_k dav_agg dav_dbscan dav_spectral \t  cal_k cal_agg cal_dbscan cal_spectral\n")
for cn in range(2,30):
# if True: #When I don't want to run the code
    # KMeans Clustering
    kmeans = KMeans(n_clusters=cn)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    # Agglomerative Clustering
    agg_clustering = AgglomerativeClustering(n_clusters=cn)
    agg_labels = agg_clustering.fit_predict(X_scaled)

    # DBSCAN Clustering
    # dbscan = DBSCAN(eps=10, min_samples=cn)  # we set min sample as different values 
    dbscan = DBSCAN(eps=0.5+float("."+str(cn))) # for folder output1 only
    dbscan_labels = dbscan.fit_predict(X_scaled)

    try:
        clustering = SpectralClustering(n_clusters=cn,
                assign_labels='discretize',
                random_state=0).fit(X_scaled)
        sc_labels = clustering.labels_
    except:
        sc_labels = [0]*(len(dbscan_labels)//2)
        sc_labels = sc_labels + [1]*(len(dbscan_labels) - (len(dbscan_labels)//2))

    data[f"{cn}_kmeans"] = kmeans_labels
    data[f"{cn}_Agglomerative"] = agg_labels
    data[f"{cn}_dbscan"] = dbscan_labels
    data[f"{cn}_spectral"] = sc_labels


    s = silhouette_score(X_scaled, kmeans_labels)
    d = davies_bouldin_score(X_scaled, kmeans_labels)
    c = calinski_harabasz_score(X_scaled, kmeans_labels)
    
    sa = silhouette_score(X_scaled, agg_labels)
    da = davies_bouldin_score(X_scaled, agg_labels)
    ca = calinski_harabasz_score(X_scaled, agg_labels)
    
    try:
        sd = silhouette_score(X_scaled, dbscan_labels)
    except:sd= 0.0
    try:
        dd = davies_bouldin_score(X_scaled, dbscan_labels)
    except:dd=0.0
    try:
        cd = calinski_harabasz_score(X_scaled, dbscan_labels)
    except:cd=0.0
    
    ssc = silhouette_score(X_scaled, sc_labels)
    dsc = davies_bouldin_score(X_scaled, sc_labels)
    csc = calinski_harabasz_score(X_scaled, sc_labels)
    

    f.write(f"\n n_clusters:{cn}; {s:.02} {sa:.02} {sd:.02} {ssc:.02} \t {d:.02} {da:.02} {dd:.02} {dsc:.02} \t  {c:.02} {ca:.02} {cd:.02} {csc:.02}")


    # Apply t-SNE (2D)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)

    # Define the labels and titles for each subplot
    labels_list = [kmeans_labels, agg_labels, dbscan_labels, sc_labels]
    titles_list = ["KMeans", "Agglomerative", "DBSCAN", "Spectral Clustering"]

    data_plot = pd.DataFrame(X_tsne,columns=["x","y"])
    data_plot["Plant order"] = data["Plant order"]
    print(data_plot,"\t :data plot")

    fig, axes = plt.subplots(1, len(labels_list), figsize=(6*len(labels_list)+4, 6))

    if len(labels_list) == 1:  # if only one subplot
        axes = [axes]

    handles, labels = None, None  # placeholders

    for i, ax in enumerate(axes):
        data_plot["Cluster_ID"] = labels_list[i]
        scatter = sns.scatterplot(
            data=data_plot,
            x="x", y="y",
            style="Cluster_ID",       # cluster → color
            hue="Plant order",     # major class → marker
            # size="Compound class",   # compound class → size
            sizes=(50,70),         # size range
            palette=palette_38,
            alpha=0.7,
            s=20,
            legend="full" if i == 0 else False,
            ax=ax
        )

        ax.set_title(f"{titles_list[i]} t-SNE")
        ax.grid(True)

        # Capture handles & labels from the first subplot
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()   #  remove legend from subplot

    # Now add ONE combined legend for the figure
    fig.legend(handles, labels,
            loc="right",
            bbox_to_anchor=(1, 0.5),
            ncol=2,
            title="Legend")

    # Adjust layout to leave space on the right for legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    # plt.show()
    plt.savefig(f"output1/{cn}.png") 
data.to_csv(f"output1/clustered_data.csv")


