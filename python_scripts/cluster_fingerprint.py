# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sns

from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

import matplotlib.gridspec as gridspec



# fname = "../data/fp_MACCS.csv"
# prefix = "MACCS"

# fname = "../data/fp_Pubchem.csv"
# prefix = "Pubchem"

# fname = "../data/fp_EXT.csv"
# prefix = "EXT"

# fname = "../data/fp_Sub_count.csv"
# prefix = "Sub_count"

# fname = "../data/fp_Sub.csv"
# prefix = "Sub"

# fname = "data_combined_ES.csv"
# prefix = "combined_ES"

# fname = "data_combined_ME.csv"
# prefix = "combined_ME"

# fname = "data_combined_MP.csv"
# prefix = "combined_MP"

# fname = "data_combined_MS.csv"
# prefix = "combined_MS"

# fname = "data_combined_PE.csv"
# prefix = "combined_PE"

# fname = "data_combined_PS.csv"
# prefix = "combined_PS"

# fname = "data_combined.csv"
# prefix = "combined"


# fnamel = [ "../data/fp_MACCS.csv", "../data/fp_Pubchem.csv", "../data/fp_EXT.csv", "../data/fp_Sub_count.csv", "../data/fp_Sub.csv", "data_combined_ES.csv",\
#     "data_combined_ME.csv",  "data_combined_MP.csv","data_combined_MS.csv", "data_combined_PE.csv", "data_combined_PS.csv", "data_combined.csv"]
# prefixl = ["MACCS", "Pubchem","EXT", "Sub_count", "Sub", "combined_ES", "combined_ME", "combined_MP",  "combined_MS", "combined_PE",  "combined_PS", "combined" ]

fnamel, prefixl = [],[]
for fname,prefix in zip(fnamel, prefixl):
    print(fname, prefix )
    data = pd.read_csv(fname)

    cols = data.columns.tolist()

    print("Name.1" in cols, "Name.2" in cols, "Name.3" in cols)
    try:
        data = data.drop(["Name.1", "Name.2", "Name.3"],axis=1)
    except:pass
    try:  
        data = data.drop(["Name.1"],axis=1)
    except:pass
    # X = data.drop(columns = "Name", axis=1).values

    data_other = pd.read_csv("/home/arpanaalka/Compounds_Data_Analysis/data/Cluster_results_newlabels.csv")
    merged_df = pd.merge(
    data,
    data_other[['Name', 'Biosynthetic_class', 'Structural_class']],
    on='Name',
    how='left'   # ensures all names from data1 are kept
    )   

    biosyn_class = merged_df["Biosynthetic_class"]
    struct_class = merged_df["Structural_class"]
    X = merged_df.drop(columns = ["Name", "Biosynthetic_class", "Structural_class"], axis=1).values

    scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    X_scaled = X
    
    

    def clustering():
        f = open(f"outputs/t-SNE/cluster_performance_{prefix}.txt","a+")
        f.write(f"fp_name, n_clusters_number ; sil_K sil_agg sil_dbscan sil_spectral \t dav_k dav_agg dav_dbscan dav_spectral \t  cal_k cal_agg cal_dbscan cal_spectral\n")
        for cn in [2,4,6,8,9,10]: 

            # KMeans Clustering
            kmeans = KMeans(n_clusters=cn)
            kmeans_labels = kmeans.fit_predict(X_scaled)

            # Agglomerative Clustering
            agg_clustering = AgglomerativeClustering(n_clusters=cn)
            agg_labels = agg_clustering.fit_predict(X_scaled)

            # DBSCAN Clustering
            dbscan = DBSCAN(eps=10, min_samples=cn)  # we set min sample as different values
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
            

            f.write(f"\n{prefix}, n_clusters:{cn}; {s:.02} {sa:.02} {sd:.02} {ssc:.02} \t {d:.02} {da:.02} {dd:.02} {dsc:.02} \t  {c:.02} {ca:.02} {cd:.02} {csc:.02}")


            # Apply t-SNE (2D)
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
            X_tsne = tsne.fit_transform(X_scaled)

            # Define the labels and titles for each subplot
            labels_list = [kmeans_labels, agg_labels, dbscan_labels, sc_labels]
            titles_list = ["KMeans", "Agglomerative", "DBSCAN", "Spectral Clustering"]

            data_plot = pd.DataFrame(X_tsne,columns=["x","y"])
            data_plot["Biosys class"] = biosyn_class
            data_plot["Struct class"] = struct_class
            

            fig, axes = plt.subplots(1, len(labels_list), figsize=(6*len(labels_list)+4, 6))

            if len(labels_list) == 1:  # if only one subplot
                axes = [axes]

            handles, labels = None, None  # placeholders

            for i, ax in enumerate(axes):
                data_plot["Cluster_ID"] = labels_list[i]
                scatter = sns.scatterplot(
                    data=data_plot,
                    x="x", y="y",
                    hue="Cluster_ID",       # cluster → color
                    style="Biosys class",     # major class → marker
                    # size="Compound class",   # compound class → size
                    sizes=(50,70),         # size range
                    palette="tab10",
                    alpha=0.7,
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
                    bbox_to_anchor=(1.02, 0.5),
                    title="Legend")

            # Adjust layout to leave space on the right for legend
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            plt.show()
            plt.savefig(f"outputs/t-SNE/Biosys_{prefix}_{cn}.png") 
            
            fig, axes = plt.subplots(1, len(labels_list), figsize=(6*len(labels_list)+4, 6))
            
            for i, ax in enumerate(axes):
                data_plot["Cluster_ID"] = labels_list[i]
                scatter = sns.scatterplot(
                    data=data_plot,
                    x="x", y="y",
                    hue="Cluster_ID",       # cluster → color
                    # style="Major class",     # major class → marker
                    style="Struct class",   # compound class → size
                    sizes=(50,70),         # size range
                    palette="tab10",
                    alpha=0.7,
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
                    bbox_to_anchor=(1.02, 0.5),
                    title="Legend")
 
            # Adjust layout to leave space on the right for legend
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            plt.show()
            plt.savefig(f"outputs/t-SNE/Struct_{prefix}_{cn}.png") 
            
        f.close()
        data.to_csv(f"outputs/t-SNE/clustered_{prefix}.csv")


    clustering()


def plot_with_other_labels():
    data_other = pd.read_csv("../data/Cluster_results_newlabels.csv")
    merged_df = pd.merge(
    data,
    data_other[['Name', 'Biosynthetic_class', 'Structural_class']],
    on='Name',
    how='left'   # ensures all names from data1 are kept
    )   
    print(data.shape, data_other.shape, merged_df.shape)
    
    clsss_id_csv = pd.read_csv(f"outputs/clustered_{prefix}.csv")
    for cn in [3,5,7]:
        kmeans_labels, agg_labels, dbscan_labels, sc_labels = clsss_id_csv[f"{cn}_kmeans"],clsss_id_csv[f"{cn}_Agglomerative"], \
            clsss_id_csv[f"{cn}_dbscan"],clsss_id_csv[f"{cn}_spectral"]
        # Apply t-SNE (2D)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        X_tsne = tsne.fit_transform(X_scaled)


        # Define the labels and titles for each subplot
        labels_list = [kmeans_labels, agg_labels, dbscan_labels, sc_labels]
        titles_list = ["KMeans", "Agglomerative", "DBSCAN", "Spectral Clustering"]

        data_plot = pd.DataFrame(X_tsne,columns=["x","y"])
        data_plot["biosyn class"] = biosyn_class
        data_plot["struct class"] = struct_class
        
        fig, axes = plt.subplots(1, len(labels_list), figsize=(6*len(labels_list)+4, 6))
        if len(labels_list) == 1:  # if only one subplot
            axes = [axes]

        handles, labels = None, None  # placeholders

        for i, ax in enumerate(axes):
            data_plot["Cluster_ID"] = labels_list[i]

            scatter = sns.scatterplot(
                data=data_plot,
                x="x", y="y",
                hue="Cluster_ID",       # cluster → color
                style="struct class",
                sizes=(50,70),         # size range
                palette="tab10",
                alpha=0.7,
                legend="full" if i == 0 else False,
                ax=ax
            )

            ax.set_title(f"{titles_list[i]} t-SNE")
            ax.grid(True)

            # Capture handles & labels from the first subplot
            if i == 0:
                handles, labels = ax.get_legend_handles_labels()
                ax.get_legend().remove()   

        # Now add ONE combined legend for the figure
        fig.legend(handles, labels,
                loc="right",
                bbox_to_anchor=(1.02, 0.5),
                title="Legend")

        # Adjust layout to leave space on the right for legend
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()
        
def plot_single_labels():
    data_other = pd.read_csv("../data/Cluster_results_newlabels.csv")
    data = pd.read_csv(f"../outputs/clustered_{prefix}.csv")

    merged_df = pd.merge(
    data,
    data_other[['Name', 'Biosynthetic_class', 'Structural_class']],
    on='Name',
    how='left'   # ensures all names from data1 are kept
    )   
    print(data.shape, data_other.shape, merged_df.shape)
    print(merged_df.head())
    
    cluster_label = merged_df["3_kmeans"]
    X_scaled = merged_df.drop(['Unnamed: 0', 'Name','2_kmeans', '2_Agglomerative', '2_dbscan', '2_spectral', '4_kmeans', '4_Agglomerative', '4_dbscan', '4_spectral', '6_kmeans', '6_Agglomerative', '6_dbscan', '6_spectral', '8_kmeans', '8_Agglomerative', '8_dbscan', '8_spectral', '9_kmeans', '9_Agglomerative', '9_dbscan', '9_spectral', '10_kmeans', '10_Agglomerative', '10_dbscan', '10_spectral', '3_kmeans', 'Biosynthetic_class', 'Structural_class'],axis=1)

    # Apply t-SNE (2D)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)

    data_plot = pd.DataFrame(X_tsne,columns=["x","y"])
    data_plot["biosyn class"] = merged_df['Biosynthetic_class']
    data_plot["struct class"] = merged_df['Structural_class']
    data_plot['Cluster_ID'] = merged_df['3_kmeans']    
    
    scatter = sns.scatterplot(
        data=data_plot,
        x="x", y="y",
        hue="Cluster_ID",       # cluster → color
        style="biosyn class",     # major class → marker
        sizes=(50,70),         # size range
        palette="tab10",
        alpha=0.8,
        legend="full"
    )

    scatter.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Species', fontsize='small')
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()


plot_with_other_labels()
plot_single_labels()