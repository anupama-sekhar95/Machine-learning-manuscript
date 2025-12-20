import pandas as pd 
import os   
import seaborn as sns
import shap
import matplotlib.pyplot as plt

import pickle

def a():
    path = 'processed_Sub-count.csv'
    processed_data = pd.read_csv(path)

    TARGETS = {
        "family": "tfamily",
        "genus": "tgenus",
        "order": "torder",
    }
    for target_key, target_col in TARGETS.items():
        X = processed_data.drop(columns=list(TARGETS.values()))
        y = processed_data[target_col]

        path_best_model = "best_estimator_overall_grid_model_Sub_tgenus_2_Sub.sav"
        
        with open(path_best_model, 'rb') as f:
            best_estimator = pickle.load(f)
            print( best_estimator)
        
        #tsting Shap
        best_estimator.fit(X,y)
        
        try:
            def f(x):
                return best_estimator.predict_proba(x)[:, 1]


            med = X.median().values.reshape((1, X.shape[1]))

            explainer = shap.Explainer(f, med)
            shap_values = explainer(X.iloc[0:200, :])
            shap.plots.beeswarm(shap_values)
            # shap.plots.beeswarm(shap_values, show=False)
            # plt.savefig("11.svg",dpi=1000,figsize=(10, 5)) #.png,.pdf will also support here
            shap.plots.heatmap(shap_values)
            # shap.plots.heatmap(shap_values, show=False)
            # plt.savefig("22.svg",dpi=1000, figsize=(10, 5))
        except Exception as e:print(0,e)
        
        
def a2():

    fp = "nofp"
    # fp = "EXT"
    # fp = "Sub"
    
    # target_col = 'tfamily'
    # target_col = 'tgenus'
    target_col = 'torder'
    
    # runnum = 1
    # runnum = 2
    runnum = 3
    
    processed_data = pd.read_csv(f'../data/processed_{fp}.csv')

    TARGETS = {
        "family": "tfamily",
        "genus": "tgenus",
        "order": "torder",
    }
    

    X = processed_data.drop(columns=list(TARGETS.values()))
    y = processed_data[target_col]

    path_best_model = f'../output/best_estimator_overall_grid_model_{fp}_{target_col}_{runnum}_{fp}.sav'
    with open(path_best_model, 'rb') as f:
        best_estimator = pickle.load(f)
        print(target_col, best_estimator)

    
    best_estimator.fit(X,y)
    try:
        def f(x):
            return best_estimator.predict_proba(x)[:, 1]


        med = X.median().values.reshape((1, X.shape[1]))

        explainer = shap.Explainer(f, med)
        shap_values = explainer(X.iloc[0:200, :])
        # shap.plots.beeswarm(shap_values, plot_size=(10,5), max_display=12)
        # shap.plots.waterfall(shap_values[0])
        # shap.plots.beeswarm(shap_values, show=False)
        # plt.savefig("11.svg",dpi=1000,figsize=(10, 5)) #.png,.pdf will also support here
        shap.plots.heatmap(shap_values,max_display=20, feature_values=shap_values.abs.max(0), instance_order=shap_values.sum(1))
        # shap.plots.heatmap(shap_values, show=False)
        # plt.savefig("22.svg",dpi=1000, figsize=(10, 5))
    except Exception as e:print(0,e)

        

def tbd():
    folder='output_cls_fp_correct_gscv/'
    files = os.listdir(folder)
    files = list(filter(lambda x: ".sav" in x, files))
    print(files)
    for file in files:
        path_best_model = folder+file
        with open(path_best_model, 'rb') as f:
            best_estimator = pickle.load(f)
            print(file, best_estimator)
            