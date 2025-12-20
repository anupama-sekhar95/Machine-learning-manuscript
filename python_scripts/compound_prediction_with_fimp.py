import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pickle
import os

import seaborn as sns
import shap
import matplotlib.pyplot as plt


# Scikit-learn imports
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance # New import

# Classifier imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Assuming these are custom functions from your project
from fingureprint_selection import jaccard, corr_sel, near_constant


prefixl = ["MACCS", "Pubchem","EXT", "Sub-count", "Sub","nofp"]
prefix = prefixl[0]

OUTPUT_DIR = Path("output_cls_fp_correct_gscv_woseed")

if prefix=="nofp":
    INPUT_CSV = "data_to_cluster_edited.csv"
else:
    INPUT_CSV = f"output_cls_fp/data_with_fpmean_{prefix}.csv"

logging.basicConfig(
    filename=f"classification_with_datasaving_woseed_{prefix}.log",
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# RUN_NUM = f"5_{prefix}"
# print("RUN_NUM", RUN_NUM)
# logging.info(f"RUN_NUM: {RUN_NUM}")

# RUN_NUM = 5
TARGETS = {
    "family": "tfamily",
    "genus": "tgenus",
    "order": "torder",
}
COLUMNS_TO_DROP = [
    "Plant order", "Plant family", "Plant genus", "Species", "plant name",
    "Plant part_Methodology"
]


# Define classifiers and their hyperparameter grids for GridSearchCV
MODELS_FOR_GRIDSEARCH = {
    'RandomForest': {
        'estimator': RandomForestClassifier(),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 'log2', 0.5],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'MLP': {
        'estimator': MLPClassifier(max_iter=1000 ),
        'params': {
            'hidden_layer_sizes': [(50,),(50,50), (100,), (100, 50)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.001],
            'learning_rate': ['constant', 'adaptive']
        }
    },
    'LogisticRegression': {
        'estimator': LogisticRegression(),
        'params': {
            'C': [0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [200]
        }
    },
    'SVC': {
    'estimator': SVC(probability=True ),
    'params': {
        'C': [0.1, 1, 10, 100],  # Regularization parameter.
        'kernel': ['linear', 'rbf', 'poly'],  # Specifies the kernel type.
        'gamma': ['scale', 'auto'],  # Kernel coefficient for 'rbf' and 'poly'.
        'degree': [2, 3]  # Degree for 'poly' kernel. Only used if kernel is 'poly'.
        }
    }
}

# Define classifiers for cross-validation
MODELS_FOR_CV = {
    'lr': LogisticRegression(),
    'rf': RandomForestClassifier(),
    'svc': SVC(probability=True ),
    'nn': MLPClassifier(max_iter=1000, )
}

def load_and_preprocess_data(filepath, returndata=True):
    """Loads and preprocesses the raw data."""
    print("Step 1: Loading and preprocessing data...")
    data = pd.read_csv(filepath)
    logging.info(f"Reading file {filepath}")
    
    # Label encode targets
    encoders = {}
    for col_name, new_col in zip(["Plant family", "Plant genus", "Plant order"], TARGETS.values()):
        le = LabelEncoder()
        data[new_col] = le.fit_transform(data[col_name])
        encoders[new_col] = le
    
    if not returndata:
        return encoders
    
    # Feature selection
    features = data.drop(COLUMNS_TO_DROP + list(TARGETS.values()), axis=1)
    print(f"\tOriginal feature shape: {features.shape}")
    logging.info(f"Original feature shape: {features.shape} \t columns:{features.columns}")
    
    features = jaccard(features)
    print(f"Shape after jaccard: {features.shape}")
    logging.info(f"\tShape after jaccard: {features.shape} \t columns:{features.columns}")
    
    features = corr_sel(features)
    print(f"Shape after corr_sel: {features.shape}")
    logging.info(f"\tShape after corr_sel: {features.shape} \t columns:{features.columns}")
    
    features = near_constant(features)
    print(f"Shape after near_constant: {features.shape}")
    logging.info(f"\tShape after near_constant: {features.shape} \t columns:{features.columns}")

    # Combine processed features and targets
    processed_data = pd.concat([features, data[list(TARGETS.values())]], axis=1)
    print("Preprocessing complete.\n")
    return processed_data, encoders

def calculate_and_save_feature_importances(model, model_name, X_test, y_test, target_name, RUN_NUM):
    """Calculates, prints, and saves feature importances for a given model."""
    print(f"  Calculating feature importances for {model_name}...")
    feature_names = X_test.columns
    importances_df = pd.DataFrame({'feature': feature_names})

    # 1. Built-in Importance (if available)
    if hasattr(model, 'feature_importances_'):
        importances_df['importance_gini'] = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For multi-class, take the average of absolute coefficient values
        if model.coef_.ndim > 1:
            importances_df['importance_coef'] = np.mean(np.abs(model.coef_), axis=0)
        else:
            importances_df['importance_coef'] = np.abs(model.coef_)

    # 2. Permutation Importance
    perm_result = permutation_importance(
        model, X_test, y_test, n_repeats=10, n_jobs=-1
    )
    importances_df['importance_permutation'] = perm_result.importances_mean

    # Sort by the most robust metric: permutation importance
    importances_df = importances_df.sort_values(by='importance_permutation', ascending=False).reset_index(drop=True)

    # Print top 10 features
    print(f"  Top 10 features for {model_name} (target: {target_name}):")
    print(importances_df.head(10).to_string(index=False))
    
    # Save to CSV
    importance_filename = OUTPUT_DIR / f"feature_importance_{target_name}_{model_name}_{RUN_NUM}.csv"
    importances_df.to_csv(importance_filename, index=False)
    print(f"  Saved full feature importance report to {importance_filename}\n")

def run_gridsearch_tuning(X, y,RUN_NUM):
    """
    Performs GridSearchCV, calculates feature importance, and returns predictions.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, 
    random_state=42)  #added random state in this run only, it was not there in folder finalcode
    best_overall_acc = 0
    best_overall_model = ""
    all_predictions = {}

    print(f"--- Running GridSearchCV for target: {y.name} ---")
    for name, model_info in MODELS_FOR_GRIDSEARCH.items():
        print(f"Tuning {name}...")
        grid_search = GridSearchCV(
            model_info['estimator'], 
            model_info['params'], 
            cv=5, 
            n_jobs=-1, 
            scoring='accuracy'
        )
        grid_search.fit(X_train, y_train)
        
        preds = grid_search.predict(X_test)
        acc = accuracy_score(y_test, preds)
        all_predictions[name] = preds
        
        print(f"  Best Params: {grid_search.best_params_}")
        print(f"  Test Accuracy: {acc:.4f}")

        # --- NEW: Calculate and save feature importance ---
        best_estimator = grid_search.best_estimator_
        
        calculate_and_save_feature_importances(best_estimator, name, X_test, y_test, y.name, RUN_NUM)
        pickle.dump(best_estimator, open(f'{OUTPUT_DIR}/best_grid_model_{prefix}_{y.name}_{RUN_NUM}.sav','wb'))
        # -----------------------------------------------

        if acc > best_overall_acc:
            best_overall_acc = acc
            best_overall_model = name
            best_estimator_overall = best_estimator
            
    pickle.dump(best_estimator_overall, open(f'{OUTPUT_DIR}/best_estimator_overall_grid_model_{prefix}_{y.name}_{RUN_NUM}.sav','wb'))
    print(f"Best performing model for {y.name}: {best_overall_model} with accuracy {best_overall_acc:.4f}\n")
    logging.info(f"Best performing model for {y.name}: {best_overall_model} with accuracy {best_overall_acc:.4f}\n")
    logging.info(f"best estimator {best_estimator_overall}")
    return y_test, all_predictions, best_estimator_overall

def run_cross_validation(X, y, best_estimator):
    """
    Generates out-of-fold predictions using cross-validation for multiple models.
    """
    predictions = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, )
    
    print(f"--- Running Cross-Validation for target: {y.name} ---")
    preds = cross_val_predict(best_estimator, X, y, cv=cv, n_jobs=-1)
    predictions["best_estimator"] = preds
    print("Cross-validation complete.\n")
    return y, preds

def main():
    """Main script to run the full classification pipeline."""
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if os.path.exists(f"processed_{prefix}.csv"):
        processed_data = pd.read_csv(f"processed_{prefix}.csv")
        encoders = load_and_preprocess_data(INPUT_CSV, returndata=False)
    else:
        processed_data, encoders = load_and_preprocess_data(INPUT_CSV)
        processed_data.to_csv(f"processed_{prefix}.csv", index=False)
            
    results={}
    cv_results_df = pd.DataFrame()
    for RUN_NUM_i in range(5):
        RUN_NUM = f"{RUN_NUM_i}_{prefix}"
                
        # --- PART 1: GridSearchCV with Train/Test Split ---
        for target_key, target_col in TARGETS.items():
            X = processed_data.drop(columns=list(TARGETS.values()))
            y = processed_data[target_col]
            results[target_col] = {}
            
            
            ######################
            y_test, predictions, best_estimator = run_gridsearch_tuning(X, y, RUN_NUM)
            cm_df_list, report_df_list = [], []
            for RUN_NUM_i_CV in range(5):
                y_true, preds = run_cross_validation(X, y, best_estimator) #y_true, cv_predictions 
                print(f"best_estimator:{best_estimator}")
                
                logging.info(f"CV done for {RUN_NUM} cv num i:{RUN_NUM_i_CV}")
                estimator_name = best_estimator.__class__.__name__
                logging.info(f"best_estimator:{best_estimator} name:{estimator_name}")
                
                cv_results_df[f"target_{target_key}_{RUN_NUM_i}_{RUN_NUM_i_CV}"] = y_true
                cv_results_df[f"target_{target_key}_name_{RUN_NUM_i}_{RUN_NUM_i_CV}"] = y_true_names = encoders[target_col].inverse_transform(y_true)
                # for model_name, preds in cv_predictions.items():
                cv_results_df[f"pred_{target_key}_{estimator_name}_{RUN_NUM_i}_{RUN_NUM_i_CV}"] = preds
                cv_results_df[f"pred_{target_key}_{estimator_name}_name_{RUN_NUM_i}_{RUN_NUM_i_CV}"] = y_pred_names = encoders[target_col].inverse_transform(preds)
                
                ################# score calculation ############
                acc = accuracy_score(y_true, preds)
                f1 = f1_score(y_true, preds, average='macro')
                report = classification_report(y_true, preds, output_dict=True)

                results[target_col][RUN_NUM_i_CV] = {
                    'model name': estimator_name,
                    'accuracy': acc,
                    'f1_score': f1,
                    'classification_report': report
                }

                # Confusion matrix with taxonomic names
                labels = sorted(set(y_true_names) | set(y_pred_names))
                cm_named = confusion_matrix(y_true_names, y_pred_names, labels=labels)
                

                plt.figure(figsize=(8, 6))
                sns.heatmap(cm_named, annot=True, fmt="d", cmap="Blues",
                            xticklabels=labels, yticklabels=labels)
                plt.title(f'Confusion Matrix: {target_col.upper()} - {estimator_name.upper()}')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.tight_layout()

                plt.savefig(os.path.join(OUTPUT_DIR,   f'confusion_matrix_{target_col}_{RUN_NUM}_{RUN_NUM_i_CV}_{estimator_name}.png'))
                plt.close()
                cm_df = pd.DataFrame(cm_named, index=labels, columns=labels)
                cm_df_list.append(cm_df)
                
                # Save classification report
                report_df = pd.DataFrame(report).transpose()
                report_df_list.append(report_df)
                
            # all cm in one data
            cm_df_cv =  pd.concat(cm_df_list, axis=1)  
            cm_df_cv.index.name = 'Actual'
            cm_df_cv.columns.name = 'Predicted'
            cm_excel_path = os.path.join(OUTPUT_DIR, f'confusion_matrix_{RUN_NUM}_{target_col}_{estimator_name}.xlsx')
            cm_df_cv.to_excel(cm_excel_path)

            report_df_cv = pd.concat(report_df_list,axis=1)
            report_path = os.path.join(OUTPUT_DIR,  f'classification_report_{RUN_NUM}_{target_col}_{estimator_name}.xlsx')
            report_df_cv.to_excel(report_path)

        output_path_cv = OUTPUT_DIR / f"data_with_prediction_{RUN_NUM}_gridsearch_cv.csv"
        cv_results_df.to_csv(output_path_cv, index=False)
        print(f"Cross-validation prediction results saved to: {output_path_cv}")
        
            

if __name__ == "__main__":
    main()