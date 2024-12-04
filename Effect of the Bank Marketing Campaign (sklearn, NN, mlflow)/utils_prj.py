
import os

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from boruta import BorutaPy
from keras import Sequential
from keras.layers import Dense
from mlflow import MlflowClient
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from utils_prj import *


def load_data(path):
    data = pd.read_csv(path)
    return data

def clean_data(df, verbose=False):
    """
    _summary_

    Parameters
    ----------
    data : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    if verbose:
        print('*** data_cleaning ***')
        print("NA summary: \n")
        print(df.isna().sum())
        print("shape (before drop): ", df.shape)
    
    df = df.dropna()
    
    if verbose:
        print("after droping na values \n")
        print(df.isna().sum())
        print("shape (after drop): ", df.shape)

    return df


def preprocess_dataset(df, verbose=False):
    """
    _summary_

    Parameters
    ----------
    df : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    # 1. replace values 
    #   education: basic.4y, basic.6y, basic.9y to basic; '.' in the name to '_'
    #   job: 'admin.' -> 'admin'
    df['education'] = df['education'].replace(r'^basic.*', 'basic', regex=True).str.replace('.','_', regex=False)
    df['job'] = df['job'].str.replace('.','')

    features_cat = df.select_dtypes(include='object').columns.tolist()
    features_num = df.select_dtypes(include='number').columns.tolist()

    # 2. preprocess categorical features
    df_cat_preprocessed = pd.get_dummies(data=df[features_cat], dtype=int)

    # 3. preprocess numerical features (reserved)
    df_num_preprocessed = df[features_num] # placeholder, replace with meaningful processing
    
    # 4. combine preprocessed features
    df_preprocessed = pd.concat([df_num_preprocessed, df_cat_preprocessed], axis=1)
    
    # 5. fix column names (lower case, w/o [. ,])
    df_preprocessed.columns = df_preprocessed.columns.str.replace(r'[\.\s]','_', regex=True).str.lower()
    
    if verbose:
        print('*** preprocess_dataset ***')

    return df_preprocessed


def oversample_dataset(X, y, how='SMOT', verbose=False):
    """
    _summary_

    Parameters
    ----------
    X : _type_
        _description_
    y : _type_
        _description_
    how : str, optional
        _description_, by default 'SMOT'
    verbose : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    from imblearn.over_sampling import SMOTE, SMOTENC
    
    assert how in ['SMOTE', 'SMOTENC'], "how MUST be one of ('SMOTE', 'SMOTENC)"

    features_cat = X.select_dtypes(include='object').columns.tolist()

    if how == 'SMOTE':
        smote = SMOTE(sampling_strategy=0.5, k_neighbors=5, random_state=42)
        X_oversampled, y_oversampled = smote.fit_resample(X, y)
    elif how == 'SMOTENC':
        smote_cat = SMOTENC(categorical_features=features_cat, sampling_strategy=0.5, k_neighbors=5, random_state=42)
        X_oversampled, y_oversampled = smote_cat.fit_resample(X, y) 

    X_oversampled = pd.DataFrame(data=X_oversampled, columns=X.columns)
    y_oversampled = pd.DataFrame(data=y_oversampled, columns=['y'])
    
    if verbose:
        print('*** oversample_dataset ***')
        print("N samples (before): ",len(X))
        print("N samples (after): ", len(X_oversampled))
        print('y value counts: \n', y.value_counts())
        print('y (over sampled) value counts: \n', y_oversampled.value_counts())
 
    return X_oversampled, y_oversampled


def train_model(model, X, y, params, verbose=False):
    """
    _summary_

    Parameters
    ----------
    X_train : _type_
        _description_
    y_train : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    assert hasattr(model, "fit"), "Model MUST have 'fit' method."
    
    if params:
        model = model.set_params(**params)
    
    model.fit(X, y)
    
    if verbose:
        print('*** train_model ***')
    
    return model


def predict_model(model, X, y=None, threshold=None, verbose=False):
    """
    Predicts binary classification results based on a given threshold.

    Parameters
    ----------
    model : model object
        A trained model that has a 'predict' method, and optionally a 'predict_proba' method.
    X : array-like
        Input data for predictions.
    y : array-like, optional
        True labels for computing the optimal threshold, by default None.
    threshold : float, str, or None, optional
        Threshold for classifying probabilities. Options:
        - None (default): uses 0.5 as threshold
        - float in range [0, 1]: uses provided threshold
        - 'optimal': computes the optimal threshold based on `y`
    verbose : bool, optional
        If True, prints additional debug information, by default False.

    Returns
    -------
    tuple
        y_pred : array-like
            Predicted class labels.
        y_pred_proba : array-like or None
            Predicted probabilities if `predict_proba` is available, otherwise None.
        threshold_optimal : float or None
            Computed optimal threshold if `threshold='optimal'` and `y` is provided, otherwise None.
    """

    assert hasattr(model, "predict"), "Model MUST have 'predict' method."

    y_pred = model.predict(X)
    y_pred_proba = None
    threshold_optimal = None

    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X)[:,1]
        
        if y is not None:
            threshold_optimal, _, _ = get_optimal_threshold(y_true=y, y_pred_proba=y_pred_proba, verbose=verbose)

    
        if isinstance(threshold, (int, float)) and 0.0 <= threshold <= 1.0:
            #y_pred_proba = model.predict_proba(X)[:,1]
            y_pred = (y_pred_proba > threshold).astype(int)
        elif threshold == 'optimal':
            if threshold_optimal is None:
                raise ValueError("To use threshold='optimal', `y` must be provided.")
            y_pred = (y_pred_proba > threshold_optimal).astype(int)
        elif threshold is None:
            threshold = 0.5
            y_pred = (y_pred_proba > threshold).astype(int)
        else:
            ValueError("threshold MUST be one of: None, float in the range [0,1], 'optimal'. If None - default 0.5.")


    if verbose:
        print('*** predict_model ***')
        print("predictions: \n", y_pred[:5])
        print("predictions (proba): \n", y_pred_proba[:5] if y_pred_proba is not None else None)
        print('threshold: ', threshold)
    
    return y_pred, y_pred_proba, threshold_optimal


def get_metrics(y_true, y_pred, y_pred_proba=None, verbose=True):
    """
    _summary_

    Parameters
    ----------
    y_true : _type_
        _description_
    y_pred : _type_
        _description_
    y_pred_proba : _type_, optional
        _description_, by default None
    verbose : bool, optional
        _description_, by default True

    Returns
    -------
    _type_
        _description_
    """

    from sklearn.metrics import (cohen_kappa_score, confusion_matrix, f1_score,
                                 matthews_corrcoef, precision_score,
                                 recall_score, roc_auc_score)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    metrics = {
        'accuracy': round(acc, 2), 
        'precision': round(prec, 2), 
        'recall': round(recall, 2),
        'f1_score': round(f1, 2),
        'matthews_corrcoef': round(mcc, 2),
        'kappa': round(kappa, 2)
        }
    
    # metrics from confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['tn'] = tn
    metrics['fp'] = fp
    metrics['fn'] = fn
    metrics['tp'] = tp
    # accuracy per class
    # metrics['accuracy_p'] = np.divide(tp, (tp + fn), where=(tp + fn) != 0, out=np.nan)
    # metrics['accuracy_n'] = np.divide(tn, (tn + fp), where=(tn + fp) != 0, out=np.nan)
    metrics['accuracy_p'] = np.nan if (tp + fn) == 0 else tp / (tp + fn)
    metrics['accuracy_n'] = np.nan if (tn + fp) == 0 else tn / (tn + fp)
    
    # Calculate AUC only if probabilities are provided
    if y_pred_proba is not None:
        auc = roc_auc_score(y_true, y_pred_proba)
        metrics['auc'] = round(auc, 2)
        # manual (for comparison)
        tpr, fpr, thr = prep_roc_auc(y_true, y_pred_proba, verbose=verbose)
        roc_auc_manual = calculate_auc(fpr, tpr)
        metrics['auc_manual'] = round(roc_auc_manual, 2)
    
    if verbose:
        print('*** get_metrics ***')
        for k, v in metrics.items():
            print(f"{k}: {v}")

    return metrics

    
def get_optimal_threshold(y_true, y_pred_proba, verbose=False):
    
    tpr, fpr, thr = prep_roc_auc(y_true, y_pred_proba, verbose=verbose)
    
    # Calculate the Youden's J statistic (TPR - FPR)
    youdens_j = np.array(tpr) - np.array(fpr)
    
    # Find the index of the maximum Youden's J statistic
    optimal_idx = np.argmax(youdens_j)
    
    # Get the optimal threshold value
    optimal_threshold = thr[optimal_idx]
    
    if verbose:
        print(f"Optimal threshold: {optimal_threshold}")
        print(f"True Positive Rate (TPR) at optimal threshold: {tpr[optimal_idx]}")
        print(f"False Positive Rate (FPR) at optimal threshold: {fpr[optimal_idx]}")

    return optimal_threshold, tpr[optimal_idx], fpr[optimal_idx]

def prep_roc_auc(y_true, y_pred_proba, verbose=False):
    """
    _summary_

    Parameters
    ----------
    y : _type_
        _description_
    y_pred_prob : _type_
        _description_
    verbose : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """

    from sklearn.metrics import confusion_matrix
    
    thr_arr = np.arange(start=0, stop=1.1, step=0.1)
    tpr_arr = []
    fpr_arr = []

    for thr in thr_arr:
        if thr == 0:
            tpr_arr.append(1)
            fpr_arr.append(1)
        elif thr ==1:
            tpr_arr.append(0)
            fpr_arr.append(0)
        else:
        #print(thr)
            df = pd.DataFrame({'y': y_true, 'y_pred_prob': y_pred_proba})
            df['y_pred'] = np.where(df['y_pred_prob'] > thr, 1, 0)
            df['is_match'] = df['y'] == df['y_pred']
            cm = confusion_matrix(df['y'], df['y_pred'])
            tp = cm[1,1]
            fp = cm[0,1]
            tn = cm[0,0]
            fn = cm[1,0]
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)
            #print('tpr: ', tpr)
            #print('fpr: ', fpr)
            tpr_arr.append(tpr)
            fpr_arr.append(fpr)
            #print(cm)

    if verbose:
        print('*** prep_roc_auc ***')

    return tpr_arr, fpr_arr, thr_arr

def calculate_auc(fpr, tpr, verbose=False):
    """
    _summary_

    Parameters
    ----------
    fpr : _type_
        _description_
    tpr : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    # insure range [0,1] for correct caculation of AUC score
    # if min(tpr) != 0:
    #     tpr.append(0)
    # if min(fpr) != 0:
    #     fpr.append(0)
    # if max(tpr) != 1:
    #     tpr.append(1)
    # if max(fpr) != 1:
    #     fpr.append(1)

    auc = 0.0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2

    auc = np.abs(auc)

    if verbose:
        print('*** calculate_auc ***')
        print('auc: ', auc)

    return auc

def plot_roc_auc(y, y_pred_proba, type='manual', ax=None, show=True, verbose=False):
    """
    _summary_

    Parameters
    ----------
    y : _type_
        _description_
    y_pred_prob : _type_
        _description_
    type : str, optional
        Prescribes how to calculate tpr, fpr: manually or use sklearn functionality. One of the ('manual', 'sklearn')
    verbose : bool, optional
        _description_, by default False
    """

    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    if type == 'manual':
        tpr, fpr, thr = prep_roc_auc(y, y_pred_proba, verbose=verbose)
    elif type == 'sklearn':
        fpr, tpr, thr = roc_curve(y, y_pred_proba, drop_intermediate=False)

    result_df = pd.DataFrame({'thr': thr, 'tpr': tpr, 'fpr': fpr}).sort_values(by='thr')

    threshold_optimal, tpr_optimal, fpr_optimal = get_optimal_threshold(y_true=y, y_pred_proba=y_pred_proba, verbose=verbose)

    #roc_auc = auc(fpr, tpr) # give not correct value
    roc_auc_manual = calculate_auc(fpr, tpr)

    if verbose:
        print('*** create_roc_auc_plot ***')
        #print(f"AUC Score: {roc_auc:.2f}")
        print(f"AUC Score (manual): {roc_auc_manual:.2f}")
        print(result_df)

    if ax is not None:
        ax=ax
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(x='fpr', y='tpr', data=result_df, color='blue', label=f'ROC curve (AUC = {roc_auc_manual:.2f})', ax=ax)
    sns.lineplot(x=[0, 1], y=[0, 1], color='gray', linestyle='--', ax=ax)  # Diagonal line representing random chance
    sns.scatterplot(x=[fpr_optimal], y=[tpr_optimal], color='green', s=100, label='Optimal Point', ax=ax)
    ax.axvline(x=fpr_optimal, color='grey', ls=':')
    ax.annotate(text=f'optimal thr: {str(threshold_optimal)}', xy=(fpr_optimal, tpr_optimal),xytext=(fpr_optimal+0.01, tpr_optimal-0.02), fontsize=8)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('FP Rate', fontsize=10)
    ax.set_ylabel('TP Rate', fontsize=10)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=12)
    ax.tick_params(axis='x', labelsize=8)  # Set x-axis tick labels size
    ax.tick_params(axis='y', labelsize=8)
    ax.legend(loc="lower right")
    plt.grid()
    plt.savefig('roc_auc_curve.png', facecolor='white', edgecolor='white')
    
    if show:
        plt.show()
    else:
        plt.close()
    

def plot_confusion_matrix(y, y_pred, ax=None, show=True, verbose=False):
    """
    _summary_

    Parameters
    ----------
    y : _type_
        _description_
    y_pred : _type_
        _description_
    """

    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    # 6. Generate the confusion matrix
    cm = confusion_matrix(y, y_pred)

    if verbose:
        print('*** create_confusion_matrix_plot ***')
        print('confusion matrix: \n', cm)

    # 7. Plot the confusion matrix using seaborn heatmap
    if ax is not None:
        ax=ax
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'], ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.legend()
    plt.savefig('confusion_matrix.png', facecolor='white', edgecolor='white')
    
    if show:
        plt.show()
    else:
        plt.close()


def get_feature_importance_rf(X, y, n_estimators=100, random_state=42, verbose=False):
    """
    Calculate feature importance using a Random Forest model.
    
    Parameters:
        X (pd.DataFrame): The input features.
        y (pd.Series): The target variable.
        n_estimators (int): Number of trees in the forest.
        random_state (int): Random state for reproducibility.
        
    Returns:
        pd.DataFrame: DataFrame containing features and their importance scores.
    """
    # Initialize and fit the RandomForest model
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    
    rf_model.fit(X.values, y.values.ravel())
    
    # Extract feature importances
    importances = rf_model.feature_importances_
    feature_imp_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    })
    
    if verbose:
        print('*** get_feature_importance_boruta ***')
        print('Feature importances: \n', feature_imp_df)

    # Sort by importance and return
    return feature_imp_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)


def get_feature_importance_boruta(X, y, max_iter=100, random_state=42, verbose=False):
    """
    Calculate feature importance using the Boruta method.
    
    Parameters:
        X (pd.DataFrame): The input features.
        y (pd.Series): The target variable.
        max_iter (int): Maximum number of iterations for the Boruta algorithm.
        random_state (int): Random state for reproducibility.
        
    Returns:
        pd.DataFrame: DataFrame containing features and their Boruta importance status.
    """
    # Initialize a RandomForestClassifier for use with Boruta
    rf = RandomForestClassifier(n_jobs=-1, max_depth=5, random_state=random_state)

    # Initialize Boruta feature selector
    boruta_selector = BorutaPy(
        estimator=rf,
        n_estimators='auto',
        max_iter=max_iter,
        random_state=random_state
    )
    
    # Fit the Boruta selector to the data
    boruta_selector.fit(X.values, y.values.ravel())
    
    # Get results and create a DataFrame
    feature_status = boruta_selector.support_  # True if the feature is selected
    feature_ranking = boruta_selector.ranking_  # Ranking of features
    feature_imp_df = pd.DataFrame({
        'Feature': X.columns,
        'Selected': feature_status,
        'Importance': feature_ranking
    })

    if verbose:
        print('*** get_feature_importance_boruta ***')
        print('Feature importances: \n', feature_imp_df)
    
    # Sort by ranking and return
    return feature_imp_df.query('Selected == True').sort_values(by='Importance')


def plot_feature_importances(X, y, verbose=False):
    """
    _summary_

    Parameters
    ----------
    df_feature_imp : _type_
        A dataframe with feature importances
    method : str, optional
        Method to use for calculating feature importance, by default 'rf'
    varbose : bool, optional
        _description_, by default True
    """

    df_feature_imp_rf = get_feature_importance_rf(X, y)
    df_feature_imp_boruta = get_feature_importance_boruta(X, y)

    limit = 15
    n_features_vis = max(limit, len(df_feature_imp_boruta))
    if len(df_feature_imp_rf) > n_features_vis:
            df_feature_imp_rf = df_feature_imp_rf.head(n_features_vis)

    if verbose:
        print('*** plot_feature_importance ***')
        
    
    _, (ax1, ax2) = plt.subplots(figsize=(8, 0.2 * n_features_vis), ncols=2)

    sns.barplot(y='Feature', x='Importance', data=df_feature_imp_rf, color='steelblue', ax=ax1)
    sns.barplot(y='Feature', x='Importance', color='steelblue', data=df_feature_imp_boruta, ax=ax2)

    ax1.set_title('Feature Importance (RF)', fontsize=12)
    ax1.set_xlabel('Importance', fontsize=10)
    ax1.set_ylabel('Feature', fontsize=10)
    ax1.tick_params(axis='x', labelsize=8)  # Set font size for x-axis ticks
    ax1.tick_params(axis='y', labelsize=8)  # Set font size for y-axis ticks

    ax2.set_title('Feature Importance (Boruta)', fontsize=12)
    ax2.set_xlabel('Importance', fontsize=10)
    ax2.set_ylabel('Feature', fontsize=10)
    ax2.tick_params(axis='x', labelsize=8)  # Set font size for x-axis ticks
    ax2.tick_params(axis='y', labelsize=8)  # Set font size for y-axis ticks

    plt.tight_layout()

    plt.savefig('feature_importances.png')
    plt.show()

    

def tune_model(model, X, y, param_grid, scoring='accuracy', verbose=False):
    """
    _summary_

    Parameters
    ----------
    model : _type_
        _description_
    X : _type_
        _description_
    y : _type_
        _description_
    param_grid : _type_
        _description_
    scoring : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    from sklearn.model_selection import RandomizedSearchCV

    # cast y as array
    y = y.values.ravel()

    # set number of iterations for random search (not more than 100)
    n_iter_total = 1
    n_iter_limit = 100
    for k, v in param_grid.items():
        n_iter_total *= len(v)
    n_iter = min(n_iter_limit, n_iter_total)
    
    rs = RandomizedSearchCV(
        estimator = model, 
        param_distributions = param_grid, 
        n_iter = n_iter,
        scoring=scoring,
        cv = 5, 
        verbose=False, 
        random_state=35, 
        n_jobs = -1)
    
    rs.fit(X, y)

    best_params = rs.best_params_
    
    # fit model with best params
    model_tuned = model.set_params(**best_params) 
    model_tuned.fit(X, y)

    if verbose:
        print('*** tune_model ***')
        print("# of total param combinations: ", n_iter_total)
        print("# of iterations: ", n_iter)
        print ('Param grid: \n', param_grid, '\n')
        # print the best parameters
        print ('Best Parameters: \n', best_params, ' \n')
    
    return model_tuned, best_params


def log_experiment(experiment_name, model, input_example, run_name, run_metrics, run_params=None, run_tags=None, run_artifacts_png=None, verbose=False):
    """
    _summary_

    Parameters
    ----------
    experiment_name : _type_
        _description_
    model : _type_
        _description_
    input_example : _type_
        _description_
    run_name : _type_
        _description_
    run_metrics : _type_
        _description_
    run_params : _type_, optional
        _description_, by default None
    run_tags : _type_, optional
        _description_, by default None
    run_artifacts_png : _type_, optional
        _description_, by default None
    """

    from keras.models import Model
    from sklearn.base import BaseEstimator

    client = MlflowClient()

    try:
        client.get_experiment_by_name(experiment_name)
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        if verbose:
            print(f"Experiment not found: {e}")
        client.create_experiment(experiment_name)
        client.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        # log metrics
        for m in run_metrics:
            mlflow.log_metric(m, run_metrics[m])
        # log model
        # Determine model type and log appropriately
        if isinstance(model, BaseEstimator):  # sklearn model
            mlflow.sklearn.log_model(sk_model=model, artifact_path='model', input_example=input_example)
            # Add any other metrics, params logging specific to sklearn models if needed
        elif isinstance(model, Model):  # Keras model
            mlflow.keras.log_model(model=model, artifact_path="model")

        # mlflow.sklearn.log_model(model=model, artifact_path='model', input_example=input_example)
        # log params
        if run_params:
            for p in run_params.items():
                mlflow.log_param(key=p[0], value=p[1])
        # log tags
        if run_tags:
            for t in run_tags.items():
                mlflow.set_tag(key=t[0], value=t[1])
        # log artifacts
        if run_artifacts_png:
            for a in run_artifacts_png:
                if os.path.exists(a):
                    mlflow.log_artifact(a)

    if verbose:
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        experiment = mlflow.get_experiment(experiment_id)
        # print('*** log_experiment ***')   
        # print('Run - is logged.')
        # # Get the run info
        # print(f'experiment name: {experiment.name}')
        # print(f'experiment id: {experiment.experiment_id}')
        # print('run_name: ', run.info.run_name)
        # print('run_id: ', run_id)
        print(f'experiment_id: {experiment.experiment_id}, run_id: {run_id}')
    

def plot_decision_boundary(model, X, y, remove_outliers=False, show=True, verbose=True):
    """
    Plot decision boundary for model given two dimensional input X and labels y (real or predicted)

    Parameters
    ----------
    model: a model decision boundary of which we want to draw, must have 'predict' method
    X: pd.DataFrame with two features, an input of the data model
    y: int or object or category labels (real or predicted) used for coloring scatterplot
    """
    
    assert isinstance(X, pd.DataFrame), "X has to be an instance of pandas DataFrame."
    assert X.shape[1] > 1, "X has to have at least 2 features."
    assert hasattr(model, "predict"), "The provided model doesn't have a predict method."
    
    if remove_outliers:
        # filter outliers: for that combine X and y into df (to filter y in parallel), filter by X, separate X and y
        df = pd.concat([X, pd.Series(y, name='y')], axis=1)
        q01_x1 = np.quantile(X.iloc[:, 0], 0.01)
        q99_x1 = np.quantile(X.iloc[:, 0], 0.99)
        q01_x2 = np.quantile(X.iloc[:, 1], 0.05)
        q99_x2 = np.quantile(X.iloc[:, 1], 0.99)
        df = df[(df.iloc[:, 0] > q01_x1) & (df.iloc[:, 0] < q99_x1) & (df.iloc[:, 1] > q01_x2) & (df.iloc[:, 1] < q99_x2)]
        X, y = df.drop(columns=['y']), df['y']

    hn = 100 # parameter grid coef = number of point on the whole range of the variable with equal space between them
    if X.shape[1] > 2:
        # Step 1: Reduce the dimensionality to 2D for visualization using PCA
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
        # Set min and max values and give it some padditng
        x1_min, x1_max = X_reduced[:, 0].min(), X_reduced[:, 0].max()
        x2_min, x2_max = X_reduced[:, 1].min(), X_reduced[:, 1].max()
        # Generate a grid of points with distance h between them
        xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, hn), np.linspace(x2_min, x2_max, hn))
        # Predict the function value for the whole grid
        Z = model.predict(pca.inverse_transform(np.c_[xx1.ravel(), xx2.ravel()]))
    elif X.shape[1] == 2:
        x1_min, x1_max = X.iloc[:, 0].min(), X.iloc[:, 0].max()
        x2_min, x2_max = X.iloc[:, 1].min(), X.iloc[:, 1].max()
        # Generate a grid of points with distance h between them
        xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, hn), np.linspace(x2_min, x2_max, hn))
        # Predict the function value for the whole grid
        Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
        
    Z = Z.reshape(xx1.shape)

    if verbose:
        print('*** plot_decision_boundary ***')
        print('x1_min, x1_max: ', x1_min, x1_max)
        print('x2_min, x2_max: ', x2_min, x2_max)
    
    # Plot the contour and training examples
    fig, (ax1, ax2) = plt.subplots(figsize=(20,10), ncols=2)
    ax1.contour(xx1, xx2, Z, cmap=plt.cm.Spectral)

    if X.shape[1] > 2:
        sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=y, palette='Set2', ax=ax1)
    else:
        sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=y, palette='Set2', ax=ax1)

    ax = fig.add_subplot(122, projection='3d')
    ax.plot_surface(xx1, xx2, Z, cmap=plt.cm.Spectral)

    ax1.set_xlabel(X.columns[0])
    ax1.set_ylabel(X.columns[1])
    ax2.set_xlabel(X.columns[0])
    ax2.set_ylabel(X.columns[1])

    plt.tight_layout()

    plt.savefig('decision_boundary.png', facecolor='white', edgecolor='white')
    
    if show:
        plt.show()
    else:
        plt.close()


def get_files_list(directory, extensions):
    """
    List all files in the specified directory (and subdirectories) with the specified extension.
    
    Args:
        directory (str): The path to the directory to search in.
        extension (str): The file extension to search for (e.g., '.txt').

    Returns:
        list: A list of paths to files with the specified extension.
    """
    files_with_extension = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(tuple(extensions)):
                files_with_extension.append(os.path.join(root, file))
    return files_with_extension


def delete_files(directory, extensions, verbose=False):
    """
    Delete all files in the provided list of file paths.
    
    Args:
        file_list (list): A list of file paths to delete.
    """
    file_list = get_files_list(directory=directory, extensions=extensions)

    cnt = 0
    for file_path in file_list:
        try:
            os.remove(file_path)
            if verbose:
                print(f"Deleted: {file_path}")
            cnt += 1
        except OSError as e:
            print(f"Error deleting {file_path}: {e}")

    if verbose:
        print(f'Deleted {cnt} files from {len(file_list)}.')

def delete_png():
    return delete_files(directory=os.getcwd(), extensions=['.png'], verbose=False)


def convert_X_preprocessed_to_dataframe(X, preprocessor, columns_num, columns_cat):
    #X_train_preprocessed = preprocessor.fit_transform(X_train)
    columns_cat_preprocessed = preprocessor.transformers_[1][1].named_steps['encoder'].get_feature_names_out(columns_cat)
    columns_preprocessed = columns_num + list(columns_cat_preprocessed)
    return pd.DataFrame(X, columns=columns_preprocessed)


def visualize_artifacts():

    import matplotlib.image as mpimg

    if os.path.exists('roc_auc_curve.png') and os.path.exists('confusion_matrix.png'):
        img1 = mpimg.imread('roc_auc_curve.png')  
        img2 = mpimg.imread('confusion_matrix.png') 
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(12, 5))
        # Display images on each subplot
        ax1.imshow(img1)
        ax1.axis('off')  
        ax2.imshow(img2)
        ax2.axis('off')
        # Show the plot
        plt.tight_layout()
        plt.show()

    if os.path.exists('decision_boundary.png'):
        img3 = mpimg.imread('decision_boundary.png')  
        # Create a figure with two subplots
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 5))
        # Display images on each subplot
        ax.imshow(img3)
        ax.axis('off')  # Turn off axis for the first image
        # Show the plot
        plt.tight_layout()
        plt.show()

    if os.path.exists('metrics_vs_epochs.png'):
        img4 = mpimg.imread('metrics_vs_epochs.png')  
        # Create a figure with two subplots
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 5))
        # Display images on each subplot
        ax.imshow(img4)
        ax.axis('off')  # Turn off axis for the first image
        # Show the plot
        plt.tight_layout()
        plt.show()


def build_model_NN(input_shape, output_shape):
    """
    Base model takes as it's input - an vector of averaged values per words embeddings in a sentence,
    push them to one dense layer, and make prediction of the smile with softmax activation.
    """

    # Build the neural network model
    model = Sequential([
        Dense(64, input_dim=input_shape, activation='relu'),  # First hidden layer (64 neurons)
        Dense(32, activation='relu'),  # Second hidden layer (32 neurons)
        Dense(output_shape, activation='sigmoid')  # Output layer (1 neuron for binary classification)
    ])

    return model


def predict_model_NN(model, X, y=None, threshold=None, verbose=False):
    """
    Get NN model predictions. When the target variable y is supplied and threshold='optimal' 
    calculates and returns the optimal value of the threshold.

    Parameters
    ----------
    model : _type_
        _description_
    X : _type_
        _description_
    y : np.array
        True values of the target. When not None is used for calculating the optimal threshold.
    threshold : _type_, optional
        _description_, by default None
    verbose : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """

    # Assertions
    assert hasattr(model, "predict"), "Model MUST have 'predict' method."

    # Get probabilities
    y_pred_proba = model.predict(X).ravel()
    y_pred = (y_pred_proba > 0.5).astype(int).ravel()
    threshold_optimal = None

    # Calculate optimal threshold if `y` is provided and `threshold='optimal'`
    if y is not None:
        threshold_optimal, _, _ = get_optimal_threshold(y_true=y, y_pred_proba=y_pred_proba, verbose=verbose)

    # Determine threshold application
    if isinstance(threshold, (int, float)) and 0.0 <= threshold <= 1.0:
        y_pred = (y_pred_proba > threshold).astype(int)
    elif threshold == 'optimal':
        if threshold_optimal is None:
            raise ValueError("To use threshold='optimal', `y` must be provided.")
        y_pred = (y_pred_proba > threshold_optimal).astype(int)
    elif threshold is None:
        threshold = 0.5
        y_pred = (y_pred_proba > threshold).astype(int)
    else:
        raise ValueError("threshold MUST be one of: None, float in the range [0,1], or 'optimal'. If None, default is 0.5.")

    # Verbose output
    if verbose:
        print('*** predict_model_NN ***')
        print("Predictions: \n", y_pred[:5])
        print("Predictions (proba): \n", y_pred_proba[:5])
        print("Threshold: ", threshold)

    return y_pred, y_pred_proba, threshold_optimal


def clean_history(history_df):

    history_df.columns = ['train_' + c if not c.startswith('val_') else c for c in history_df.columns]

    return history_df

def get_metrics_from_history(history_df):
    """
    Keep the result of last epoch.

    Parameters
    ----------
    history_df : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    metrics = {col:np.round(history_df[col].values[-1],4)  for col in history_df.columns}

    return metrics


def plot_metrics_vs_epoch(history_df, file_name, verbose=True):
    """
    Plot model fit metrics vs epoch.

    Parameters
    ----------
    model_history : _type_
        _description_
    """
    import re

    import matplotlib.pyplot as plt
    import seaborn as sns

    #history_df = pd.DataFrame(model_history.history)
    # p = re.compile(r'loss$')
    # cols_vis = list(filter(lambda x: not p.search(x), history_df.columns))
    
    cols_vis_accuracy = [c for c in history_df.columns if re.search('.*accuracy.*', c)]
    cols_vis_loss = [c for c in history_df.columns if re.search('.*loss.*', c)]

    # Create a figure and 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for c in cols_vis_accuracy:
        sns.lineplot(x=history_df.index, y=history_df[c], label=c, ax=ax1)

    for c in cols_vis_loss:
        sns.lineplot(x=history_df.index, y=history_df[c], label=c, ax=ax2)

    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    #ax1.legend(loc='upper left')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=4, fontsize=8)
    

    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')

    # Show the plot
    plt.tight_layout()

    plt.savefig(file_name)
    if verbose:
        plt.show()
    else:
        plt.close()


def plot_metrics_vs_epoch_2(history_df, metrics, file_name, verbose=True):
    """
    Plot model fit metrics vs epoch.

    Parameters
    ----------
    model_history : _type_
        _description_
    """
    import re

    import matplotlib.pyplot as plt
    import seaborn as sns

    #history_df = pd.DataFrame(model_history.history)
    # p = re.compile(r'loss$')
    # cols_vis = list(filter(lambda x: not p.search(x), history_df.columns))
    
    n_plots = len(metrics)
    ncols = 2
    nrows = n_plots // ncols
    colors = sns.color_palette("Set2", 10)
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12, 5*nrows))
    for i, m in enumerate(metrics): 
        cols_vis = [c for c in history_df.columns if re.search(f'.*{m}.*', c)]
        
        ax = axes[i//ncols, i%ncols]
        for j, c in enumerate(cols_vis):
            sns.lineplot(x=history_df.index, y=history_df[c], color=colors[j], label=c, ax=ax)

        ax.set_title(f'{m.capitalize()}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(f'{m.capitalize()}')
    #ax1.legend(loc='upper left')
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=4, fontsize=8)

    plt.suptitle('Fitting Metrics')
    # Show the plot
    plt.tight_layout()

    plt.savefig(file_name, facecolor='white', edgecolor='white')
    if verbose:
        plt.show()
    else:
        plt.close()