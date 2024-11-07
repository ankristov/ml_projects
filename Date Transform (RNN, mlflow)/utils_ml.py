import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from keras.optimizers import Adam
from mlflow import MlflowClient


def plot_metrics_vs_epoch(history_df, file_name, verbose):
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


def plot_model_history(history_df):
    # Set seaborn style
    #sns.set(style='whitegrid')

    # Create a figure and 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot accuracy on the first subplot (ax1)
    sns.lineplot(x=range(history_df.shape[0]), y=history_df['accuracy'], ax=ax1, label='Train')
    sns.lineplot(x=range(history_df.shape[0]), y=history_df['val_accuracy'], ax=ax1, label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='upper left')

    # Plot loss on the second subplot (ax2)
    sns.lineplot(x=range(history_df.shape[0]), y=history_df['loss'], ax=ax2, label='Train')
    sns.lineplot(x=range(history_df.shape[0]), y=history_df['val_loss'], ax=ax2, label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper left')

    # Show the plot
    plt.tight_layout()
    plt.show()


def get_metrics_from_history(history_df):
    
    metrics = {col:np.round(history_df[col].values[-1],4)  for col in history_df.columns}

    return metrics


def get_metrics_from_model(y_true, y_pred, verbose=False):
    """
    _summary_

    Parameters
    ----------
    y_true : _type_
        _description_
    y_pred : _type_
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
    print('*** get_metrics ***')

    acc = (y_true == y_pred).mean()

    metrics = {
        'test_accuracy': round(acc, 2), 
        }

    if verbose:
        print('metrics: ', metrics)

    return metrics


def log_experiment(experiment_name, run_name, model, input_example, run_metrics, run_params=None, run_tags=None, run_artifacts_png=None, verbose=False):
    """
    Loging of experiments.

    For logging keras model check this: https://medium.com/@rajavenkatesrajamanickam/mlflow-integration-with-keras-3309fd9fb6c9

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

    client = MlflowClient()

    try:
        client.get_experiment_by_name(experiment_name)
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        if verbose:
            print(f"Experiment not found: {e}")
        client.create_experiment(experiment_name)
        client.set_experiment(experiment_name)

    # mlflow.create_experiment(experiment_name)
    # mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        # log metrics
        for m in run_metrics:
            if run_metrics[m]:
                mlflow.log_metric(m, run_metrics[m])
        # log model
        #mlflow.keras.log_model(model, artifact_path='model', input_example=input_example)
        mlflow.keras.log_model(model, artifact_path='model')
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
        

        # Get the run id
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        experiment = mlflow.get_experiment(experiment_id)

        if verbose: 
            print('*** log_experiment ***')
            print('Run - is logged.')
            print(f'experiment name: {experiment.name}')
            print(f'experiment id: {experiment.experiment_id}')
            print('run_name: ', run.info.run_name)
            print('run_id: ', run_id)


def get_png():
    """
    Get the list of .png files in the current folder.

    Returns
    -------
    _type_
        _description_
    """

    import glob
    import os

    # Get the current directory
    current_dir = os.getcwd()

    # Find all .png files in the current directory
    png_files = glob.glob(os.path.join(current_dir, "*.png"))

    return png_files


def clean_png(verbose):
    """
    Clean all .png files in the current folder.
    """

    import glob
    import os

    # Get the current directory
    current_dir = os.getcwd()

    # Find all .png files in the current directory
    png_files = glob.glob(os.path.join(current_dir, "*.png"))

    # Loop through the list of .png files and delete them
    for file in png_files:
        try:
            os.remove(file)
            if verbose:
                print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")

    if verbose:
        print("All .png files have been deleted.")


def run_model(model, X_train, Y_train, X_test, Y_test, func_prepare_XY, experiment_name, exp_params, run_id, verbose=False):
    """
    _summary_

    Parameters
    ----------
    model : _type_
        _description_
    X : _type_
        _description_
    Y : _type_
        _description_
    experiment_name : _type_
        _description_
    run_id : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    clean_png(verbose=verbose)

    X_train_prep, Y_train_prep, X_test_prep, Y_test_prep = func_prepare_XY(X_train, Y_train, X_test, Y_test)

    optimizer_adam = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer_adam, metrics=['accuracy'])
    callback__early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    start_fit_time = time.time()
    history = model.fit(
        X_train_prep, 
        Y_train_prep, 
        epochs = exp_params['epochs'], 
        batch_size = exp_params['batch_size'], 
        shuffle=exp_params['shuffle'], 
        validation_split=0.2,
        callbacks=[callback__early_stopping],
        verbose=verbose)
    end_fit_time = time.time()

    history_df = pd.DataFrame(history.history)

    metrics = get_metrics_from_history(history_df=history_df)

    # plot model fit history
    plot_metrics_vs_epoch(history_df, file_name='model_fit_history.png', verbose=verbose)

    # evaluate
    loss_eval, accuracy_eval = model.evaluate(X_test_prep, Y_test_prep)
    

    # predict on test
    pred_test = model.predict(X_test_prep)
    pred_test = np.argmax(pred_test, axis=1)
    accuracy_test = (pred_test == Y_test).mean()

    # predict on train
    pred_train = model.predict(X_train_prep)
    pred_train = np.argmax(pred_train, axis=1)
    accuracy_train = (pred_train == Y_train).mean()

    if verbose:
        print('Evaluation: ')
        print('loss_eval, accuracy_eval: ', loss_eval, accuracy_eval)
        print('Predict on test: ')
        print('accuracy_test: ', accuracy_test)
        print('Predict on train: ')
        print('accuracy_train: ', accuracy_train)

    params = exp_params.update({'fitting_time_s': end_fit_time - start_fit_time}) 
    metrics.update({'accuracy_test': accuracy_test,
                    'accuracy_train': accuracy_train,
                    'accuracy_eval': accuracy_eval})

    log_experiment(
                experiment_name = experiment_name,  
                run_name = f"run__e{exp_params['epochs']}__r{run_id}",
                model = model, 
                run_metrics = metrics, 
                run_params = params,
                run_tags = {
                    'ts': datetime.now().strftime('%Y_%m_%d_%H%M%S')},
                input_example = X_train_prep[:100], 
                run_artifacts_png = get_png(),
                verbose=verbose)
    
    return accuracy_train, accuracy_test, accuracy_eval    return accuracy_train, accuracy_test, accuracy_eval