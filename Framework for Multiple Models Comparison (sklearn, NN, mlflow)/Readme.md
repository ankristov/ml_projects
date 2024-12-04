# Project Objective

**The goal**: To develop a framework for testing multiple models on the task of binary classification, including preparation steps for deploying the best-performing model.

The following models were included: Naive Bayes, Logistic Regression, XGBoost, SVM, and a simple neural network.

For each model, the following versions were evaluated:

- A basic model (without tuning) on the original dataset
- A basic model (without tuning) on the oversampled dataset (using SMOTENC)
- A tuned model (with grid parameter tuning) on the - A tuned model (with grid parameter tuning) on the oversampled dataset (using SMOTENC)

For each version of the model, metrics were calculated for the default threshold (0.5) and the optimal threshold (corresponding to the closest point on the ROC AUC curve to the upper-left corner).

Results and artifacts were logged with MLFlow.

The best-performing model was registered in the model registry and prepared for deployment.

## Results

Here is the summary of all model runs.

![Result of experiments on binary classification task](./mlflow_results_styled.png)

## Initial Setup

Follow these steps to set up the environment. Ensure that **Conda** is installed, or use the Anaconda Prompt, which comes with the Anaconda installation.

### Steps

1. **Create the Conda environment**  
   Run the command below in the terminal to create a Conda environment and install Python 3.9 and `ipykernel` inside it:
   ```bash
   conda create --prefix path/to/the/folder/env_ml/ python=3.9 ipykernel
   ```
   or (in default folder)
   ```bash
   conda create -n env_ml
   ```
2. **Create a symbolic link for the environment**
    This step allows using a short environment name by linking it to the default Conda environment directory (in case if custom path was used):

    ```bash
    ln -s /path/to/the/folder/with/conda/env_ml/ /opt/anaconda3/envs/env_ml/
    ```
3. **Activate the environment**
    To activate the environment, use one of these commands:
    
    ```bash
    conda activate "path/to/the/folder/env_ml/"
    ```
    Or, if it's located in the default envs_dir, use:
    ```bash
    conda activate env_ml
    ```
4. **Install packages**
    Run the following command to install the necessary packages:
    ```bash
    conda install python=3.9 ipykernel pandas numpy scikit-learn matplotlib seaborn mlflow notebook
    ```
5. **Install imblearn with pip**
    Use pip to install imblearn:
    
    ```bash
    pip install imblearn
    ```
    If this command doesn’t work, you may need to specify the absolute path to the pip executable from this environment:
    ```bash
    "path/to/the/folder/env_ml/bin/pip" install imblearn
    ```

6. **Add the environment to Jupyter Notebook**
    Register the new environment as a kernel in Jupyter Notebook:
    ```bash
    python -m ipykernel install --user --name=env_ml
    ```
7. **Launch Jupyter Notebook**
    Run the command below to open Jupyter Notebook (inside the activated Conda environment):
    ```bash
    jupyter notebook
    ```

### Initial Setups (mlflow)

1. **Before logging experiments**
    ```bash
    mlflow server --backend-store-uri /path/to/the/project/folder/mlruns  --default-artifact-root /path/to/the/project/folder/mlruns --port 5000
    ```
2. **To clean processes running on the port**
    ```bash
    lsof -i :5000
    kill -9 <PID1> <PID2> <PID3> ...
    ```

3. **This can be helpful if something is not running**
    ```python
    import os
    os.environ['MLFLOW_ARTIFACT_URI'] = './mlruns'
    #del os.environ['MLFLOW_ARTIFACT_URI']
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_tracking_uri("file:///path/to/the/project/folder/mlruns/mlruns")
    ```



