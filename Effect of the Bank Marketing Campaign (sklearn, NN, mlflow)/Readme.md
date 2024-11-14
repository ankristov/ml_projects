# Model Comparison and Deployment Framework

This project is part of an experiment to compare models for binary classification tasks. The goal is to develop a framework for testing multiple models, logging results, and deploying them. Models considered include Naive Bayes, Logistic Regression, XGBoost, SVM, and a simple neural network.

## Initial Setup

Follow these steps to set up the environment. Ensure that **Conda** is installed, or use the Anaconda Prompt, which comes with the Anaconda installation.

### Steps

1. **Create the Conda environment**  
   Run the command below in the terminal to create a Conda environment and install Python 3.9 and `ipykernel` inside it:
   ```bash
   conda create --prefix /Users/andreikristov/Documents/python/conda_envs/git_mlflow_demo python=3.9 ipykernel
   ```
2. **Create a symbolic link for the environment**
    This step allows using a short environment name by linking it to the default Conda environment directory:

    ```bash
    ln -s /Users/andreikristov/Documents/python/conda_envs/git_mlflow_demo /opt/anaconda3/envs/git_mlflow_demo
    ```
3. **Activate the environment**
    To activate the environment, use one of these commands:
    
    ```bash
    conda activate "/Users/andreikristov/Documents/python/conda_envs/git_mlflow_demo"
    ```
    Or, if it's located in the default envs_dir, use:
    ```bash
    conda activate git_mlflow_demo
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
    "/Users/andreikristov/Documents/python/conda_envs/git_mlflow_demo/bin/pip" install imblearn
    ```

6. **Add the environment to Jupyter Notebook**
    Register the new environment as a kernel in Jupyter Notebook:
    ```bash
    python -m ipykernel install --user --name=git_mlflow_demo
    ```
7. **Launch Jupyter Notebook**
    Run the command below to open Jupyter Notebook (inside the activated Conda environment):
    ```bash
    jupyter notebook
    ```



