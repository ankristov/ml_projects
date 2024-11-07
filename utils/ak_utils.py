import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os


def plot_decision_boundary(model, X, y):
    """
    Plot decision boundary for model given two dimensional input X and labels y (real or predicted)

    Parameters
    ----------
    model: a model decision boundary of which we want to draw, must have 'predict' method
    X: pd.DataFrame with two features, an input of the data model
    y: int or object or category labels (real or predicted) used for coloring scatterplot
    """
    
    assert isinstance(X, pd.DataFrame), "X has to be an instance of pandas DataFrame."
    assert X.shape[1] == 2, "X has to have shape (, 2)"
    assert hasattr(model, "predict"), "The provided model doesn't have a predict method."
    
    # filter outliers: for that combine X and y into df (to filter y in parallel), filter by X, separate X and y
    df = pd.concat([X, pd.Series(y, name='y')], axis=1)
    q01_x1 = np.quantile(X.iloc[:, 0], 0.01)
    q99_x1 = np.quantile(X.iloc[:, 0], 0.99)
    q01_x2 = np.quantile(X.iloc[:, 1], 0.05)
    q99_x2 = np.quantile(X.iloc[:, 1], 0.99)
    df = df[(df.iloc[:, 0] > q01_x1) & (df.iloc[:, 0] < q99_x1) & (df.iloc[:, 1] > q01_x2) & (df.iloc[:, 1] < q99_x2)]
    X, y = df.drop(columns=['y']), df['y']

    # Set min and max values and give it some padditng
    x1_min, x1_max = X.iloc[:, 0].min(), X.iloc[:, 0].max()
    x2_min, x2_max = X.iloc[:, 1].min(), X.iloc[:, 1].max()
    #h = 0.01
    hn = 100

    print('x1_min, x1_max: ', x1_min, x1_max)
    print('x2_min, x2_max: ', x2_min, x2_max)
    # Generate a grid of points with distance h between them
    #xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, hn), np.linspace(x2_min, x2_max, hn))
    # Predict the function value for the whole grid
    Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
    print(Z)
    if not pd.api.types.is_any_real_numeric_dtype(Z[0]):
        print('Z (before): ', Z[:5])
        print('Z not numeric')
        Z = LabelEncoder().fit_transform(Z)
        print('Z (after): ', Z[:5])
    Z = Z.reshape(xx1.shape)
    
    # Plot the contour and training examples
    fig, (ax1, ax2) = plt.subplots(figsize=(20,10), ncols=2)
    ax1.contour(xx1, xx2, Z, cmap=plt.cm.Spectral)

    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=y, palette='Set2', ax=ax1)

    ax = fig.add_subplot(122, projection='3d')
    ax.plot_surface(xx1, xx2, Z, cmap=plt.cm.Spectral)

    ax1.set_xlabel(X.columns[0])
    ax1.set_ylabel(X.columns[1])
    ax2.set_xlabel(X.columns[0])
    ax2.set_ylabel(X.columns[1])

    # ax1.set_xlim(x1_min, x1_max)
    # ax1.set_ylim(x2_min, x2_max)
    # ax2.set_xlim(x1_min, x1_max)
    # ax2.set_ylim(x2_min, x2_max)

    #ax1.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    plt.show()


def search_files_in_directory(directory, extensions, search_string):
    """
    Search files in the directry (and subdirectories) containing search_string

    Parameters
    ----------
    directory : string
        Path to the directory
    extensions : _type_
        Files extensions to search in.
    search_string : _type_
        String to search in files

    Returns
    -------
    _type_
        _description_
    """
    matching_files = []
    
    # Walk through all subdirectories and files in the specified directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file has one of the desired extensions
            if file.endswith(tuple(extensions)):
                file_path = os.path.join(root, file)
                
                # Open the file and search for the string
                with open(file_path, 'r', errors='ignore') as f:
                    content = f.read()
                    if search_string in content.lower():
                        matching_files.append(file_path)

    return matching_files

# Set the directory to search and the file extensions to look for
directory_to_search = "/Users/andreikristov/Desktop/recover"
extensions_to_search = ['.txt', '.py', '.ipynb', '.html']
search_string = "mlflow"

# Call the function
result_files = search_files_in_directory(directory_to_search, extensions_to_search, search_string)

# Print the result
if result_files:
    print("Found 'xgboost' in the following files:")
    for file in result_files:
        print(file)
else:
    print("No files containing 'xgboost' were found.")